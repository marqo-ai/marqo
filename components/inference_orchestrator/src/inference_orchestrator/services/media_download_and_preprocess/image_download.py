import base64
import os
from io import BytesIO
from typing import Optional

import certifi
import pycurl
import requests
import validators
from PIL import Image, UnidentifiedImageError

from inference_orchestrator import marqo_docs
from inference_orchestrator.api.telemetry import RequestMetrics
from inference_orchestrator.core.settings import get_settings
from inference_orchestrator.services.errors import (
    ImageDownloadError,
    InternalServerError,
)

# TODO Merge this with the one in clip_utils in the future refactoring

settings = get_settings()

DEFAULT_HEADERS = {"User-Agent": "Marqobot/1.0"}


def get_allowed_image_types():
    return {".jpg", ".png", ".bmp", ".jpeg"}


def is_base64_image(s: str) -> bool:
    """
    Check if a string is a base64-encoded image.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is a base64-encoded image, False otherwise.
    """
    if not isinstance(s, str):
        return False

    # Check for data URL prefix
    if s.startswith("data:image/"):
        return True

    return False


def _load_base64_image(content: str) -> Image.Image:
    """
    Load a base64-encoded image string into a PIL Image.

    Args:
        content: Base64-encoded image string (with or without data URL prefix)

    Returns:
        ImageType: PIL Image object

    Raises:
        UnidentifiedImageError: If the content cannot be decoded or loaded as an image
    """
    _, _, b64data = content.partition("base64,")

    try:
        img_bytes = base64.b64decode(b64data)
    except ValueError as e:
        raise UnidentifiedImageError(f"Invalid base64 data: {e}")

    # Open and load directly from the in-memory buffer
    with BytesIO(img_bytes) as buf:
        img = Image.open(buf)
        img.load()

    return img


def load_image_from_path(
    image_path: str,
    media_download_headers: dict,
    timeout_ms=3000,
    metrics_obj: Optional[RequestMetrics] = None,
) -> Image.Image:
    """Loads an image into PIL from a string path that is either local or a url

    Args:
        image_path (str): Local or remote path to image, or base64-encoded image string.
        media_download_headers (dict): header for the image download
        timeout_ms (int): timeout (in milliseconds), for the whole request
    Raises:
        ValueError: If the local path is invalid, and is not a url
        UnidentifiedImageError: If the image is irretrievable or unprocessable.

    Returns:
        ImageType: In-memory PIL image.
    """
    # Check if it's a base64-encoded image first
    if is_base64_image(image_path):
        return _load_base64_image(image_path)

    if os.path.isfile(image_path):
        img = Image.open(image_path)
    elif validators.url(image_path):
        if metrics_obj is not None:
            metrics_obj.start(f"media_download.image.{image_path}")
        try:
            img_io: BytesIO = download_image_from_url(
                image_path, media_download_headers, timeout_ms
            )
            img = Image.open(img_io)
        except ImageDownloadError as e:
            raise UnidentifiedImageError(str(e)) from e
        except OSError as e:
            if "could not create decoder object" in str(e):
                raise UnidentifiedImageError(
                    f"Marqo encountered an error when downloading the image from {image_path}. "
                    f"The image could not be decoded properly. Original error: {e}"
                )
            else:
                raise e
        finally:
            if metrics_obj is not None:
                metrics_obj.stop(f"media_download.image.{image_path}")
    else:
        raise UnidentifiedImageError(
            f"Input str of {image_path} is not a local file, a valid url, or a base64-encoded image. "
            f"If you are using Marqo Cloud, please note that images can only be downloaded "
            f"from a URL and local files are not supported. "
            f"If you are running Marqo in a Docker container, you will need to use a Docker "
            f"volume so that your container can access host files. "
            f"For more information, please refer to: "
            f"{marqo_docs.indexing_images()}"
        )

    return img


def download_image_from_url(
    image_path: str,
    media_download_headers: dict,
    timeout_ms: int = 3000,
    modality: Optional[str] = None,
) -> BytesIO:
    """Download an image from a URL and return a PIL image using pycurl.

    For video/audio files, we check the file size during download rather than making a separate HEAD request upfront.
    While checking Content-Length beforehand is possible, it would add latency to every request. Since most files
    are expected to be under the size limit, we optimize for the common case by checking size during download.

    Args:
        image_path (str): URL to the image.
        media_download_headers (dict): Headers for the image download.
        timeout_ms (int): Timeout in milliseconds, for the whole request.
        modality (Optional[str]): Type of media being downloaded ('video', 'audio', or None)

    Returns:
        buffer (BytesIO): The image as a BytesIO object.

    Raises:
        ImageDownloadError: If the image download fails or exceeds size limit for video/audio.
    """

    if not isinstance(timeout_ms, int):
        raise InternalServerError(
            f"timeout must be an integer but received {timeout_ms} of type {type(timeout_ms)}"
        )

    try:
        encoded_url = encode_url(image_path)
    except UnicodeEncodeError as e:
        raise ImageDownloadError(
            f"Marqo encountered an error when downloading the media url {image_path}. "
            f"The url could not be encoded properly. Original error: {e}"
        )
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.CAINFO, certifi.where())
    c.setopt(pycurl.URL, encoded_url)
    c.setopt(pycurl.WRITEDATA, buffer)
    c.setopt(pycurl.TIMEOUT_MS, timeout_ms)
    c.setopt(pycurl.FOLLOWLOCATION, 1)

    headers = DEFAULT_HEADERS.copy()
    if media_download_headers is None:
        media_download_headers = dict()
    headers.update(media_download_headers)
    c.setopt(pycurl.HTTPHEADER, [f"{k}: {v}" for k, v in headers.items()])

    try:
        c.perform()
        if c.getinfo(pycurl.RESPONSE_CODE) != 200:
            raise ImageDownloadError(
                f"media url `{image_path}` returned {c.getinfo(pycurl.RESPONSE_CODE)}"
            )
    except pycurl.error as e:
        error_message = str(e)
        if len(e.args) > 0:
            error_code = e.args[0]
            if error_code == pycurl.E_ABORTED_BY_CALLBACK:
                error_message = f"Media file `{image_path}` exceeds the maximum allowed size for {modality}."
        raise ImageDownloadError(
            f"Marqo encountered an error when downloading the media url {image_path}. "
            f"The original error is: {error_message}"
        )

    finally:
        c.close()

    buffer.seek(0)
    return buffer


def encode_url(url: str) -> str:
    """
    Encode a URL to a valid format with only ASCII characters and reserved characters using percent-encoding.

    In version 2.8, we replaced the requests library with pycurl for image downloads. Consequently, we need to implement
    the URL encoding function ourselves. This function replicates the encoding behavior of the
    'requests.utils.requote_uri' function from the requests library.

    Args:
        url (str): The URL to encode.

    Returns:
        str: The encoded URL.

    Raises:
        UnicodeEncodeError: If the URL cannot be encoded properly.

    """
    return requests.utils.requote_uri(url)


def download_media_from_url(
    media_path: str,
    media_download_headers: dict,
    timeout_ms: int = 3000,
    modality: Optional[str] = None,
):
    return download_image_from_url(
        media_path, media_download_headers, timeout_ms, modality
    )
