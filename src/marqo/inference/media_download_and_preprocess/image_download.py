import os
from io import BytesIO

import certifi
import numpy as np
import pycurl
import requests
import torch
import validators
from PIL import Image, UnidentifiedImageError
from requests.utils import requote_uri

from marqo import marqo_docs
from marqo.api.exceptions import InternalError
from marqo.s2_inference.errors import ImageDownloadError
from marqo.s2_inference.types import *
from marqo.s2_inference.types import Modality
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.telemetry import RequestMetrics
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints

# TODO Merge this with the one in clip_utils in the future refactoring

DEFAULT_HEADERS = {'User-Agent': 'Marqobot/1.0'}


def get_allowed_image_types():
    return {'.jpg', '.png', '.bmp', '.jpeg'}


def _is_image(inputs: Union[str, List[Union[str, ImageType, ndarray]]]) -> bool:
    # some logic to determine if something is an image or not
    # assume the batch is the same type
    # maybe we use something like this https://github.com/ahupp/python-magic

    _allowed = get_allowed_image_types()

    # we assume the batch is this way if a list
    # otherwise apply over each element
    if isinstance(inputs, list):

        if len(inputs) == 0:
            raise UnidentifiedImageError("received empty list, expected at least one element.")

        thing = inputs[0]
    else:
        thing = inputs

    # if it is a string, determine if it is a local file or url
    if isinstance(thing, str):
        name, extension = os.path.splitext(thing.lower())

        # if it has the correct extension, asssume yes
        if extension in _allowed:
            return True

        # if it is a local file without extension, then raise an error
        if os.path.isfile(thing):
            # we could also read the first part of the file and infer
            raise UnidentifiedImageError(
                f"local file [{thing}] extension {extension} does not match allowed file types of {_allowed}")
        else:
            # if it is not a local file and does not have an extension
            # check if url
            if validators.url(thing):
                return True
            else:
                return False

    # if it is an array, then it is an image
    elif isinstance(thing, (ImageType, ndarray, Tensor)):
        return True
    else:
        raise UnidentifiedImageError(f"expected type Image or str for inputs but received type {type(thing)}")


def format_and_load_CLIP_images(images: List[Union[str, ndarray, ImageType, Tensor]],
                                media_download_headers: dict) -> Union[List[ImageType], List[Tensor]]:
    """takes in a list of strings, arrays or urls and either loads and/or converts to PIL
        for the clip model

    Args:
        images (List[Union[str, np.ndarray, ImageType]]): list of file locations or arrays (can be mixed)

    Raises:
        TypeError: _description_

    Returns:
        List[ImageType]: list of PIL images
    """
    if not isinstance(images, list):
        raise TypeError(f"expected list but received {type(images)}")

    results = []
    for image in images:
        results.append(format_and_load_CLIP_image(image, media_download_headers))

    return results


def validate_url(url: str) -> bool:
    """Validate a URL to ensure it is a valid URL. Returns True if the URL is valid or the encoded URL is valid.
    Args:
        url (str): URL to validate.
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if isinstance(url, str):
        return validators.url(url) or validators.url(encode_url(url))
    else:
        return False


def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType, Tensor],
                               media_download_headers: dict) -> Union[ImageType, Tensor]:
    """standardizes the input to be a PIL image

    Args:
        image (Union[str, np.ndarray, ImageType, Tensor]): can be a local file, url, array or a tensor

    Raises:
        ValueError: _description_
        TypeError: _description_

    Returns:
        standardized the image:
            ImageType: PIL image if input is a string, an array or a PIL image
            Tensor: torch tensor if input is a torch tensor
    """
    # check for the input type
    if isinstance(image, str):
        img = load_image_from_path(image, media_download_headers)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, torch.Tensor):
        img = image
    elif isinstance(image, ImageType):
        img = image
    else:
        raise UnidentifiedImageError(f"input of type {type(image)} "
                                     f"did not match allowed types of str, np.ndarray, ImageType, Tensor")

    return img


def load_image_from_path(image_path: str, media_download_headers: dict, timeout_ms=3000,
                         metrics_obj: Optional[RequestMetrics] = None) -> ImageType:
    """Loads an image into PIL from a string path that is either local or a url

    Args:
        image_path (str): Local or remote path to image.
        media_download_headers (dict): header for the image download
        timeout_ms (int): timeout (in milliseconds), for the whole request
    Raises:
        ValueError: If the local path is invalid, and is not a url
        UnidentifiedImageError: If the image is irretrievable or unprocessable.

    Returns:
        ImageType: In-memory PIL image.
    """
    if os.path.isfile(image_path):
        img = Image.open(image_path)
    elif validators.url(image_path):
        if metrics_obj is not None:
            metrics_obj.start(f"image_download.{image_path}")
        try:
            img_io: BytesIO = download_image_from_url(image_path, media_download_headers, timeout_ms)
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
                metrics_obj.stop(f"image_download.{image_path}")
    else:
        raise UnidentifiedImageError(f"Input str of {image_path} is not a local file or a valid url. "
                                     f"If you are using Marqo Cloud, please note that images can only be downloaded "
                                     f"from a URL and local files are not supported. "
                                     f"If you are running Marqo in a Docker container, you will need to use a Docker "
                                     f"volume so that your container can access host files. "
                                     f"For more information, please refer to: "
                                     f"{marqo_docs.indexing_images()}")

    return img


def download_image_from_url(image_path: str, media_download_headers: dict, timeout_ms: int = 3000,
                            modality: Optional[str] = None) -> BytesIO:
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
        raise InternalError(f"timeout must be an integer but received {timeout_ms} of type {type(timeout_ms)}")

    try:
        encoded_url = encode_url(image_path)
    except UnicodeEncodeError as e:
        raise ImageDownloadError(f"Marqo encountered an error when downloading the media url {image_path}. "
                                 f"The url could not be encoded properly. Original error: {e}")
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

    # callback to check file size for video and audio
    if modality in [Modality.VIDEO, Modality.AUDIO]:
        max_size = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE)

        def progress(download_total, downloaded, upload_total, uploaded):
            if downloaded > max_size:
                return 1

        c.setopt(pycurl.NOPROGRESS, False)
        c.setopt(pycurl.XFERINFOFUNCTION, progress)

    try:
        c.perform()
        if c.getinfo(pycurl.RESPONSE_CODE) != 200:
            raise ImageDownloadError(f"media url `{image_path}` returned {c.getinfo(pycurl.RESPONSE_CODE)}")
    except pycurl.error as e:
        error_message = str(e)
        if len(e.args) > 0:
            error_code = e.args[0]
            if error_code == pycurl.E_ABORTED_BY_CALLBACK:
                error_message = f"Media file `{image_path}` exceeds the maximum allowed size for {modality}."
        raise ImageDownloadError(f"Marqo encountered an error when downloading the media url {image_path}. "
                                 f"The original error is: {error_message}")

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


def download_media_from_url(media_path: str, media_download_headers: dict, timeout_ms: int = 3000,
                            modality: Optional[str] = None):
    return download_image_from_url(media_path, media_download_headers, timeout_ms, modality)
