import io
import os
from contextlib import contextmanager
from typing import Optional, Union, List
from urllib.parse import urlparse

import magic
import requests
import validators

from marqo.core.inference.api import Modality, MediaDownloadError


@contextmanager
def fetch_content_sample(url: str, media_download_headers: Optional[dict] = None, sample_size=10240):  # 10 KB
    # It's ok to pass None to requests.get() for headers and it won't change the default headers
    """Fetch a sample of the content from the URL.

    Raises:
        HTTPError: If the response status code is not 200
    """
    response = requests.get(url, stream=True, headers=media_download_headers)
    response.raise_for_status()
    buffer = io.BytesIO()
    try:
        # FIXME what is the point of having both sample_size and max chunk size hard coded?
        for chunk in response.iter_content(chunk_size=min(sample_size, 8192)):
            buffer.write(chunk)
            if buffer.tell() >= sample_size:
                break
        buffer.seek(0)
        yield buffer
    finally:
        buffer.close()
        response.close()


def _infer_modality_based_on_extension(extension: str) -> Optional[Modality]:
    """
    Infer the modality based on the file extension. Is it is not a known extension, return None.

    Args:
        extension: A string representing the file extension (e.g., 'jpg', 'mp4', etc.)

    Returns:
        Modality: The inferred modality (IMAGE, VIDEO, AUDIO, or None if unknown)
    """
    if not extension or not isinstance(extension, str):
        return None

    extension = extension.lower()

    if extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
        return Modality.IMAGE
    elif extension in ['mp4', 'avi', 'mov']:
        return Modality.VIDEO
    elif extension in ['mp3', 'wav', 'ogg']:
        return Modality.AUDIO
    else:
        return None


def _infer_modality_based_on_mime_type(mime_object: str) -> Modality:
    """
    Infer the modality based on the MIME type. If it is not a known MIME type, return TEXT.

    Args:
        mime_object: the MIME type of the content (e.g., 'image/jpeg', 'video/mp4', etc.)

    Returns:
        Modality: The inferred modality (IMAGE, VIDEO, AUDIO, or TEXT if unknown)
    """
    if mime_object.startswith('image/'):
        return Modality.IMAGE
    elif mime_object.startswith('video/'):
        return Modality.VIDEO
    elif mime_object.startswith('audio/'):
        return Modality.AUDIO
    else:
        return Modality.TEXT


# TODO this method is copied from s2_inference.multimodal_modal_load class, improve it
def infer_modality(content: Union[str, List[str], bytes], media_download_headers: Optional[dict] = None) -> Modality:
    """
    Infer the modality of the content. Video, audio, image or text.

    If the content is a URL, we will firstly infer the modality based on the file extension, and
    return the modality if it is known. This will be a short-circuit operatio, and we accept the edge cases
    that the content has an incorrect extension

    If the content is a bytes object, we will infer the modality based on the MIME type and return the modality.

    If the content is neither a URL nor a bytes object, we will return TEXT.
    """
    if isinstance(content, str):
        if not validate_url(content):
            return Modality.TEXT

        # Encode the URL
        encoded_url = encode_url(content)
        extension = get_url_file_extension(encoded_url)

        # Check if the URL has a file extension
        if extension:
            modality: Optional[Modality] = _infer_modality_based_on_extension(extension)
            if modality:
                return modality

        # Use context manager to handle content sample
        try:
            with fetch_content_sample(encoded_url, media_download_headers) as sample:
                mime = magic.from_buffer(sample.read(), mime=True)
                modality: Modality = _infer_modality_based_on_mime_type(mime)
                return modality
        except requests.exceptions.RequestException as e:
            raise MediaDownloadError(f"Error downloading media file {content}: {e}") from e
        except magic.MagicException as e:
            raise MediaDownloadError(f"Error determining MIME type for {encoded_url}: {e}") from e
        except IOError as e:
            raise MediaDownloadError(f"IO error while processing {encoded_url}: {e}") from e

    elif isinstance(content, bytes):
        # Use python-magic for byte content
        mime = magic.from_buffer(content, mime=True)
        return _infer_modality_based_on_mime_type(mime)
    else:
        return Modality.TEXT


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


def get_url_file_extension(url: str) -> Optional[str]:
    """Get the file extension from a URL.

    This function removes the query parameters and fragments from the URL and then extracts the file extension.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path  # This excludes query parameters and fragments

    # Get the basename (e.g., 'image.jpg')
    filename = os.path.basename(path)

    # Split the extension
    _, ext = os.path.splitext(filename)

    if ext:
        return ext.lstrip('.').lower()
    return None
