import io
from contextlib import contextmanager
from typing import Optional, Union, List

import magic
import requests
from requests.utils import requote_uri
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


# TODO this method is copied from s2_inference.multimodal_modal_load class, improve it
def infer_modality(content: Union[str, List[str], bytes], media_download_headers: Optional[dict] = None) -> Modality:
    """
    Infer the modality of the content. Video, audio, image or text.
    """
    if isinstance(content, str):
        if not validate_url(content):
            return Modality.TEXT

        # Encode the URL
        encoded_url = encode_url(content)
        extension = encoded_url.split('.')[-1].lower()
        if extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            return Modality.IMAGE
        elif extension in ['mp4', 'avi', 'mov']:
            return Modality.VIDEO
        elif extension in ['mp3', 'wav', 'ogg']:
            return Modality.AUDIO
        if validate_url(encoded_url):
            # Use context manager to handle content sample
            try:
                with fetch_content_sample(encoded_url, media_download_headers) as sample:
                    mime = magic.from_buffer(sample.read(), mime=True)
                    if mime.startswith('image/'):
                        return Modality.IMAGE
                    elif mime.startswith('video/'):
                        return Modality.VIDEO
                    elif mime.startswith('audio/'):
                        return Modality.AUDIO
            except requests.exceptions.RequestException as e:
                raise MediaDownloadError(f"Error downloading media file {content}: {e}") from e
            except magic.MagicException as e:
                raise MediaDownloadError(f"Error determining MIME type for {encoded_url}: {e}") from e
            except IOError as e:
                raise MediaDownloadError(f"IO error while processing {encoded_url}: {e}") from e

        return Modality.TEXT

    elif isinstance(content, bytes):
        # Use python-magic for byte content
        mime = magic.from_buffer(content, mime=True)
        if mime.startswith('image/'):
            return Modality.IMAGE
        elif mime.startswith('video/'):
            return Modality.VIDEO
        elif mime.startswith('audio/'):
            return Modality.AUDIO
        else:
            return Modality.TEXT

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
