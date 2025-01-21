"""Abstractions for Multimodal Models"""

import io
from contextlib import contextmanager
from typing import Optional, Union, List

import magic
import requests

from marqo.core.inference.image_download import encode_url
from marqo.s2_inference.clip_utils import validate_url
from marqo.s2_inference.errors import MediaDownloadError
from marqo.s2_inference.types import Modality


@contextmanager
def fetch_content_sample(url, media_download_headers: Optional[dict] = None, sample_size=10240):  # 10 KB
    # It's ok to pass None to requests.get() for headers and it won't change the default headers
    """Fetch a sample of the content from the URL.

    Raises:
        HTTPError: If the response status code is not 200
    """
    response = requests.get(url, stream=True, headers=media_download_headers)
    response.raise_for_status()
    buffer = io.BytesIO()
    try:
        for chunk in response.iter_content(chunk_size=min(sample_size, 8192)):
            buffer.write(chunk)
            if buffer.tell() >= sample_size:
                break
        buffer.seek(0)
        yield buffer
    finally:
        buffer.close()
        response.close()


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