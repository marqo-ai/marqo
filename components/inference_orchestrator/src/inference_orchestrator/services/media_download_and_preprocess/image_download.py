import base64
import logging
import os
import socket
import time
from io import BytesIO
from typing import Optional
from urllib.parse import quote, urlparse

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

logger = logging.getLogger(__name__)

settings = get_settings()

DEFAULT_HEADERS = {"User-Agent": "Marqobot/1.0"}

_PROXY_FALLBACK_CURL_CODES = frozenset({
    pycurl.E_COULDNT_CONNECT,
    pycurl.E_COULDNT_RESOLVE_HOST,
    pycurl.E_OPERATION_TIMEDOUT,
})

_PROXY_FALLBACK_HTTP_CODES = frozenset({502, 503, 504})

_AAAA_CACHE: dict[str, tuple[bool, float]] = {}
_AAAA_CACHE_TTL = 3600  # 1 hour


def _origin_has_ipv6(hostname: str) -> bool:
    """Check if hostname has AAAA records (supports IPv6).
    Results are cached in-memory with a 1-hour TTL."""
    now = time.monotonic()
    cached = _AAAA_CACHE.get(hostname)
    if cached and (now - cached[1]) < _AAAA_CACHE_TTL:
        return cached[0]
    try:
        results = socket.getaddrinfo(
            hostname, 443, socket.AF_INET6, socket.SOCK_STREAM,
        )
        has_ipv6 = len(results) > 0
    except socket.gaierror:
        has_ipv6 = False
    _AAAA_CACHE[hostname] = (has_ipv6, now)
    return has_ipv6


def _maybe_proxy_url(image_path: str) -> str:
    """Construct the proxy URL for a given image path.
    Returns the original path unchanged if the proxy is not configured."""
    proxy_url = settings.marqo_media_proxy_url
    if not proxy_url:
        return image_path
    return f"{proxy_url}?url={quote(image_path, safe='')}"


def _get_proxy_headers() -> dict:
    """Return Cloudflare Access service token headers if configured."""
    if settings.cf_access_client_id and settings.cf_access_client_secret:
        return {
            "CF-Access-Client-Id": settings.cf_access_client_id,
            "CF-Access-Client-Secret": settings.cf_access_client_secret,
        }
    return {}


def _is_proxy_failure(exc: ImageDownloadError) -> bool:
    """Return True if the failure is due to the proxy being down,
    not the origin. We only fallback for connectivity/timeout
    errors and 502/503/504 from the proxy itself."""
    # Check for pycurl connection-level errors via the cause chain
    cause = exc.__cause__
    if isinstance(cause, pycurl.error) and len(cause.args) > 0:
        if cause.args[0] in _PROXY_FALLBACK_CURL_CODES:
            return True
    # Check for HTTP gateway errors in the message
    msg = str(exc)
    for code in _PROXY_FALLBACK_HTTP_CODES:
        if f"returned {code}" in msg:
            return True
    return False


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


def _do_download(
    image_path: str,
    media_download_headers: dict,
    timeout_ms: int,
    modality: Optional[str] = None,
    proxy_headers: Optional[dict] = None,
    force_ipv6: bool = False,
    fetch_url: Optional[str] = None,
) -> BytesIO:
    """Low-level pycurl download.

    Args:
        image_path: Original media URL (used in error messages, never mutated).
        media_download_headers: Caller-provided headers for the download.
        timeout_ms: Timeout in milliseconds for the whole request.
        modality: Type of media being downloaded ('video', 'audio', or None).
        proxy_headers: Optional Cloudflare Access headers to include.
        force_ipv6: If True, force pycurl to resolve and connect over IPv6.
        fetch_url: If provided, the actual URL to fetch (e.g. proxy URL).
            Defaults to image_path when not set.

    Returns:
        buffer: The downloaded content as a BytesIO object.

    Raises:
        ImageDownloadError: If the download fails or exceeds size limit.
    """
    actual_url = fetch_url if fetch_url is not None else image_path

    try:
        encoded_url = encode_url(actual_url)
    except UnicodeEncodeError as e:
        raise ImageDownloadError(
            f"Marqo encountered an error when downloading the media url {image_path}. "
            f"The url could not be encoded properly. Original error: {e}"
        )

    buffer = BytesIO()
    c = pycurl.Curl()
    try:
        c.setopt(pycurl.CAINFO, certifi.where())
        c.setopt(pycurl.URL, encoded_url)
        c.setopt(pycurl.WRITEDATA, buffer)
        c.setopt(pycurl.TIMEOUT_MS, timeout_ms)
        c.setopt(pycurl.FOLLOWLOCATION, 1)

        if force_ipv6:
            c.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V6)

        headers = DEFAULT_HEADERS.copy()
        if media_download_headers is None:
            media_download_headers = dict()
        headers.update(media_download_headers)
        if proxy_headers:
            headers.update(proxy_headers)
        c.setopt(pycurl.HTTPHEADER, [f"{k}: {v}" for k, v in headers.items()])

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
        ) from e
    finally:
        c.close()

    buffer.seek(0)
    return buffer


def download_image_from_url(
    image_path: str,
    media_download_headers: dict,
    timeout_ms: int = 3000,
    modality: Optional[str] = None,
) -> BytesIO:
    """Download an image from a URL using a three-tier strategy:

    1. Direct IPv6 if origin has AAAA records (free via EIGW)
    2. Route through Cloudflare Worker proxy over IPv6 if origin is IPv4-only
    3. Fall back to direct IPv4 through NAT Gateway if proxy is unreachable

    When MARQO_MEDIA_PROXY_URL is not set, downloads directly over IPv4,
    preserving existing behaviour exactly.

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

    if not settings.marqo_media_proxy_url:
        # Feature disabled -- behave identically to before.
        return _do_download(image_path, media_download_headers, timeout_ms, modality)

    origin_host = urlparse(image_path).hostname

    # --- Tier 1: Direct IPv6 (if origin supports it) ---
    if origin_host and _origin_has_ipv6(origin_host):
        try:
            return _do_download(
                image_path, media_download_headers, timeout_ms, modality,
                force_ipv6=True,
            )
        except ImageDownloadError:
            logger.warning(
                "Direct IPv6 download failed for %s, falling through to proxy",
                origin_host,
            )

    # --- Tier 2: Proxy over IPv6 ---
    proxy_url = _maybe_proxy_url(image_path)
    try:
        return _do_download(
            image_path, media_download_headers, timeout_ms, modality,
            proxy_headers=_get_proxy_headers(),
            fetch_url=proxy_url,
        )
    except ImageDownloadError as exc:
        if _is_proxy_failure(exc):
            logger.warning(
                "Proxy download failed for %s, falling back to direct IPv4: %s",
                image_path, exc,
            )
            # --- Tier 3: Direct IPv4 (last resort) ---
            return _do_download(image_path, media_download_headers, timeout_ms, modality)
        raise


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
