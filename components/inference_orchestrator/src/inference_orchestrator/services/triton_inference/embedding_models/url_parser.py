import os
from urllib.parse import urlparse


def get_base_filename(path_or_url: str) -> str:
    """Extract the base filename from a local file path or a URL (ignoring query params)."""
    parsed = urlparse(path_or_url)

    if parsed.scheme in ("http", "https", "ftp", "s3"):
        return os.path.basename(parsed.path)
    else:
        return os.path.basename(path_or_url)
