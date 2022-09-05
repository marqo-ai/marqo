from marqo.errors import InvalidArgError, InternalError
from marqo.tensor_search import enums
from typing import Optional
from marqo.tensor_search.utils import construct_authorized_url
from marqo import config


def generate_config(opensearch_url) -> config.Config:
    http_sep = "://"
    if http_sep not in opensearch_url:
        raise InternalError(f"Could not parse backend url: {opensearch_url}")
    if "@" not in opensearch_url.split("/")[2]:
        authorized_url = construct_authorized_url(
            url_base=opensearch_url,
            username="admin",
            password="admin"
        )
    else:
        authorized_url = opensearch_url
    return config.Config(url=authorized_url)


def translate_api_device(device: Optional[str]) -> Optional[str]:
    """Translates an API device as given through the API into an internal enum.

    Args:
        device: A device as given as url arg. For example: "cuda2" and "cpu".
            Assumes it has already been validated/

    Returns:
        device in its internal form (cuda2 -> cuda:2)

    Raises:
        InvalidArgError if device is invalid
    """
    if device is None:
        return device

    lowered_device = device.lower()
    acceptable_devices = [d.value.lower() for d in enums.Device]

    match_attempt = [
        (
            lowered_device.startswith(acceptable),
            lowered_device.replace(acceptable, ""),
            acceptable
         )
        for acceptable in acceptable_devices]

    try:
        matched = [attempt for attempt in match_attempt if attempt[0]][0]
        prefix = matched[2]
        suffix = matched[1]
        if not suffix:
            return prefix
        else:
            formatted = f"{prefix}:{suffix}"
            return formatted
    except (IndexError, ValueError) as k:
        raise InvalidArgError(f"Given device `{device}` isn't  a known device type. "
                              f"Acceptable device types: {acceptable_devices}")

