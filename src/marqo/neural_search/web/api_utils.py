from marqo.errors import InvalidArgError
from marqo.neural_search import enums

def translate_api_device(device: str) -> str:
    """Translates an API device as given through the API into an internal enum.

    Args:
        device: A device as given as url arg. For example: "cuda2" and "cpu".
            Assumes it has already been validated/

    Returns:
        device in its internal form (cuda2 -> cuda:2)

    Raises:
        InvalidArgError if device is invalid
    """
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

