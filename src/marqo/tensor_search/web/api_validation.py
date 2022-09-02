from marqo.tensor_search.validation import validate_str_against_enum
from marqo.errors import InvalidArgError
from marqo.tensor_search import enums
import typing


def validate_api_device(device: typing.Optional[str]) -> typing.Optional[str]:
    """Validates a device which is an API parameter

    Args:
        device: the string to be checked. Examples of acceptable device args
            include "cuda2" and "cpu"

    Returns:
        device, if it is acceptable

    Raises:
        InvalidArgError if device is invalid
    """
    if device is None:
        return device

    if not isinstance(device, str):
        raise InvalidArgError(f"Device must be a str! Given "
                              f"device `{device}` of type {type(device).__name__} ")
    lowered_device = device.lower()
    acceptable_devices = [d.value.lower() for d in enums.Device]

    match_attempt = [
        (lowered_device.startswith(acceptable),
         lowered_device.replace(acceptable, ""))
        for acceptable in acceptable_devices]

    try:
        prefix_match = [attempt[1] for attempt in match_attempt if attempt[0]][0]
    except IndexError as k:
        raise InvalidArgError(f"Given device `{device}` doesn't start with a known device type. "
                              f"Acceptable device types: {acceptable_devices}")
    if not prefix_match:
        return device
    try:
        int(prefix_match)

    except ValueError:
        raise InvalidArgError(f"Given device `{device}` not recognised. "
                        f"Acceptable devices: {acceptable_devices}")
    return device
