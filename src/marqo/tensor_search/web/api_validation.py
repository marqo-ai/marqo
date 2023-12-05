from marqo.api.exceptions import InvalidArgError, HardwareCompatabilityError
from marqo.tensor_search import enums, utils
from marqo.tensor_search.web import api_utils
import typing


def validate_api_device_string(device: typing.Optional[str]) -> typing.Optional[str]:
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


async def validate_device(device: typing.Optional[str] = None) -> typing.Optional[str]:
    """Translates and validates the device string. Checks if the requested
    device is available.

    "cuda1" -> "cuda:1"

    Args:
        device:

    Returns:
        The device translated for internal use. If it has passed validation.
    """
    if device is None:
        return None
    translated = api_utils.translate_api_device(validate_api_device_string(device))
    if utils.check_device_is_available(translated):
        return translated
    else:
        raise HardwareCompatabilityError(message="Requested device is not available to this Marqo instance."
                                                 f" Requested device: `{translated}`")
