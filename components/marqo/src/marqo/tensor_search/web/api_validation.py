import typing

from marqo.api.exceptions import InvalidArgError
from marqo.tensor_search import enums
from marqo.tensor_search.web import api_utils


def validate_api_device_string(device: typing.Optional[str]) -> typing.Optional[str]:
    # TODO [Refactoring device logic] move this logic to device manager
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
    """Translates the device string for internal use.
    
    This function only performs basic string translation and does not validate
    if the device is available as inference runs in a separate service.

    Args:
        device: Device string to translate (can be None)

    Returns:
        The device translated for internal use or None if no device was provided
    """
    if device is None:
        return None
        
    return api_utils.translate_api_device(validate_api_device_string(device))
