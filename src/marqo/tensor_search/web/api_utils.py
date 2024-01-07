import json
import urllib.parse
from typing import Union, List, Optional, Dict

from marqo.api.exceptions import InvalidArgError
from marqo.tensor_search import enums
from marqo.tensor_search.models.add_docs_objects import AddDocsParams, AddDocsBodyParams
from marqo.tensor_search.models.add_docs_objects import ModelAuth


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


def decode_image_download_headers(image_download_headers: Optional[str] = None) -> dict:
    """Decodes an image download header string into a Python dict

    Args:
        image_download_headers: JSON-serialised, URL encoded header dictionary

    Returns:
        image_download_headers as a dict

    Raises:
        InvalidArgError if there is trouble parsing the dictionary
    """
    if not image_download_headers:
        return dict()
    else:
        try:
            as_str = urllib.parse.unquote_plus(image_download_headers)
            as_dict = json.loads(as_str)
            return as_dict
        except json.JSONDecodeError as e:
            raise InvalidArgError(f"Error parsing image_download_headers. Message: {e}")


def decode_query_string_model_auth(model_auth: Optional[str] = None) -> Optional[ModelAuth]:
    """Decodes a url encoded ModelAuth string into a ModelAuth object

    Args:
        model_auth: JSON-serialised, URL encoded ModelAuth dictionary

    Returns:
        model_auth as a ModelAuth object, if found. Otherwise None

    Raises:
        ValidationError if there is trouble parsing the string
    """
    if not model_auth:
        return None
    else:
        as_str = urllib.parse.unquote_plus(model_auth)
        as_objc = ModelAuth.parse_raw(as_str)
        return as_objc


def decode_mappings(mappings: Optional[str] = None) -> dict:
    """Decodes mappings string into a Python dict

       Args:
           mappings: JSON-serialised, URL encoded mappings object

       Returns:
           mappings as a dict

       Raises:
           InvalidArgError is there is trouble parsing the dictionary
    """
    if not mappings:
        return dict()
    else:
        try:
            as_str = urllib.parse.unquote_plus(mappings)
            as_dict = json.loads(as_str)
            return as_dict
        except json.JSONDecodeError as e:
            raise InvalidArgError(f"Error parsing mappings. Message: {e}")


def add_docs_params_orchestrator(index_name: str, body: Union[AddDocsBodyParams, List[Dict]],
                                 device: str, auto_refresh: bool = True) -> AddDocsParams:
    """An orchestrator for the add_documents API.
    All the arguments are decoded and validated in the API function. This function is only responsible for orchestrating.

    Returns:
        AddDocsParams: An AddDocsParams object for internal use
    """
    docs = body.documents

    mappings = body.mappings
    tensor_fields = body.tensorFields
    use_existing_tensors = body.useExistingTensors
    model_auth = body.modelAuth
    image_download_headers = body.imageDownloadHeaders
    image_download_thread_count = body.imageDownloadThreadCount

    return AddDocsParams(
        index_name=index_name, docs=docs, auto_refresh=auto_refresh,
        device=device, tensor_fields=tensor_fields,
        use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers,
        image_download_thread_count=image_download_thread_count,
        mappings=mappings, model_auth=model_auth
    )
