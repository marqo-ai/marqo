import json
import urllib.parse
from marqo.errors import InvalidArgError, InternalError
from marqo.tensor_search import enums
from typing import Optional
from marqo.tensor_search.utils import construct_authorized_url
from marqo.tensor_search.models.add_docs_objects import ModelAuth
from marqo.tensor_search.models.add_docs_objects import AddDocsParams, AddDocsBodyParams
from marqo.errors import BadRequestError
from typing import Union, List, Optional, Dict
from fastapi import Request


def upconstruct_authorized_url(opensearch_url: str) -> str:
    """Generates an authorized URL, if it is not already authorized
    """
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
    return authorized_url


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
                                device: str, auto_refresh: bool = True, non_tensor_fields: Optional[List[str]] = None,
                                mappings: Optional[dict] = dict(), model_auth: Optional[ModelAuth] = None,
                                image_download_headers: Optional[dict] = dict(),
                                use_existing_tensors: Optional[bool] = False, query_parameters: Optional[Dict] = dict()) -> AddDocsParams:
    """An orchestrator for the add_documents API to support both old and new versions of the API.
    All the arguments are decoded and validated in the API function. This function is only responsible for orchestrating.

    Returns:
        AddDocsParams: An AddDocsParams object for internal use
    """

    if isinstance(body, AddDocsBodyParams):
        docs = body.documents

        # Check for query parameters that are not supported in the new API
        deprecated_fields = ["non_tensor_fields", "use_existing_tensors", "image_download_headers", "model_auth", "mappings"]
        if any(field in query_parameters for field in deprecated_fields):
            raise BadRequestError("Marqo is not accepting any of the following parameters in the query string: "
                                  "`non_tensor_fields`, `use_existing_tensors`, `image_download_headers`, `model_auth`, `mappings`. "
                                  "Please move these parameters to the request body as "
                                  "`nonTensorFields`,` useExistingTensors`, `imageDownloadHeaders`, `modelAuth`, `mappings`. and try again. "
                                  "Please check `https://docs.marqo.ai/latest/API-Reference/documents/` for the correct APIs.")

        mappings = body.mappings
        non_tensor_fields = body.nonTensorFields
        tensor_fields = body.tensorFields
        use_existing_tensors = body.useExistingTensors
        model_auth = body.modelAuth
        image_download_headers = body.imageDownloadHeaders

        if tensor_fields is not None and non_tensor_fields is not None:
            raise BadRequestError('Cannot provide `nonTensorFields` when `tensorFields` is defined. '
                                  '`nonTensorField`s has been deprecated and will be removed in Marqo 2.0.0. '
                                  'Its use is discouraged.')

        if tensor_fields is None and non_tensor_fields is None:
            raise BadRequestError('Required parameter `tensorFields` is missing from the request body. '
                                  'Use `tensorFields=[]` to index for lexical-only search.')

        return AddDocsParams(
            index_name=index_name, docs=docs, auto_refresh=auto_refresh,
            device=device, non_tensor_fields=non_tensor_fields, tensor_fields=tensor_fields,
            use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers,
            mappings=mappings, model_auth=model_auth
        )

    elif isinstance(body, list) and all(isinstance(item, dict) for item in body):
        docs = body

        if non_tensor_fields is None:
            raise BadRequestError('Required parameter `tensorFields` is missing from the request body. '
                                  'This endpoint now requires `tensorFields` in request body. Providing '
                                  'a list of documents as body has been deprecated and will not be '
                                  'supported in Marqo 2.0.0')

        return AddDocsParams(
            index_name=index_name, docs=docs, auto_refresh=auto_refresh,
            device=device, non_tensor_fields=non_tensor_fields,
            use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers,
            mappings=mappings, model_auth=model_auth
        )

    else:
        raise InternalError(f"Unexpected request body type `{type(body).__name__} for `/documents` API. ")
