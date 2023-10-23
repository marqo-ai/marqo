import copy
import json
import time
import pprint
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Union
import requests
from json.decoder import JSONDecodeError
from marqo.config import Config
from marqo.errors import (
    MarqoWebError,
    BackendCommunicationError,
    BackendTimeoutError,
    IndexNotFoundError,
    DocumentNotFoundError,
    IndexAlreadyExistsError,
    InvalidIndexNameError,
    HardwareCompatabilityError,
    IndexMaxFieldsError, TooManyRequestsError,
    DiskWatermarkBreachError
)
from urllib3.exceptions import InsecureRequestWarning
import warnings
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.tensor_search.constants import DEFAULT_MARQO_MAX_BACKEND_RETRY_ATTEMPTS, DEFAULT_MARQO_MAX_BACKEND_RETRY_BACKOFF

logger = get_logger(__name__)

ALLOWED_OPERATIONS = {requests.delete, requests.get, requests.post, requests.put}

OPERATION_MAPPING = {'delete': requests.delete, 'get': requests.get,
                     'post': requests.post, 'put': requests.put}


class HttpRequests:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.headers = dict()

    def send_request(
        self,
        http_method: Callable,
        path: str,
        body: Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[str], str]] = None,
        content_type: Optional[str] = None,
        max_retry_attempts: Optional[int] = None,
        max_retry_backoff_seconds: Optional[int] = None
    ) -> Any:
        if max_retry_attempts is None:
            max_retry_attempts = DEFAULT_MARQO_MAX_BACKEND_RETRY_ATTEMPTS
        if max_retry_backoff_seconds is None:
            max_retry_backoff_seconds = DEFAULT_MARQO_MAX_BACKEND_RETRY_BACKOFF

        to_verify = False #  self.config.cluster_is_remote

        if http_method not in ALLOWED_OPERATIONS:
            raise ValueError("{} not an allowed operation {}".format(http_method, ALLOWED_OPERATIONS))

        req_headers = copy.deepcopy(self.headers)

        if content_type is not None and content_type:
            req_headers['Content-Type'] = content_type

        with warnings.catch_warnings():
            if not self.config.cluster_is_remote:
                warnings.simplefilter('ignore', InsecureRequestWarning)
            
            for attempt in range(max_retry_attempts + 1):
                try:
                    request_path = self.config.url + '/' + path
                    if isinstance(body, (bytes, str)):
                        response = http_method(
                            request_path,
                            timeout=self.config.timeout,
                            headers=req_headers,
                            data=body,
                            verify=to_verify
                        )
                    else:
                        response = http_method(
                            request_path,
                            timeout=self.config.timeout,
                            headers=req_headers,
                            data=json.dumps(body) if body else None,
                            verify=to_verify
                        )
                    return self.__validate(response)
                except requests.exceptions.Timeout as err:
                    raise BackendTimeoutError(str(err)) from err
                except requests.exceptions.ConnectionError as err:
                    if (attempt == max_retry_attempts):
                        raise BackendCommunicationError(str(err)) from err
                    else:
                        logger.info(f"BackendCommunicationError encountered... Retrying request to {request_path}. Attempt {attempt + 1} of {max_retry_attempts}")
                        backoff_sleep = self.calculate_backoff_sleep(attempt, max_retry_backoff_seconds)
                        time.sleep(backoff_sleep)

    def get(
        self, path: str,
        body: Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[str], str]] = None,
        max_retry_attempts: Optional[int] = None,
        max_retry_backoff_seconds: Optional[int] = None
    ) -> Any:
        content_type = None
        if body is not None:
            content_type = 'application/json'
        res = self.send_request(
            http_method=requests.get,
            path=path,
            body=body,
            content_type=content_type,
            max_retry_attempts=max_retry_attempts,
            max_retry_backoff_seconds=max_retry_backoff_seconds
        )
        return res

    def post(
        self,
        path: str,
        body: Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[str], str]] = None,
        content_type: Optional[str] = 'application/json',
        max_retry_attempts: Optional[int] = None,
        max_retry_backoff_seconds: Optional[int] = None
    ) -> Any:
        return self.send_request(
            http_method=requests.post,
            path=path,
            body=body,
            content_type=content_type,
            max_retry_attempts=max_retry_attempts,
            max_retry_backoff_seconds=max_retry_backoff_seconds
        )

    def put(
        self,
        path: str,
        body: Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[str], str]] = None,
        content_type: Optional[str] = None,
        max_retry_attempts: Optional[int] = None,
        max_retry_backoff_seconds: Optional[int] = None
    ) -> Any:
        if body is not None:
            content_type = 'application/json'
        return self.send_request(
            http_method=requests.put,
            path=path,
            body=body,
            content_type=content_type,
            max_retry_attempts=max_retry_attempts,
            max_retry_backoff_seconds=max_retry_backoff_seconds
        )

    def delete(
        self,
        path: str,
        body: Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[str]]] = None,
        max_retry_attempts: Optional[int] = None,
        max_retry_backoff_seconds: Optional[int] = None
    ) -> Any:
        return self.send_request(
            http_method=requests.delete,
            path=path,
            body=body,
            max_retry_attempts=max_retry_attempts,
            max_retry_backoff_seconds=max_retry_backoff_seconds
        )

    @staticmethod
    def __to_json(
        request: requests.Response
    ) -> Any:
        if request.content == b'':
            return request
        return request.json()

    @staticmethod
    def __validate(
        request: requests.Response
    ) -> Any:
        try:
            request.raise_for_status()
            return HttpRequests.__to_json(request)
        except requests.exceptions.HTTPError as err:
            convert_to_marqo_web_error_and_raise(response=request, err=err)

    def calculate_backoff_sleep(self, attempt: int, cap: int) -> float:
        """Calculates the backoff sleep time for a given attempt

        Args:
            attempt (int): the attempt number

        Returns:
            float: the backoff sleep time
        """
        return min(
            cap * 1000, # convert to milliseconds
            (2 ** attempt) * 10 # start at 10ms for first attempt
        ) / 1000 # convert to seconds

def convert_to_marqo_web_error_and_raise(response: requests.Response, err: requests.exceptions.HTTPError):
    """Translates OpenSearch errors into Marqo errors, which are then raised

    If the incoming OpenSearch error can't be matched, a default catch all
    MarqoWebError is raised

    Raises:
        MarqoWebError - some type of Marqo Web error
    """

    try:
        response_dict = response.json()
    except JSONDecodeError:
        raise_catchall_http_as_marqo_error(response=response, err=err)

    try:
        open_search_error_type = response_dict["error"]["type"]

        if open_search_error_type == "index_not_found_exception":
            raise IndexNotFoundError(message=f"Index `{response_dict['error']['index']}` not found.") from err
        elif open_search_error_type == "resource_already_exists_exception" and "index" in response_dict["error"]["reason"]:
            raise IndexAlreadyExistsError(message=f"Index `{response_dict['error']['index']}` already exists") from err
        elif open_search_error_type == "invalid_index_name_exception":
            raise InvalidIndexNameError(
                message=f"{response_dict['error']['reason'].replace('[','`').replace(']','`')}"
            ) from err
        elif open_search_error_type == "parsing_exception":
            reason = response_dict["error"]["reason"].lower()
            if "knn" in reason and "filter" in reason:
                raise HardwareCompatabilityError(
                    message=f"Filtering is not yet supported for arm-based architectures"
                ) from err
        elif open_search_error_type == "illegal_argument_exception":
            reason = response_dict["error"]["reason"].lower()
            if "limit of total fields" in reason and "exceeded" in reason:
                raise IndexMaxFieldsError(message="Exceeded maximum number of "
                                                  "allowed fields for this index.")
        elif open_search_error_type == "cluster_block_exception":
            raise DiskWatermarkBreachError(message="Your Marqo storage is full. "
                                           "Please delete documents before attempting to add any new documents.")
    except KeyError:
        pass

    # for throttling
    if response.status_code == 429:
        raise TooManyRequestsError(
            message="Marqo-OS received too many requests! "
                    "Please try reducing the frequency of add_documents and update_documents calls.")
    
    try:
        if response_dict["found"] is False:
            raise DocumentNotFoundError(
                message=f"Document `{response_dict['_id']}` not found."
            ) from err
    except KeyError:
        pass

    raise_catchall_http_as_marqo_error(response=response, err=err)


def raise_catchall_http_as_marqo_error(response: requests.Response, err: requests.exceptions.HTTPError) -> None:
    """Raises a generic MarqoWebError for a given HTTPError"""
    try:
        response_msg = response.json()
    except JSONDecodeError:
        response_msg = response.text

    raise MarqoWebError(message=response_msg, code="unhandled_backend_error",
                        error_type="backend_error", status_code=response.status_code) from err
