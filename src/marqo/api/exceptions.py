import json
from http import HTTPStatus

from requests import Response


# TODO: DELETE
class MarqoError(Exception):
    """Generic class for Marqo error handling
    Can be used for value errors etc.
    These will be caught and returned to the user as 5xx internal errors

    """
    code = None

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'MarqoError {self.__class__.__name__} Message: {self.message}'


class EnvVarError(MarqoError):
    code = "env_var_error"

    def __init__(self, message: str):
        self.message = message


# TODO: DELETE
class MarqoApiError(MarqoError):
    """Error sent by Marqo API"""

    def __init__(self, error: str, request: Response) -> None:
        self.status_code = request.status_code
        self.code = None
        self.link = None
        self.type = None

        if request.text:
            json_data = json.loads(request.text)
            self.message = json_data
            self.code = json_data.get('status')
            self.link = ''
            self.type = ''
            if 'error' in json_data and 'root_cause' in json_data["error"] \
                    and len(json_data.get('error').get('root_cause')) > 0:
                self.type = json_data.get('error').get('root_cause')[0].get('type')
        else:
            self.message = error
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.code and self.link:
            return f'MarqoApiError. Error code: {self.code}. Error message: {self.message} Error documentation: {self.link} Error type: {self.type}'

        return f'MarqoApiError. {self.message}'


# MARQO WEB ERROR

class MarqoWebError(Exception):
    status_code: int = 500
    error_type: str = None
    message: str = None
    code: str = None
    link: str = ""
    base_message = ("Please create an issue on Marqo's GitHub repo"
                    " (https://github.com/marqo-ai/marqo/issues) "
                    "if this problem persists.")

    def __init__(self, message: str, status_code: int = None,
                 error_type: str = None, code: str = None,
                 link: str = None) -> None:
        self.message = f"{message}\n{self.base_message}"

        if self.status_code is None:
            self.status_code = status_code

        if self.error_type is None:
            self.error_type = error_type

        if self.code is None:
            self.code = code

        if self.link is None:
            self.link = link
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: {self.message}'


# ---MARQO USER ERRORS---


class __InvalidRequestError(MarqoWebError):
    """abstract error"""

    def __init__(self, message: str, link: str = None):
        self.message = message
        self.link = link

    error_type = "invalid_request"


class TooManyRequestsError(__InvalidRequestError):
    code = "too_many_requests"
    status_code = HTTPStatus.TOO_MANY_REQUESTS


class VectorStoreTimeoutError(__InvalidRequestError):
    code = "vector_store_timeout"
    status_code = HTTPStatus.GATEWAY_TIMEOUT


class IndexAlreadyExistsError(__InvalidRequestError):
    code = "index_already_exists"
    status_code = HTTPStatus.CONFLICT


class ModelCacheManagementError(__InvalidRequestError):
    code = "model_cache_management_error"
    status_code = HTTPStatus.CONFLICT


class IndexNotFoundError(__InvalidRequestError):
    code = "index_not_found"
    status_code = HTTPStatus.NOT_FOUND


class InvalidIndexNameError(__InvalidRequestError):
    code = "invalid_index_name"
    status_code = HTTPStatus.BAD_REQUEST


class InvalidDocumentIdError(__InvalidRequestError):
    code = "invalid_document_id"
    status_code = HTTPStatus.BAD_REQUEST


class InvalidFieldNameError(__InvalidRequestError):
    code = "invalid_field_name"
    status_code = HTTPStatus.BAD_REQUEST


class InvalidArgError(__InvalidRequestError):
    code = "invalid_argument"
    status_code = HTTPStatus.BAD_REQUEST


class IllegalRequestedDocCount(__InvalidRequestError):
    code = "illegal_requested_doc_count"
    status_code = HTTPStatus.BAD_REQUEST


class DocTooLargeError(__InvalidRequestError):
    code = "doc_too_large"
    status_code = HTTPStatus.BAD_REQUEST


class BadRequestError(__InvalidRequestError):
    code = "bad_request"
    status_code = HTTPStatus.BAD_REQUEST


class DocumentNotFoundError(__InvalidRequestError):
    code = "document_not_found"
    status_code = HTTPStatus.NOT_FOUND


class NonTensorIndexError(__InvalidRequestError):
    """Error trying to use a non-tensor OpenSearch index like a tensor one"""
    code = "document_not_found"
    status_code = HTTPStatus.BAD_REQUEST


class HardwareCompatabilityError(__InvalidRequestError):
    """Error when a request incorrectly assumes that the server has a certain
    hardware configuration"""
    code = "hardware_compatability_error"
    status_code = HTTPStatus.BAD_REQUEST


class IndexMaxFieldsError(__InvalidRequestError):
    """Error when attempting to index a document that increases the indexes' number of
    fields above the index limit"""
    code = "index_max_fields_error"
    status_code = HTTPStatus.BAD_REQUEST


class ModelNotInCacheError(__InvalidRequestError):
    code = "model_not_in_cache"
    status_code = HTTPStatus.NOT_FOUND


# ---MARQO INTERNAL ERROR---


class InternalError(MarqoWebError):
    error_type = "internal"
    code = "internal"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class BackendCommunicationError(InternalError):
    """Error when connecting to Vespa"""
    code = "backend_communication_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class BackendDataParsingError(InternalError):
    """Error when parsing data from Vespa"""
    code = "backend_data_parsing_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class BackendTimeoutError(InternalError):
    """Error when Marqo operation takes longer than expected"""
    code = "backend_timeout_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class BatchInferenceSizeError(InternalError):
    """Error when batch inference does not return expected size"""
    code = "batch_inference_size_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class ConfigurationError(InternalError):
    code = "server_configuration_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR


class DiskWatermarkBreachError(InternalError):
    """Error when OpenSearch disk watermark is breached"""
    code = "disk_watermark_breach_error"
    status_code = HTTPStatus.BAD_REQUEST
