from http import HTTPStatus

from marqo.exceptions import (
    MarqoError,
    InvalidArgumentError,
)


class InternalError(MarqoError):
    pass


class ApplicationNotInitializedError(MarqoError):
    """
    This exception is raised when the Vespa application is not bootstrapped when receiving
    index operation requests.
    """
    pass


class ApplicationRollbackError(MarqoError):
    pass


class IndexExistsError(MarqoError):
    pass


class IndexNotFoundError(MarqoError):
    pass


class OperationConflictError(MarqoError):
    pass


class BackendCommunicationError(MarqoError):
    pass


class ZookeeperLockNotAcquiredError(MarqoError):
    pass


class ParsingError(MarqoError):
    pass


class VespaDocumentParsingError(ParsingError):
    pass


class MarqoDocumentParsingError(ParsingError, InvalidArgumentError):
    pass


class InvalidDataTypeError(MarqoDocumentParsingError):
    pass


class InvalidDataRangeError(MarqoDocumentParsingError):
    pass


class InvalidFieldNameError(MarqoDocumentParsingError):
    pass


class FilterStringParsingError(ParsingError, InvalidArgumentError):
    pass


class InvalidTensorFieldError(MarqoDocumentParsingError):
    pass


class UnsupportedFeatureError(InvalidArgumentError):
    pass


class ZeroMagnitudeVectorError(InvalidArgumentError):
    pass


class FieldTypeMismatchError(InvalidArgumentError):
    pass


class AddDocumentsError(Exception):
    status_code: int = int(HTTPStatus.BAD_REQUEST)
    error_code: str = 'invalid_argument'
    error_message: str

    def __init__(self, error_message: str,
                 error_code: str = 'invalid_argument',
                 status_code: int = int(HTTPStatus.BAD_REQUEST)) -> None:
        self.error_code = error_code
        self.error_message = error_message
        self.status_code = int(status_code)

    def __str__(self) -> str:
        return f'{self.error_code}: {self.error_message}'


class DuplicateDocumentError(AddDocumentsError):
    pass


class TooManyFieldsError(MarqoError):
    pass


# TODO inherit InferenceError?
class DeviceError(MarqoError):
    pass


class CudaDeviceNotAvailableError(DeviceError):
    pass


class CudaOutOfMemoryError(DeviceError):
    pass
