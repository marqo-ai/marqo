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