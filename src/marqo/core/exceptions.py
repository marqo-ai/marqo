from marqo.exceptions import (
    MarqoError,
    InternalError,
    InvalidArgumentError,
    NotFoundError,
    AlreadyExistsError,
)


class IndexExistsError(AlreadyExistsError):
    pass


class IndexNotFoundError(NotFoundError):
    pass


class ParsingError(MarqoError):
    """
    Not subclass of any specific exception. Subclasses of this error have different behaviors.
    """
    pass


class VespaDocumentParsingError(ParsingError):
    """
    Not subclass of any specific exception. Will be caught by generic handler.
    """
    pass


class MarqoDocumentParsingError(ParsingError, InvalidArgumentError):
    pass


class InvalidDataTypeError(MarqoDocumentParsingError):
    pass


class InvalidFieldNameError(MarqoDocumentParsingError):
    pass


class FilterStringParsingError(ParsingError, InvalidArgumentError):
    pass
