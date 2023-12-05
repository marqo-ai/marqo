from marqo.exceptions import (
    MarqoError,
    InternalError,
    InvalidArgumentError,
)


class IndexExistsError(MarqoError):
    pass


class IndexNotFoundError(MarqoError):
    pass


class ParsingError(MarqoError):
    pass


class VespaDocumentParsingError(ParsingError):
    pass


class MarqoDocumentParsingError(ParsingError, InvalidArgumentError):
    pass


class InvalidDataTypeError(MarqoDocumentParsingError):
    pass


class InvalidFieldNameError(MarqoDocumentParsingError):
    pass


class FilterStringParsingError(ParsingError, InvalidArgumentError):
    pass
