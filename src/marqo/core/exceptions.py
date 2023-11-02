from marqo.exceptions import MarqoError


class IndexExistsError(MarqoError):
    pass


class IndexNotFoundError(MarqoError):
    pass


class ParsingError(MarqoError):
    pass


class VespaDocumentParsingError(ParsingError):
    pass


class MarqoDocumentParsingError(ParsingError):
    pass


class InvalidDataTypeError(MarqoDocumentParsingError):
    pass


class InvalidFieldNameError(MarqoDocumentParsingError):
    pass


class FilterStringParsingError(MarqoError):
    pass
