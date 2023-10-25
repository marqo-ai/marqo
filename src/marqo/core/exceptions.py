from marqo.exceptions import MarqoError


class IndexExistsError(MarqoError):
    pass


class IndexNotFoundError(MarqoError):
    pass


class DocumentParsingError(MarqoError):
    pass


class InvalidDataTypeError(DocumentParsingError):
    pass


class InvalidFieldNameError(DocumentParsingError):
    pass
