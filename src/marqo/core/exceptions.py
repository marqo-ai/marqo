from marqo.exceptions import MarqoError


class IndexExistsError(MarqoError):
    pass


class IndexNotFoundError(MarqoError):
    pass
