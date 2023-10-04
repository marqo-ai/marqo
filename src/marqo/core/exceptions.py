from typing import Optional


class MarqoError(Exception):
    def __init__(
            self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.cause = cause


class MarqoErrorMeta(type):
    def __init__(cls, name, bases, attrs):
        if '__init__' not in attrs:
            def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None):
                super(cls, self).__init__(message, cause)

            setattr(cls, '__init__', __init__)
        super().__init__(name, bases, attrs)


class MarqoInternalError(MarqoError, metaclass=MarqoErrorMeta):
    pass


class MarqoIndexExistsError(MarqoError, metaclass=MarqoErrorMeta):
    pass


class MarqoIndexNotFoundError(MarqoError, metaclass=MarqoErrorMeta):
    pass
