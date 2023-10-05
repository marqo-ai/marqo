from typing import Optional


class MarqoErrorMeta(type):
    """
    This metaclass adds a default __init__ method that takes an optional `message: str` and
    an optional `cause: Exception` to subclasses of `MarqoError`.
    """

    def __new__(cls, name, bases, attrs):
        if name != 'MarqoError':
            if not any(issubclass(base, MarqoError) for base in bases):
                raise TypeError(f"Class {name} must inherit from {MarqoError.__name__}. "
                                f"Do not use this metaclass directly. Inherit from {MarqoError.__name__} instead.")

        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):

        if '__init__' not in attrs:
            def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None):
                super(cls, self).__init__(message, cause)

            setattr(cls, '__init__', __init__)
        super().__init__(name, bases, attrs)


class MarqoError(Exception, metaclass=MarqoErrorMeta):
    """
    Base class for all Marqo errors.
    """

    def __init__(
            self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.cause = cause


class InternalError(MarqoError):
    pass


class InvalidArgumentError(MarqoError):
    pass
