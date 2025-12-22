from typing import Optional


class AppBaseError(Exception):
    """
    Base class for all Marqo errors.
    """

    def __init__(
        self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.cause = cause
