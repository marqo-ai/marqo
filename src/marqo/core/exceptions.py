from typing import Optional


class MarqoError(Exception):
    def __init__(
            self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.cause = cause


class MarqoInternalError(MarqoError):
    def __init__(
            self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message, cause)
