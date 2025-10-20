from typing import Any, Optional

from fastapi import status as http


class AppError(Exception):
    http_status: int = http.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"
    extras: dict[str, Any]

    def __init__(
        self, message: Optional[str] = None, *, extras: Optional[dict[str, Any]] = None
    ):
        super().__init__(message or self.__class__.__name__)
        self.extras = extras or {}
