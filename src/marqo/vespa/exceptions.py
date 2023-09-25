from typing import Optional

from httpx import HTTPStatusError


class VespaError(Exception):
    def __init__(
            self, cause: Exception
    ):
        super().__init__(str(cause))
        self.cause = cause


class VespaStatusError(VespaError):
    def __init__(self, cause: HTTPStatusError):
        super().__init__(cause)

    @property
    def status_code(self) -> int:
        try:
            return self.cause.response.status_code
        except Exception as e:
            raise Exception(f"Could not get status code from {self.cause}") from e