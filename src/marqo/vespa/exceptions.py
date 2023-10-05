from httpx import HTTPStatusError

from marqo.exceptions import MarqoError


class VespaError(MarqoError):
    pass


class VespaStatusError(VespaError):
    def __init__(self, cause: HTTPStatusError):
        super().__init__(cause=cause)

    @property
    def status_code(self) -> int:
        try:
            return self.cause.response.status_code
        except Exception as e:
            raise Exception(f"Could not get status code from {self.cause}") from e
