from marqo.exceptions import MarqoError


class VespaError(MarqoError):
    pass


class VespaStatusError(VespaError):

    @property
    def status_code(self) -> int:
        try:
            return self.cause.response.status_code
        except Exception as e:
            raise Exception(f"Could not get status code from {self.cause}") from e

    def __str__(self) -> str:
        try:
            return f"{self.status_code}: {self.message}"
        except:
            return super().__str__()


class VespaTimeoutError(VespaStatusError):
    """
    Raised when Vespa responds with a timeout error.
    """
    pass


class InvalidVespaApplicationError(VespaError):
    pass
