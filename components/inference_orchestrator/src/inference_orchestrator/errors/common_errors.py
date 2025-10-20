from .base_error import AppBaseError


class EnvironmentVariableParsingError(AppBaseError):
    """
    Raised when there is an error parsing environment variables.
    """

    pass


class StartupSanityCheckError(AppBaseError):
    """
    Raised when the startup sanity checks fail.
    """

    pass
