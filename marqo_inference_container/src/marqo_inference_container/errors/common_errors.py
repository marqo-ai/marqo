from .base_error import AppBaseError


class InvalidArgumentError(AppBaseError):
    """Raised when an argument provided to a function is invalid"""
    pass


class EnvironmentVariableParsingError(AppBaseError):
    """Raised when there is an error parsing environment variables"""
    pass


class InvalidModelPropertiesError(AppBaseError):
    """Raised when the model properties provided are invalid or incomplete"""
    pass


class StartupSanityCheckError(AppBaseError):
    """Raised when a sanity check during startup fails"""
    pass


class InternalError(AppBaseError):
    """Raised when an internal error occurs"""
    pass