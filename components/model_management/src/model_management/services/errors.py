"""
Service layer domain errors.

These errors represent business logic failures in the service layer.
They should be mapped to appropriate API errors by the error mapper in the API layer.
"""


class ServiceError(Exception):
    """Base class for all service layer errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelDownloadFailedError(ServiceError):
    """Raised when model files cannot be downloaded from the specified source."""

    pass


class TritonCommunicationError(ServiceError):
    """Raised when communication with Triton server fails."""

    pass


class TritonModelLoadError(ServiceError):
    """Raised when Triton fails to load a model."""

    pass


class TritonModelUnloadError(ServiceError):
    """Raised when Triton fails to unload a model."""

    pass


class ModelOperationInProgressError(ServiceError):
    """Raised when a model operation is requested while another operation is in progress."""

    pass


class InternalServerError(ServiceError):
    """Raised when there is an unexpected error within the service."""

    pass
