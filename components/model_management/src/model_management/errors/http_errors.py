from fastapi import status as http

from .base import AppError


class ValidationAppError(AppError):
    http_status = http.HTTP_400_BAD_REQUEST
    code = "VALIDATION_ERROR"


class InvalidArgumentError(AppError):
    http_status = http.HTTP_400_BAD_REQUEST
    code = "INVALID_ARGUMENT"


class ForbiddenError(AppError):
    http_status = http.HTTP_403_FORBIDDEN
    code = "FORBIDDEN"


class NotFoundError(AppError):
    http_status = http.HTTP_404_NOT_FOUND
    code = "NOT_FOUND"


class OperationConflictError(AppError):  # your preferred name
    http_status = http.HTTP_409_CONFLICT
    code = "OPERATION_CONFLICT"


class RateLimitedError(AppError):
    http_status = http.HTTP_429_TOO_MANY_REQUESTS
    code = "RATE_LIMITED"


class InternalServerError(AppError):
    http_status = http.HTTP_500_INTERNAL_SERVER_ERROR
    code = "INTERNAL_ERROR"


class DependencyUnavailableError(AppError):
    http_status = http.HTTP_503_SERVICE_UNAVAILABLE
    code = "DEPENDENCY_UNAVAILABLE"


class DependencyTimeoutError(AppError):
    http_status = http.HTTP_504_GATEWAY_TIMEOUT
    code = "DEPENDENCY_TIMEOUT"


class DependencyBadGatewayError(AppError):
    http_status = http.HTTP_502_BAD_GATEWAY
    code = "DEPENDENCY_BAD_GATEWAY"
