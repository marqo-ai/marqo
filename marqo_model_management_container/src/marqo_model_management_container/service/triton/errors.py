from marqo_model_management_container.errors.base import AppError


class ModelLoadingError(AppError):
    """Raised when there is an error loading a model into Triton."""
    http_status = 400
    code = "MODEL_LOADING_ERROR"


