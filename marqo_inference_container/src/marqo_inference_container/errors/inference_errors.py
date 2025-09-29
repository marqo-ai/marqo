from .base_error import AppBaseError


class InferenceError(AppBaseError):
    """A generic root error type for any inference related errors"""
    pass


class ModelError(InferenceError):
    """The root error type of any errors related to embedding models"""
    pass


class PreprocessingError(InferenceError):
    """The root error type of any errors related to content preprocessing"""
    pass


class MediaDownloadError(InferenceError):
    """Raised when media download fails"""
    pass


class UnsupportedModalityError(InferenceError):
    """Raises if a modality is not supported by a specific model"""
    pass


class MediaExceedsMaxSizeError(InferenceError):
    """Raised when the media exceeds the maximum size limit"""
    pass


class MediaMismatchError(InferenceError):
    """Raised when the media does not match the expected type"""
    pass


class UnsupportedModelError(InferenceError):
    """Raised when a model is not supported"""
    pass


class InvalidModelPropertiesError(InferenceError):
    """Raised when the model properties are invalid"""
    pass


class ImageDownloadError(MediaDownloadError):
    """Raised when image download fails"""
    pass
