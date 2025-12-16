from marqo.exceptions import MarqoError


class InferenceError(MarqoError):
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
    """Raised when the specified model is not recognized or supported"""
    pass


class InvalidModelPropertiesError(InferenceError):
    """Raised when the provided model properties are invalid or not supported"""
    pass