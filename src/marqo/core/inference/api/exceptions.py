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
