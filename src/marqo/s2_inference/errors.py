from typing import Optional


class S2InferenceError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if Optional is not None:
            self.message = message
            super().__init__(self.message)

class TruncatedImageError(S2InferenceError):
    pass


class ChunkerError(S2InferenceError):
    pass


class ChunkerMethodProcessError(S2InferenceError):
    pass


class VectoriseError(S2InferenceError):
    pass


class InvalidModelPropertiesError(S2InferenceError):
    pass


class UnknownModelError(S2InferenceError):
    pass


class ModelLoadError(S2InferenceError):
    pass


class ModelDownloadError(S2InferenceError):
    pass


class RerankerError(S2InferenceError):
    pass


class RerankerImageError(S2InferenceError):
    pass


class RerankerNameError(S2InferenceError):
    pass


class ModelNotInCacheError(S2InferenceError):
    pass