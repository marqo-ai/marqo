from typing import Optional


class S2InferenceError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if Optional is not None:
            self.message = message
            super().__init__(self.message)


class ChunkerError(S2InferenceError):
    pass


class VectoriseError(S2InferenceError):
    pass

class EnrichmentError(S2InferenceError):
    pass
