from enum import Enum


class Modality(str, Enum):
    """language/TEXT is widely used by language models so we keep it as internal use for now."""

    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
