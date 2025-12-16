from enum import Enum


class Modality(str, Enum):
    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
