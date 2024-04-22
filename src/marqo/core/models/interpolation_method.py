from enum import Enum


class InterpolationMethod(str, Enum):
    LERP = "lerp"
    NLERP = "nlerp"
    SLERP = "slerp"
