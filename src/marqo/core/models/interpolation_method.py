from marqo.case_insensitive_enum import CaseInsensitiveEnum


class InterpolationMethod(str, CaseInsensitiveEnum):
    LERP = "lerp"
    NLERP = "nlerp"
    SLERP = "slerp"
