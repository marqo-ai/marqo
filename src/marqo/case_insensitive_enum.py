from enum import Enum
from typing import Optional


class CaseInsensitiveEnum(Enum):
    @classmethod
    def _missing_(cls, value: str) -> Optional['CaseInsensitiveEnum']:
        value = value.lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None
