from typing import List

from marqo.base_model import ImmutableStrictBaseModel


class MemoryProfile(ImmutableStrictBaseModel):
    memory_used: float
    stats: List[str]
