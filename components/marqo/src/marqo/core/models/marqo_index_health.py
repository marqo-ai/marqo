from enum import Enum
from types import DynamicClassAttribute
from typing import Optional

from marqo.base_model import StrictBaseModel


class HealthStatus(Enum):
    Green = ("green", 1)
    Yellow = ("yellow", 2)
    Red = ("red", 3)

    @DynamicClassAttribute
    def value(self):
        return super().value[0]

    @property
    def priority(self):
        return super().value[1]

    def __gt__(self, other):
        return self.priority > other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.priority == other.priority


class InferenceHealthStatus(StrictBaseModel):
    status: HealthStatus


class VespaHealthStatus(StrictBaseModel):
    status: HealthStatus
    memory_is_available: Optional[bool]
    storage_is_available: Optional[bool]


class MarqoHealthStatus(StrictBaseModel):
    status: HealthStatus
    inference: InferenceHealthStatus
    backend: VespaHealthStatus
