from enum import Enum
from types import DynamicClassAttribute

from marqo.core.models.strict_base_model import StrictBaseModel


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


class VespaHealthStatus(StrictBaseModel):
    status: HealthStatus


class MarqoHealthStatus(StrictBaseModel):
    status: HealthStatus
    backend: VespaHealthStatus
