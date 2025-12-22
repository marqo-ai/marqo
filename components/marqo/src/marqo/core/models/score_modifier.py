from enum import Enum
from marqo.base_model import StrictBaseModel


class ScoreModifierType(Enum):
    Multiply = 'multiply'
    Add = 'add'


class ScoreModifier(StrictBaseModel):
    field: str
    weight: float
    type: ScoreModifierType