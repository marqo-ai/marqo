from enum import Enum
from typing import List

from pydantic import BaseModel


class VectorNumericType(Enum):
    Float = "float"
    Bfloat16 = "bfloat16"


class FieldFeature(Enum):
    TensorSearch = "tensor_search"
    LexicalSearch = "lexical_search"
    ScoreModifier = "score_modifier"
    Filter = "filter"


class Field(BaseModel):
    name: str
    type: str
    features: List[FieldFeature]


class MarqoIndex(BaseModel):
    model: str
    vector_numeric_type: VectorNumericType
    fields: List[Field]
