from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class IndexType(Enum):
    Typed = 'typed'
    Dynamic = 'dynamic'


class FieldType(Enum):
    Text = 'text'
    Bool = 'bool'
    Int = 'int'
    Float = 'float'
    ArrayText = 'array<text>'
    ArrayInt = 'array<int>'
    ArrayFloat = 'array<float>'
    ImagePointer = 'image_pointer'
    MultimodalCombination = 'multimodal_combination'


class VectorNumericType(Enum):
    Float = 'float'
    Bfloat16 = 'bfloat16'


class FieldFeature(Enum):
    LexicalSearch = 'lexical_search'
    ScoreModifier = 'score_modifier'
    Filter = 'filter'


class DistanceMetric(Enum):
    PrenormalizedAnguar = 'prenormalized-angular'


class Field(BaseModel):
    name: str
    type: FieldType
    features: List[FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]
    lexical_field_name: Optional[str]
    filter_field_name: Optional[str]


class TensorField(BaseModel):
    name: str
    chunk_field_name: Optional[str]
    embeddings_field_name: Optional[str]


class HnswConfig(BaseModel):
    ef_construction: int
    m: int


class Model(BaseModel):
    name: str
    properties: Optional[Dict[str, Any]]


class MarqoIndex(BaseModel):
    name: str
    type: IndexType
    model: Model
    distance_metric: DistanceMetric
    vector_numeric_type: VectorNumericType
    hnsw_config: HnswConfig
    fields: Optional[List[Field]]
    tensor_fields: Optional[List[TensorField]]

    @property
    def lexical_fields(self) -> List[str]:
        return [field.lexical_field_name for field in self.fields if
                field.lexical_field_name is not None]

    @property
    def score_modifier_fields(self) -> List[str]:
        return [field.name for field in self.fields if
                FieldFeature.ScoreModifier in field.features]
