from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel


class FieldType(Enum):
    Text = 'text'
    Bool = 'bool'
    Int = 'int'
    Float = 'float'
    ArrayText = 'array<text>'
    ArrayBool = 'array<bool>'
    ArrayInt = 'array<int>'
    ArrayFloat = 'array<float>'
    ImagePointer = 'image_pointer'
    MultimodalCombination = 'multimodal_combination'


class VectorNumericType(Enum):
    Float = 'float'
    Bfloat16 = 'bfloat16'


class FieldFeature(Enum):
    TensorSearch = 'tensor_search'
    LexicalSearch = 'lexical_search'
    ScoreModifier = 'score_modifier'
    Filter = 'filter'


class DistanceMetric(Enum):
    PrenormalizedAnguar = 'prenormalized-angular'


class Field(BaseModel):
    name: str
    type: FieldType
    features: List[FieldFeature]
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


class MarqoIndex(BaseModel):
    name: str
    model: str
    distance_metric: DistanceMetric
    vector_numeric_type: VectorNumericType
    hnsw_config: HnswConfig
    fields: List[Field]
    tensor_fields: List[TensorField]

    @property
    def lexical_fields(self) -> List[str]:
        return [field.lexical_field_name for field in self.fields if
                field.lexical_field_name is not None]

    @property
    def score_modifier_fields(self) -> List[str]:
        return [field.name for field in self.fields if
                FieldFeature.ScoreModifier in field.features]
