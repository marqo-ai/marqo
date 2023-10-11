from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import Field as PydanticField
from pydantic import PrivateAttr

from marqo.core.models.strict_base_model import StrictBaseModel


class IndexType(Enum):
    Structured = 'structured'
    Unstructured = 'unstructured'


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


class Field(StrictBaseModel):
    name: str
    type: FieldType
    features: List[FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]
    lexical_field_name: Optional[str]
    filter_field_name: Optional[str]


class TensorField(StrictBaseModel):
    name: str
    chunk_field_name: Optional[str]
    embeddings_field_name: Optional[str]


class HnswConfig(StrictBaseModel):
    ef_construction: int
    m: int


class Model(StrictBaseModel):
    name: str
    properties: Optional[Dict[str, Any]]


class MarqoIndex(StrictBaseModel):
    name: str
    type: IndexType
    model: Model
    distance_metric: DistanceMetric
    vector_numeric_type: VectorNumericType
    hnsw_config: HnswConfig
    fields: Optional[List[Field]]
    tensor_fields: Optional[List[TensorField]]
    model_enable_cache: bool = PydanticField(default=False, allow_mutation=False)
    _cache: Dict[str, Any] = PrivateAttr()

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        self._cache = dict()

    @property
    def lexical_fields(self) -> List[str]:
        return self._cache_or_get('lexical_fields',
                                  lambda: [field.lexical_field_name for field in self.fields if
                                           field.lexical_field_name is not None]
                                  )

    @property
    def score_modifier_fields(self) -> List[str]:
        return self._cache_or_get('score_modifier_fields',
                                  lambda: [field.name for field in self.fields if
                                           FieldFeature.ScoreModifier in field.features]
                                  )

    @property
    def field_map(self) -> Dict[str, Field]:
        return self._cache_or_get('field_map',
                                  lambda: {field.name: field for field in self.fields}
                                  )

    @property
    def tensor_field_map(self) -> Dict[str, TensorField]:
        return self._cache_or_get('tensor_field_map',
                                  lambda: {tensor_field.name: tensor_field for tensor_field in self.tensor_fields}
                                  )

    def _cache_or_get(self, key: str, func):
        if self.model_enable_cache:
            if key not in self._cache:
                self._cache[key] = func()
            return self._cache[key]
        else:
            return func()

    def copy_with_caching(self):
        model_dict = self.dict()
        del model_dict['model_enable_cache']

        return MarqoIndex(**model_dict, model_enable_cache=True)


if __name__ == '__main__':
    marqo_index = MarqoIndex(
        name='index1',
        model=Model(name='ViT-B/32'),
        distance_metric=DistanceMetric.PrenormalizedAnguar,
        type=IndexType.Structured,
        vector_numeric_type=VectorNumericType.Float,
        hnsw_config=HnswConfig(ef_construction=100, m=16),
        fields=[
            Field(name='title', type=FieldType.Text),
            Field(name='description', type=FieldType.Text),
            Field(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
        ],
        tensor_fields=[
            TensorField(name='title'),
            TensorField(name='description')
        ],
        model_enable_cache=True,
    )
    marqo_index.lexical_fields

    mq1 = marqo_index
    mq2 = mq1.copy_with_caching()
    pass
