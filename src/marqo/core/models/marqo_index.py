from enum import Enum
from typing import List, Optional, Dict, Any, Set

from pydantic import Field as PydanticField
from pydantic import PrivateAttr

from marqo.core.models.strict_base_model import StrictBaseModel
from marqo.exceptions import InvalidArgumentError
from marqo.logging import get_logger
from marqo.s2_inference import s2_inference
from marqo.s2_inference.errors import UnknownModelError

logger = get_logger(__name__)


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
    Euclidean = 'euclidean'
    Angular = 'angular'
    DotProduct = 'dotproduct'
    PrenormalizedAnguar = 'prenormalized-angular'
    Geodegrees = 'geodegrees'
    Hamming = 'hamming'


class TextSplitMethod(Enum):
    Character = 'character'
    Word = 'word'
    Sentence = 'sentence'
    Passage = 'passage'


class PatchMethod(Enum):
    Simple = 'simple'
    Frcnn = 'frcnn'


class Field(StrictBaseModel):
    name: str
    type: FieldType
    features: List[FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]
    lexical_field_name: Optional[str]
    filter_field_name: Optional[str]


class TensorField(StrictBaseModel):
    """
    A tensor field that has a corresponding field.

    chunk_field_name and embeddings_field_name must be unique across all tensor fields.
    """
    name: str
    chunk_field_name: Optional[str]
    embeddings_field_name: Optional[str]


class HnswConfig(StrictBaseModel):
    ef_construction: int
    m: int


class TextPreProcessing(StrictBaseModel):
    split_length: int
    split_overlap: int
    split_method: TextSplitMethod


class ImagePreProcessing(StrictBaseModel):
    patch_method: Optional[PatchMethod]


class Model(StrictBaseModel):
    name: str
    properties: Optional[Dict[str, Any]]
    custom: bool = False

    def get_dimension(self):
        self._update_model_properties_from_registry()
        try:
            return self.properties['dimensions']
        except KeyError:
            raise InvalidArgumentError(
                "The given model properties does not contain a 'dimensions' key"
            )

    def get_properties(self):
        """
        Get model properties. Try to update model properties from the registry first if model properties
        are not populated.

        Raises:
            InvalidArgumentError: If model properties are not populated and the model is not found in the registry.
            UnknownModelError: If model properties are not populated and the model is not found in the registry.
        """
        self._update_model_properties_from_registry()
        return self.properties

    def _update_model_properties_from_registry(self) -> None:
        if not self.properties:
            logger.debug('Model properties not populated. Trying to update from registry')

            model_name = self.name
            try:
                self.properties = s2_inference.get_model_properties_from_registry(model_name)
            except UnknownModelError:
                raise InvalidArgumentError(
                    f'Could not find model properties for model={model_name}. '
                    f'Please check that the model name is correct. '
                    f'Please provide model_properties if the model is a custom model and is not supported by default')


class MarqoIndex(StrictBaseModel):
    name: str
    type: IndexType
    model: Model
    normalize_embeddings: bool
    text_preprocessing: TextPreProcessing
    image_preprocessing: ImagePreProcessing
    treat_urls_and_pointers_as_images: Optional[bool]
    distance_metric: DistanceMetric
    vector_numeric_type: VectorNumericType
    hnsw_config: HnswConfig
    fields: Optional[List[Field]]  # all fields, including tensor fields
    tensor_fields: Optional[List[TensorField]]
    model_enable_cache: bool = PydanticField(default=False, allow_mutation=False)
    _cache: Dict[str, Any] = PrivateAttr()

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        self._cache = dict()

    @property
    def lexical_fields_names(self) -> Set[str]:
        return self._cache_or_get('lexical_fields',
                                  lambda: {field.lexical_field_name for field in self.fields if
                                           field.lexical_field_name is not None}
                                  )

    @property
    def score_modifier_fields_names(self) -> Set[str]:
        return self._cache_or_get('score_modifier_fields',
                                  lambda: {field.name for field in self.fields if
                                           FieldFeature.ScoreModifier in field.features}
                                  )

    @property
    def field_map(self) -> Dict[str, Field]:
        """
        A map from field name to the field.
        """
        return self._cache_or_get('field_map',
                                  lambda: {field.name: field for field in self.fields}
                                  )

    @property
    def all_field_map(self) -> Dict[str, Field]:
        """
        A map from field name, lexical name and filter name to the field.
        """
        return self._cache_or_get('all_field_map',
                                  lambda: {
                                      **{field.name: field for field in self.fields + self.tensor_fields},
                                      **{field.lexical_field_name: field for field in self.fields if
                                         field.lexical_field_name is not None},
                                      **{field.filter_field_name: field for field in self.fields if
                                         field.filter_field_name is not None}
                                  }
                                  )

    @property
    def tensor_field_map(self) -> Dict[str, TensorField]:
        return self._cache_or_get('tensor_field_map',
                                  lambda: {tensor_field.name: tensor_field for tensor_field in self.tensor_fields}
                                  )

    @property
    def tensor_subfield_map(self) -> Dict[str, TensorField]:
        """
        A map from tensor chunk and embeddings field name to the tensor field.
        """

        def generate():
            the_map = dict()
            for tensor_field in self.tensor_fields:
                if tensor_field.chunk_field_name is not None:
                    if tensor_field.chunk_field_name in the_map:
                        raise ValueError(
                            f"Duplicate chunk field name {tensor_field.chunk_field_name} "
                            f"for tensor field {tensor_field.name}"
                        )
                    the_map[tensor_field.chunk_field_name] = tensor_field
                if tensor_field.embeddings_field_name is not None:
                    if tensor_field.embeddings_field_name in the_map:
                        raise ValueError(
                            f"Duplicate embeddings field name {tensor_field.embeddings_field_name} "
                            f"for tensor field {tensor_field.name}"
                        )
                    the_map[tensor_field.embeddings_field_name] = tensor_field

            return the_map

        return self._cache_or_get('tensor_subfield_map', generate)

    @property
    def field_map_by_type(self) -> Dict[FieldType, List[Field]]:
        return self._cache_or_get('field_map_by_type',
                                  lambda: {field_type: [field for field in self.fields if field.type == field_type]
                                           for field_type in FieldType}
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

        copied = MarqoIndex(**model_dict, model_enable_cache=True)

        # Retrieve all properties to populate cache
        for name, value in vars(MarqoIndex).items():
            if isinstance(value, property):
                getattr(copied, name)

        return copied
