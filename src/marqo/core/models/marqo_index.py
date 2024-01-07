import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Union

import pydantic
from pydantic import PrivateAttr, root_validator
from pydantic import ValidationError, validator
from pydantic.error_wrappers import ErrorWrapper
from pydantic.utils import ROOT_KEY

from marqo.base_model import ImmutableStrictBaseModel, StrictBaseModel
from marqo.core import constants
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


class Field(ImmutableStrictBaseModel):
    name: str
    type: FieldType
    features: List[FieldFeature] = []
    lexical_field_name: Optional[str]
    filter_field_name: Optional[str]
    dependent_fields: Optional[Dict[str, float]]

    @root_validator
    def check_all_fields(cls, values):
        validate_structured_field(values, marqo_index=True)

        return values


class TensorField(ImmutableStrictBaseModel):
    """
    A tensor field that has a corresponding field.

    chunk_field_name and embeddings_field_name must be unique across all tensor fields.
    """
    name: str
    chunk_field_name: str
    embeddings_field_name: str


class HnswConfig(ImmutableStrictBaseModel):
    ef_construction: int = pydantic.Field(gt=0, alias='efConstruction')
    m: int = pydantic.Field(gt=0)


class TextPreProcessing(ImmutableStrictBaseModel):
    split_length: int = pydantic.Field(gt=0, alias='splitLength')
    split_overlap: int = pydantic.Field(ge=0, alias='splitOverlap')
    split_method: TextSplitMethod = pydantic.Field(alias='splitMethod')


class ImagePreProcessing(ImmutableStrictBaseModel):
    patch_method: Optional[PatchMethod] = pydantic.Field(alias='patchMethod')


class Model(StrictBaseModel):
    name: str
    properties: Optional[Dict[str, Any]]
    custom: bool = False

    def dict(self, *args, **kwargs):
        """
        Custom dict method that removes the properties field if the model is not custom. This ensures we don't store
        non-custom model properties when serializing and saving the index.
        """
        d = super().dict(*args, **kwargs)
        if not self.custom:
            d.pop('properties', None)
        return d

    def get_dimension(self) -> int:
        self._update_model_properties_from_registry()
        try:
            return self.properties['dimensions']
        except KeyError:
            raise InvalidArgumentError(
                "The given model properties does not contain a 'dimensions' key"
            )

    def get_properties(self) -> Dict[str, Any]:
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


class MarqoIndex(ImmutableStrictBaseModel, ABC):
    """
    Base class for a Marqo index.
    """
    name: str
    schema_name: str
    type: IndexType  # We need this so that we can deserialize the correct subclass
    model: Model
    normalize_embeddings: bool
    text_preprocessing: TextPreProcessing
    image_preprocessing: ImagePreProcessing
    distance_metric: DistanceMetric
    vector_numeric_type: VectorNumericType
    hnsw_config: HnswConfig
    marqo_version: str
    created_at: int = pydantic.Field(gt=0)
    updated_at: int = pydantic.Field(gt=0)
    _cache: Dict[str, Any] = PrivateAttr()

    class Config:
        allow_mutation = False

    def __init__(self, **data: Any):
        super().__init__(**data)

        self._cache = dict()

        # Retrieve all properties to populate cached properties
        for name, value in vars(MarqoIndex).items():
            if isinstance(value, property):
                getattr(self, name)

    @classmethod
    @abstractmethod
    def _valid_type(cls) -> IndexType:
        pass

    @validator('type')
    def validate_type(cls, type):
        if type not in [cls._valid_type(), cls._valid_type().value]:
            raise ValueError(f"Cannot assign a different type to {cls.__name__}")
        return type

    @validator('name')
    def validate_name(cls, name):
        validate_index_name(name)
        return name

    @classmethod
    def parse_obj(cls, obj: Any) -> Union['UnstructuredMarqoIndex', 'StructuredMarqoIndex']:
        obj = cls._enforce_dict_if_root(obj)
        if not isinstance(obj, dict):
            try:
                obj = dict(obj)
            except (TypeError, ValueError) as e:
                exc = TypeError(f'{cls.__name__} expected dict not {obj.__class__.__name__}')
                raise ValidationError([ErrorWrapper(exc, loc=ROOT_KEY)], cls) from e

        if 'type' in obj:
            if obj['type'] == IndexType.Structured.value:
                return StructuredMarqoIndex(**obj)
            elif obj['type'] == IndexType.Unstructured.value:
                return UnstructuredMarqoIndex(**obj)
            else:
                raise ValidationError(f"Invalid index type {obj['type']}")

        raise ValidationError(f"Index type not found in {obj}")

    def _cache_or_get(self, key: str, func):
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]


class UnstructuredMarqoIndex(MarqoIndex):
    type = IndexType.Unstructured
    treat_urls_and_pointers_as_images: bool
    short_string_length_threshold: int

    @classmethod
    def _valid_type(cls) -> IndexType:
        return IndexType.Unstructured


class StructuredMarqoIndex(MarqoIndex):
    type = IndexType.Structured
    fields: List[Field]  # all fields, including tensor fields
    tensor_fields: List[TensorField]

    def __init__(self, **data):
        super().__init__(**data)

    @root_validator
    def validate_model(cls, values):
        # Verify all combination fields are tensor fields
        combination_fields = [field for field in values.get('fields', []) if
                              field.type == FieldType.MultimodalCombination]
        tensor_field_names = {tensor_field.name for tensor_field in values.get('tensor_fields', [])}
        for field in combination_fields:
            if field.name not in tensor_field_names:
                raise ValueError(f'Field {field.name} has type {field.type.value()} and must be a tensor field')

        return values

    @classmethod
    def _valid_type(cls) -> IndexType:
        return IndexType.Structured

    @validator('tensor_fields')
    def validate_tensor_fields(cls, tensor_fields, values):
        field_names = {field.name for field in values.get('fields', [])}
        for tensor_field in tensor_fields:
            if tensor_field.name not in field_names:
                raise ValueError(f'Tensor field {tensor_field.name} is not a defined field. '
                                 f'Field names: {", ".join(field_names)}')
        return tensor_fields

    @property
    def lexical_field_map(self) -> Dict[str, Field]:
        return self._cache_or_get('lexical_field_map',
                                  lambda: {field.lexical_field_name: field for field in self.fields if
                                           FieldFeature.LexicalSearch in field.features}
                                  )

    @property
    def filter_field_map(self) -> Dict[str, Field]:
        return self._cache_or_get('filter_field_map',
                                  lambda: {field.filter_field_name: field for field in self.fields if
                                           FieldFeature.Filter in field.features}
                                  )

    @property
    def lexically_searchable_fields_names(self) -> Set[str]:
        return self._cache_or_get('lexically_searchable_fields_names',
                                  lambda: {field.name for field in self.fields if
                                           FieldFeature.LexicalSearch in field.features}
                                  )

    @property
    def filterable_fields_names(self) -> Set[str]:
        return self._cache_or_get('filterable_fields_names',
                                  lambda: {field.name for field in self.fields if
                                           FieldFeature.Filter in field.features}
                                  )

    @property
    def score_modifier_fields_names(self) -> Set[str]:
        return self._cache_or_get('score_modifier_fields_names',
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
        A map from field name, lexical name and filter name to the Field object.
        """
        return self._cache_or_get('all_field_map',
                                  lambda: {
                                      **{field.name: field for field in self.fields},
                                      **{field.lexical_field_name: field for field in self.fields if
                                         field.lexical_field_name is not None},
                                      **{field.filter_field_name: field for field in self.fields if
                                         field.filter_field_name is not None}
                                  }
                                  )

    @property
    def tensor_field_map(self) -> Dict[str, TensorField]:
        """
        A map from tensor field name to the TensorField object.
        """
        return self._cache_or_get('tensor_field_map',
                                  lambda: {tensor_field.name: tensor_field for tensor_field in self.tensor_fields}
                                  )

    @property
    def tensor_subfield_map(self) -> Dict[str, TensorField]:
        """
        A map from tensor chunk and embeddings field name to the TensorField object.
        """

        def generate():
            the_map = dict()
            for tensor_field in self.tensor_fields:
                if tensor_field.chunk_field_name in the_map:
                    raise ValueError(
                        f"Duplicate chunk field name {tensor_field.chunk_field_name} "
                        f"for tensor field {tensor_field.name}"
                    )
                the_map[tensor_field.chunk_field_name] = tensor_field

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


_PROTECTED_FIELD_NAMES = ['_id', '_tensor_facets', '_highlights', '_score', '_found']
_VESPA_NAME_PATTERN = r'[a-zA-Z_][a-zA-Z0-9_]*'
_INDEX_NAME_PATTERN = r'[a-zA-Z_-][a-zA-Z0-9_-]*'

_VESPA_NAME_REGEX = re.compile(_VESPA_NAME_PATTERN)
_INDEX_NAME_REGEX = re.compile(_INDEX_NAME_PATTERN)


def _is_valid_vespa_name(name: str) -> bool:
    """
    Validate a Vespa name.

    Returns:
        True if the name is valid, False otherwise
    """
    return _VESPA_NAME_REGEX.fullmatch(name) is not None


def validate_index_name(name: str) -> None:
    """
    Validate a MarqoIndex name. Raises ValueError if validation fails.
    """
    if _INDEX_NAME_REGEX.fullmatch(name) is None:
        raise ValueError(f'"{name}" is not a valid index name. Index name must match {_INDEX_NAME_PATTERN} '
                         f'and must not start with "{constants.MARQO_RESERVED_PREFIX}"')
    if name.startswith(constants.MARQO_RESERVED_PREFIX):
        raise ValueError(f'Index name must not start with "{constants.MARQO_RESERVED_PREFIX}"')


def validate_structured_field(values, marqo_index: bool) -> None:
    """
    Validate a Field or FieldRequest. Raises ValueError if validation fails.

    Args:
        marqo_index: Whether the validation is for a MarqoIndex Field (True) or a MarqoIndexRequest FieldRequest (False)
    """
    name: str = values['name']
    type: FieldType = values['type']
    features: List[FieldFeature] = values['features']
    dependent_fields: Optional[Dict[str, float]] = values['dependent_fields']

    if not _is_valid_vespa_name(name):
        raise ValueError(f'"{name}": Field name must match {_VESPA_NAME_PATTERN} '
                         f'and must not start with "{constants.MARQO_RESERVED_PREFIX}"')
    if name.startswith(constants.MARQO_RESERVED_PREFIX):
        raise ValueError(f'{name}: Field name must not start with "{constants.MARQO_RESERVED_PREFIX}"')
    if name in _PROTECTED_FIELD_NAMES:
        raise ValueError(f'{name}: Field name must not be one of {", ".join(_PROTECTED_FIELD_NAMES)}')

    if type in [FieldType.ImagePointer, FieldType.MultimodalCombination] and features:
        raise ValueError(f'{name}: Cannot specify features for field of type {type.value}')

    if type == FieldType.MultimodalCombination:
        if not dependent_fields:
            raise ValueError(f'{name}: dependent_fields must be defined for a field of type {type.value}')
    elif dependent_fields:
        raise ValueError(
            f'{name}: dependent_fields must only be defined for fields of type '
            f'{FieldType.MultimodalCombination.value}'
        )

    if FieldFeature.LexicalSearch in features and type not in [FieldType.Text, FieldType.ArrayText]:
        raise ValueError(
            f'{name}: Field with {FieldFeature.LexicalSearch.value} feature must be of type '
            f'{FieldType.Text.value} or {FieldType.ArrayText.value}'
        )

    # These validations are specific to marqo_index.Field
    if marqo_index:
        lexical_field_name: Optional[str] = values['lexical_field_name']
        filter_field_name: Optional[str] = values['filter_field_name']

        if FieldFeature.LexicalSearch in features and not lexical_field_name:
            raise ValueError(
                f'{name}: lexical_field_name must be populated when {FieldFeature.LexicalSearch.value} '
                f'feature is present'
            )

        if FieldFeature.Filter in features and not filter_field_name:
            # We can filter anything other than ImagePointer and MultimodalCombination, which don't allow features
            # so no type validation here
            raise ValueError(
                f'{name}: filter_field_name must be populated when {FieldFeature.Filter.value} '
                f'feature is present'
            )

        if FieldFeature.ScoreModifier in features and type not in [FieldType.Float, FieldType.Int]:
            raise ValueError(
                f'{name}: Field with {FieldFeature.ScoreModifier.value} feature must be of type '
                f'{FieldType.Float.value} or {FieldType.Int.value}'
            )

        if lexical_field_name and FieldFeature.LexicalSearch not in features:
            raise ValueError(
                f'{name}: lexical_field_name must only be populated when '
                f'{FieldFeature.LexicalSearch.value} feature is present'
            )

        if filter_field_name and FieldFeature.Filter not in features:
            raise ValueError(
                f'{name}: filter_field_name must only be populated when {FieldFeature.Filter.value} '
                f'feature is present'
            )
