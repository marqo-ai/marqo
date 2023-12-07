from abc import ABC
from typing import List, Dict, Optional

import pydantic
from pydantic import root_validator, validator

import marqo.core.models.marqo_index as marqo_index
from marqo.base_model import StrictBaseModel, ImmutableStrictBaseModel


class MarqoIndexRequest(ImmutableStrictBaseModel, ABC):
    """
    Base class for a Marqo index request. Instances of this class represent requests to create or modify a Marqo index,
    while a Marqo index itself is represented by an instance of the MarqoIndex class.

    The validation source of truth for a Marqo index is the MarqoIndex class and its subclasses. However, some of this
    validation is repeated here so that user input errors (caught here) can be distinguished from internal
    errors (caught in MarqoIndex).
    """
    name: str
    model: marqo_index.Model
    normalize_embeddings: bool
    text_preprocessing: marqo_index.TextPreProcessing
    image_preprocessing: marqo_index.ImagePreProcessing
    distance_metric: marqo_index.DistanceMetric
    vector_numeric_type: marqo_index.VectorNumericType
    hnsw_config: marqo_index.HnswConfig
    marqo_version: str
    created_at: int
    updated_at: int

    @validator('name')
    def validate_name(cls, name):
        marqo_index.validate_index_name(name)
        return name


class UnstructuredMarqoIndexRequest(MarqoIndexRequest):
    treat_urls_and_pointers_as_images: bool


class FieldRequest(StrictBaseModel):
    name: str
    type: marqo_index.FieldType
    features: List[marqo_index.FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]] = pydantic.Field(alias='dependentFields')

    @root_validator
    def check_all_fields(cls, values):
        marqo_index.validate_structured_field(values, marqo_index=False)

        return values


class StructuredMarqoIndexRequest(MarqoIndexRequest):
    fields: List[FieldRequest]  # all fields, including tensor fields
    tensor_fields: List[str]

    @root_validator
    def validate_model(cls, values):
        # Verify all tensor fields are valid fields
        field_names = {field.name for field in values.get('fields', [])}
        tensor_field_names = values.get('tensor_fields', [])
        for tensor_field in tensor_field_names:
            if tensor_field not in field_names:
                raise ValueError(f"Tensor field {tensor_field} is not a defined field. "
                                 f'Field names: {", ".join(field_names)}')

        # Verify all combination fields are tensor fields
        combination_fields = [field for field in values.get('fields', []) if
                              field.type == marqo_index.FieldType.MultimodalCombination]
        for field in combination_fields:
            if field.name not in tensor_field_names:
                raise ValueError(f'Field {field.name} has type {field.type.value} and must be a tensor field')

        return values
