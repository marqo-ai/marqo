from abc import ABC
from typing import List, Dict, Optional

from pydantic import root_validator, validator

import marqo.core.models.marqo_index as marqo_index
from marqo.core.models.strict_base_model import StrictBaseModel, ImmutableStrictBaseModel


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


class UnstructuredMarqoIndexRequest(MarqoIndexRequest):
    treat_urls_and_pointers_as_images: bool


class FieldRequest(StrictBaseModel):
    name: str
    type: marqo_index.FieldType
    features: List[marqo_index.FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]

    @root_validator
    def check_all_fields(cls, values):
        marqo_index.validate_field(values, marqo_index=False)

        return values


class StructuredMarqoIndexRequest(MarqoIndexRequest):
    fields: List[FieldRequest]  # all fields, including tensor fields
    tensor_fields: List[str]

    @validator('tensor_fields')
    def validate_tensor_fields(cls, tensor_fields, values):
        field_names = {field.name for field in values.get('fields', [])}
        for tensor_field in tensor_fields:
            if tensor_field not in field_names:
                raise ValueError(f'Tensor field {tensor_field.name} is not a defined field. '
                                 f'Field names: {", ".join(field_names)}')
        return tensor_fields
