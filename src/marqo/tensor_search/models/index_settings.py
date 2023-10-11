from typing import Dict, Any, Optional, List

import marqo.core.models.marqo_index as core
from marqo.tensor_search.models.api_models import BaseMarqoModel


class Field(BaseMarqoModel):
    name: str
    type: core.FieldType
    features: List[core.FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]


class IndexSettings(BaseMarqoModel):
    type: core.IndexType
    fields: Optional[List[Field]]
    model: str
    model_properties: Optional[Dict[str, Any]]
    vector_numeric_type: core.VectorNumericType
    distance_metric: core.DistanceMetric
    ann_parameters: core.HnswConfig
    tensor_fields: Optional[List[str]]

    def to_marqo_index(self, index_name: str):
        marqo_fields = None
        if self.fields is not None:
            marqo_fields = [
                core.Field(
                    name=field.name,
                    type=field.type,
                    features=field.features,
                    dependent_fields=field.dependent_fields
                ) for field in self.fields
            ]

        marqo_tensor_fields = None
        if self.tensor_fields is not None:
            marqo_tensor_fields = [
                core.TensorField(
                    name=field.name,
                ) for field in self.tensor_fields
            ]

        return core.MarqoIndex(
            name=index_name,
            type=self.type,
            model=core.Model(
                name=self.model,
                properties=self.model_properties
            ),
            distance_metric=self.distance_metric,
            vector_numeric_type=self.vector_numeric_type,
            hnsw_config=self.ann_parameters,
            fields=marqo_fields,
            tensor_fields=marqo_tensor_fields
        )
