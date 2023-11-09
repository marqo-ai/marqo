import time
from typing import Dict, Any, Optional, List

import marqo.core.models.marqo_index as core
import marqo.errors as errors
from marqo import version
from marqo.tensor_search.models.api_models import BaseMarqoModel


class Field(BaseMarqoModel):
    name: str
    type: core.FieldType
    features: List[core.FieldFeature] = []
    dependent_fields: Optional[Dict[str, float]]


class AnnParameters(BaseMarqoModel):
    space_type: core.DistanceMetric
    parameters: core.HnswConfig


class IndexSettings(BaseMarqoModel):
    type: core.IndexType = core.IndexType.Unstructured
    all_fields: Optional[List[Field]]
    tensor_fields: Optional[List[str]]
    treat_urls_and_pointers_as_images: Optional[bool]
    model: str = 'hf/all_datasets_v4_MiniLM-L6'
    model_properties: Optional[Dict[str, Any]]
    normalize_embeddings: bool = True
    text_preprocessing: core.TextPreProcessing = core.TextPreProcessing(
        split_length=2,
        split_overlap=0,
        split_method=core.TextSplitMethod.Sentence
    )
    image_preprocessing: core.ImagePreProcessing = core.ImagePreProcessing(
        patch_method=None
    )
    vector_numeric_type: core.VectorNumericType = core.VectorNumericType.Float
    ann_parameters: AnnParameters = AnnParameters(
        space_type=core.DistanceMetric.Angular,
        parameters=core.HnswConfig(
            ef_construction=128,
            m=16
        )
    )

    def to_marqo_index(self, index_name: str) -> core.MarqoIndex:
        marqo_fields = None
        if type == core.IndexType.Structured:
            if self.treat_urls_and_pointers_as_images is not None:
                raise errors.InvalidArgError(
                    "treat_urls_and_pointers_as_images is not a valid parameter for structured indexes"
                )

            if self.all_fields is not None:
                marqo_fields = [
                    core.Field(
                        name=field.name,
                        type=field.type,
                        features=field.features,
                        dependent_fields=field.dependent_fields
                    ) for field in self.all_fields
                ]

            marqo_tensor_fields = None
            if self.tensor_fields is not None:
                marqo_tensor_fields = [
                    core.TensorField(
                        name=field,
                    ) for field in self.tensor_fields
                ]

            return core.StructuredMarqoIndex(
                name=index_name,
                type=self.type,
                model=core.Model(
                    name=self.model,
                    properties=self.model_properties,
                    custom=self.model_properties is not None
                ),
                normalize_embeddings=self.normalize_embeddings,
                text_preprocessing=self.text_preprocessing,
                image_preprocessing=self.image_preprocessing,
                distance_metric=self.ann_parameters.space_type,
                vector_numeric_type=self.vector_numeric_type,
                hnsw_config=self.ann_parameters.parameters,
                fields=marqo_fields,
                tensor_fields=marqo_tensor_fields,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        elif type == core.IndexType.Unstructured:
            if self.all_fields is not None:
                raise errors.InvalidArgError(
                    "all_fields is not a valid parameter for unstructured indexes"
                )
            if self.tensor_fields is not None:
                raise errors.InvalidArgError(
                    "tensor_fields is not a valid parameter for unstructured indexes"
                )

            if self.treat_urls_and_pointers_as_images is None:
                # Default value for treat_urls_and_pointers_as_images is False, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                self.treat_urls_and_pointers_as_images = False

            return core.UnstructuredMarqoIndex(
                name=index_name,
                type=self.type,
                model=core.Model(
                    name=self.model,
                    properties=self.model_properties,
                    custom=self.model_properties is not None
                ),
                normalize_embeddings=self.normalize_embeddings,
                text_preprocessing=self.text_preprocessing,
                image_preprocessing=self.image_preprocessing,
                distance_metric=self.ann_parameters.space_type,
                vector_numeric_type=self.vector_numeric_type,
                hnsw_config=self.ann_parameters.parameters,
                treat_urls_and_pointers_as_images=self.treat_urls_and_pointers_as_images,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        else:
            raise errors.InternalError(f"Unknown index type: {type}")
