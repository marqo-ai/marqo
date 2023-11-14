import time
from typing import Dict, Any, Optional, List

import marqo.core.models.marqo_index as core
import marqo.errors as errors
from marqo import version
from marqo.core.models.marqo_index_request import FieldRequest, MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest
from marqo.tensor_search.models.api_models import BaseMarqoModel


class AnnParameters(BaseMarqoModel):
    space_type: core.DistanceMetric
    parameters: core.HnswConfig


class IndexSettings(BaseMarqoModel):
    type: core.IndexType = core.IndexType.Unstructured
    all_fields: Optional[List[FieldRequest]]
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

    def to_marqo_index_request(self, index_name: str) -> MarqoIndexRequest:
        marqo_fields = None
        if self.type == core.IndexType.Structured:
            if self.treat_urls_and_pointers_as_images is not None:
                raise errors.InvalidArgError(
                    "treat_urls_and_pointers_as_images is not a valid parameter for structured indexes"
                )

            if self.all_fields is not None:
                marqo_fields = [
                    FieldRequest(
                        name=field.name,
                        type=field.type,
                        features=field.features,
                        dependent_fields=field.dependent_fields
                    ) for field in self.all_fields
                ]

            return StructuredMarqoIndexRequest(
                name=index_name,
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
                tensor_fields=self.tensor_fields,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        elif self.type == core.IndexType.Unstructured:
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

            return UnstructuredMarqoIndexRequest(
                name=index_name,
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
            raise errors.InternalError(f"Unknown index type: {self.type}")

    @classmethod
    def from_marqo_index(cls, marqo_index: core.MarqoIndex) -> "IndexSettings":
        if isinstance(marqo_index, core.StructuredMarqoIndex):
            return cls(
                type=marqo_index.type,
                all_fields=[
                    FieldRequest(
                        name=field.name,
                        type=field.type,
                        features=field.features,
                        dependent_fields=field.dependent_fields
                    ) for field in marqo_index.fields
                ],
                tensor_fields=[field.name for field in marqo_index.tensor_fields],
                model=marqo_index.model.name,
                model_properties=marqo_index.model.properties,
                normalize_embeddings=marqo_index.normalize_embeddings,
                text_preprocessing=marqo_index.text_preprocessing,
                image_preprocessing=marqo_index.image_preprocessing,
                vector_numeric_type=marqo_index.vector_numeric_type,
                ann_parameters=AnnParameters(
                    space_type=marqo_index.distance_metric,
                    parameters=marqo_index.hnsw_config
                )
            )
        elif isinstance(marqo_index, core.UnstructuredMarqoIndex):
            return cls(
                type=marqo_index.type,
                treat_urls_and_pointers_as_images=marqo_index.treat_urls_and_pointers_as_images,
                model=marqo_index.model.name,
                model_properties=marqo_index.model.properties,
                normalize_embeddings=marqo_index.normalize_embeddings,
                text_preprocessing=marqo_index.text_preprocessing,
                image_preprocessing=marqo_index.image_preprocessing,
                vector_numeric_type=marqo_index.vector_numeric_type,
                ann_parameters=AnnParameters(
                    space_type=marqo_index.distance_metric,
                    parameters=marqo_index.hnsw_config
                )
            )
        else:
            raise errors.InternalError(f"Unknown index type: {type(marqo_index)}")

