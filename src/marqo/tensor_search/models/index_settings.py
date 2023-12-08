import time
from typing import Dict, Any, Optional, List

import marqo.core.models.marqo_index as core
import marqo.api.exceptions as api_exceptions
from marqo import version
from marqo.base_model import StrictBaseModel
from marqo.core.models.marqo_index_request import FieldRequest, MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest


class AnnParameters(StrictBaseModel):
    spaceType: core.DistanceMetric
    parameters: core.HnswConfig


class IndexSettings(StrictBaseModel):
    type: core.IndexType = core.IndexType.Unstructured
    allFields: Optional[List[FieldRequest]]
    tensorFields: Optional[List[str]]
    treatUrlsAndPointersAsImages: Optional[bool]
    model: str = 'hf/all_datasets_v4_MiniLM-L6'
    modelProperties: Optional[Dict[str, Any]]
    normalizeEmbeddings: bool = True
    textPreprocessing: core.TextPreProcessing = core.TextPreProcessing(
        splitLength=2,
        splitOverlap=0,
        splitMethod=core.TextSplitMethod.Sentence
    )
    imagePreprocessing: core.ImagePreProcessing = core.ImagePreProcessing(
        patchMethod=None
    )
    vectorNumericType: core.VectorNumericType = core.VectorNumericType.Float
    annParameters: AnnParameters = AnnParameters(
        spaceType=core.DistanceMetric.Angular,
        parameters=core.HnswConfig(
            efConstruction=128,
            m=16
        )
    )

    def to_marqo_index_request(self, index_name: str) -> MarqoIndexRequest:
        marqo_fields = None
        if self.type == core.IndexType.Structured:
            if self.treatUrlsAndPointersAsImages is not None:
                raise api_exceptions.InvalidArgError(
                    "treat_urls_and_pointers_as_images is not a valid parameter for structured indexes"
                )

            if self.allFields is not None:
                marqo_fields = [
                    FieldRequest(
                        name=field.name,
                        type=field.type,
                        features=field.features,
                        dependent_fields=field.dependent_fields
                    ) for field in self.allFields
                ]

            return StructuredMarqoIndexRequest(
                name=index_name,
                model=core.Model(
                    name=self.model,
                    properties=self.modelProperties,
                    custom=self.modelProperties is not None
                ),
                normalize_embeddings=self.normalizeEmbeddings,
                text_preprocessing=self.textPreprocessing,
                image_preprocessing=self.imagePreprocessing,
                distance_metric=self.annParameters.spaceType,
                vector_numeric_type=self.vectorNumericType,
                hnsw_config=self.annParameters.parameters,
                fields=marqo_fields,
                tensor_fields=self.tensorFields,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        elif self.type == core.IndexType.Unstructured:
            if self.allFields is not None:
                raise api_exceptions.InvalidArgError(
                    "all_fields is not a valid parameter for unstructured indexes"
                )
            if self.tensorFields is not None:
                raise api_exceptions.InvalidArgError(
                    "tensor_fields is not a valid parameter for unstructured indexes"
                )

            if self.treatUrlsAndPointersAsImages is None:
                # Default value for treat_urls_and_pointers_as_images is False, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                self.treatUrlsAndPointersAsImages = False

            return UnstructuredMarqoIndexRequest(
                name=index_name,
                model=core.Model(
                    name=self.model,
                    properties=self.modelProperties,
                    custom=self.modelProperties is not None
                ),
                normalize_embeddings=self.normalizeEmbeddings,
                text_preprocessing=self.textPreprocessing,
                image_preprocessing=self.imagePreprocessing,
                distance_metric=self.annParameters.spaceType,
                vector_numeric_type=self.vectorNumericType,
                hnsw_config=self.annParameters.parameters,
                treat_urls_and_pointers_as_images=self.treatUrlsAndPointersAsImages,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        else:
            raise api_exceptions.InternalError(f"Unknown index type: {self.type}")

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
            raise api_exceptions.InternalError(f"Unknown index type: {type(marqo_index)}")
