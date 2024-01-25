import time
from typing import Dict, Any, Optional, List, Union

from pydantic import root_validator

import marqo.api.exceptions as api_exceptions
import marqo.core.models.marqo_index as core
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
    filterStringMaxLength: Optional[int]
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
        spaceType=core.DistanceMetric.PrenormalizedAnguar,
        parameters=core.HnswConfig(
            efConstruction=512,
            m=16
        )
    )

    @root_validator(pre=True)
    def validate_field_names(cls, values):
        # Verify no snake case field names (pydantic won't catch these due to allow_population_by_field_name = True)
        def validate_keys(d: Union[dict, list]):
            if isinstance(d, dict):
                for key in d.keys():
                    if '_' in key:
                        raise ValueError(f"Invalid field name '{key}'. "
                                         f"See Create Index API reference here https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/")

                    if key not in ['dependentFields', 'modelProperties']:
                        validate_keys(d[key])
            elif isinstance(d, list):
                for item in d:
                    validate_keys(item)

        validate_keys(values)

        return values

    def to_marqo_index_request(self, index_name: str) -> MarqoIndexRequest:
        marqo_fields = None
        if self.type == core.IndexType.Structured:
            if self.treatUrlsAndPointersAsImages is not None:
                raise api_exceptions.InvalidArgError(
                    "treat_urls_and_pointers_as_images is not a valid parameter for structured indexes"
                )
            if self.filterStringMaxLength is not None:
                raise api_exceptions.InvalidArgError(
                    "filterStringMaxLength is not a valid parameter for structured indexes"
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

            if self.filterStringMaxLength is None:
                # Default value for filter_string_max_length is 20, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                self.filterStringMaxLength = 20

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
                filter_string_max_length=self.filterStringMaxLength,
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
                allFields=[
                    FieldRequest(
                        name=field.name,
                        type=field.type,
                        features=field.features,
                        dependent_fields=field.dependent_fields
                    ) for field in marqo_index.fields
                ],
                tensorFields=[field.name for field in marqo_index.tensor_fields],
                model=marqo_index.model.name,
                modelProperties=marqo_index.model.properties,
                normalizeEmbeddings=marqo_index.normalize_embeddings,
                textPreprocessing=marqo_index.text_preprocessing,
                imagePreprocessing=marqo_index.image_preprocessing,
                vectorNumericType=marqo_index.vector_numeric_type,
                annParameters=AnnParameters(
                    spaceType=marqo_index.distance_metric,
                    parameters=marqo_index.hnsw_config
                )
            )
        elif isinstance(marqo_index, core.UnstructuredMarqoIndex):
            return cls(
                type=marqo_index.type,
                treatUrlsAndPointersAsImages=marqo_index.treat_urls_and_pointers_as_images,
                filterStringMaxLength=marqo_index.filter_string_max_length,
                model=marqo_index.model.name,
                modelProperties=marqo_index.model.properties,
                normalizeEmbeddings=marqo_index.normalize_embeddings,
                textPreprocessing=marqo_index.text_preprocessing,
                imagePreprocessing=marqo_index.image_preprocessing,
                vectorNumericType=marqo_index.vector_numeric_type,
                annParameters=AnnParameters(
                    spaceType=marqo_index.distance_metric,
                    parameters=marqo_index.hnsw_config
                )
            )
        else:
            raise api_exceptions.InternalError(f"Unknown index type: {type(marqo_index)}")


class IndexSettingsWithName(IndexSettings):
    indexName: str
