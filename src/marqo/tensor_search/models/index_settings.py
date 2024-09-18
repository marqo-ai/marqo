import time
from typing import Dict, Any, Optional, List, Union

from pydantic import root_validator

import marqo.api.exceptions as api_exceptions
import marqo.core.models.marqo_index as core
from marqo import version, marqo_docs
from marqo.base_model import StrictBaseModel
from marqo.core.models.marqo_index_request import FieldRequest, MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest


class AnnParameters(StrictBaseModel):
    spaceType: core.DistanceMetric
    parameters: core.HnswConfig


class IndexSettings(StrictBaseModel):
    type: core.IndexType = core.IndexType.SemiStructured
    allFields: Optional[List[FieldRequest]]
    tensorFields: Optional[List[str]]
    treatUrlsAndPointersAsImages: Optional[bool]
    treatUrlsAndPointersAsMedia: Optional[bool]
    filterStringMaxLength: Optional[int]
    model: str = 'hf/e5-base-v2'
    modelProperties: Optional[Dict[str, Any]]
    textQueryPrefix: Optional[str] = None
    textChunkPrefix: Optional[str] = None
    normalizeEmbeddings: bool = True
    textPreprocessing: core.TextPreProcessing = core.TextPreProcessing(
        splitLength=2,
        splitOverlap=0,
        splitMethod=core.TextSplitMethod.Sentence
    )
    imagePreprocessing: core.ImagePreProcessing = core.ImagePreProcessing(
        patchMethod=None
    )
    videoPreprocessing: core.VideoPreProcessing = core.VideoPreProcessing(
        splitLength=20,
        splitOverlap=3,
    )
    audioPreprocessing: core.AudioPreProcessing = core.AudioPreProcessing(
        splitLength=10,
        splitOverlap=3,
    )
    vectorNumericType: core.VectorNumericType = core.VectorNumericType.Float
    annParameters: AnnParameters = AnnParameters(
        spaceType=core.DistanceMetric.PrenormalizedAngular,
        parameters=core.HnswConfig(
            efConstruction=512,
            m=16
        )
    )
    
    @root_validator
    def validate_url_pointer_treatment(cls, values):
        treat_as_images = values.get('treatUrlsAndPointersAsImages')
        treat_as_media = values.get('treatUrlsAndPointersAsMedia')

        if treat_as_images and not treat_as_media:
            # Deprecation warning
            import warnings
            warnings.warn("'treatUrlsAndPointersAsImages' is deprecated. Use 'treatUrlsAndPointersAsMedia' instead.", DeprecationWarning)

        if treat_as_images == False and treat_as_media:
            raise api_exceptions.InvalidArgError(
                "Invalid combination: 'treatUrlsAndPointersAsImages' cannot be False when 'treatUrlsAndPointersAsMedia' is True."
            )

        # If treatUrlsAndPointersAsMedia is True, ensure treatUrlsAndPointersAsImages is also True
        if treat_as_media:
            values['treatUrlsAndPointersAsImages'] = True

        return values

    @root_validator(pre=True)
    def validate_field_names(cls, values):
        # Verify no snake case field names (pydantic won't catch these due to allow_population_by_field_name = True)
        def validate_keys(d: Union[dict, list]):
            if isinstance(d, dict):
                for key in d.keys():
                    if '_' in key:
                        raise ValueError(f"Invalid field name '{key}'. "
                                         f"See Create Index API reference here {marqo_docs.create_index()}")

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
                    "treatUrlsAndPointersAsImages is not a valid parameter for structured indexes"
                )
            if self.treatUrlsAndPointersAsMedia is not None:
                raise api_exceptions.InvalidArgError(
                    "treatUrlsAndPointersAsMedia is not a valid parameter for structured indexes"
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
                    custom=self.modelProperties is not None,
                    text_query_prefix=self.textQueryPrefix,
                    text_chunk_prefix=self.textChunkPrefix
                ),
                normalize_embeddings=self.normalizeEmbeddings,
                text_preprocessing=self.textPreprocessing,
                image_preprocessing=self.imagePreprocessing,
                video_preprocessing=self.videoPreprocessing,
                audio_preprocessing=self.audioPreprocessing,
                distance_metric=self.annParameters.spaceType,
                vector_numeric_type=self.vectorNumericType,
                hnsw_config=self.annParameters.parameters,
                fields=marqo_fields,
                tensor_fields=self.tensorFields,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time(),
            )
        elif self.type in [core.IndexType.Unstructured, core.IndexType.SemiStructured]:
            if self.allFields is not None:
                raise api_exceptions.InvalidArgError(
                    "allFields is not a valid parameter for unstructured indexes"
                )
            if self.tensorFields is not None:
                raise api_exceptions.InvalidArgError(
                    "tensorFields is not a valid parameter for unstructured indexes"
                )

            if self.treatUrlsAndPointersAsImages is None:
                # Default value for treat_urls_and_pointers_as_images is False, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                if self.treatUrlsAndPointersAsMedia is True:
                    self.treatUrlsAndPointersAsImages = True
                else:
                    self.treatUrlsAndPointersAsImages = False
            
            if self.treatUrlsAndPointersAsMedia is None:
                # Default value for treat_urls_and_pointers_as_media is False, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                self.treatUrlsAndPointersAsMedia = False

            if self.filterStringMaxLength is None:
                # Default value for filter_string_max_length is 20, but we can't set it in the model
                # as it is not a valid parameter for structured indexes
                self.filterStringMaxLength = 50
    
            return UnstructuredMarqoIndexRequest(
                name=index_name,
                model=core.Model(
                    name=self.model,
                    properties=self.modelProperties,
                    custom=self.modelProperties is not None,
                    text_query_prefix=self.textQueryPrefix,
                    text_chunk_prefix=self.textChunkPrefix
                ),
                normalize_embeddings=self.normalizeEmbeddings,
                text_preprocessing=self.textPreprocessing,
                image_preprocessing=self.imagePreprocessing,
                video_preprocessing=self.videoPreprocessing,
                audio_preprocessing=self.audioPreprocessing,
                distance_metric=self.annParameters.spaceType,
                vector_numeric_type=self.vectorNumericType,
                hnsw_config=self.annParameters.parameters,
                treat_urls_and_pointers_as_images=self.treatUrlsAndPointersAsImages,
                treat_urls_and_pointers_as_media=self.treatUrlsAndPointersAsMedia,
                filter_string_max_length=self.filterStringMaxLength,
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        else:
            raise api_exceptions.InternalError(f"Unknown index type: {self.type}")

    @classmethod
    def from_marqo_index(cls, marqo_index: core.MarqoIndex) -> "IndexSettings":
        if isinstance(marqo_index, core.UnstructuredMarqoIndex):
            # This covers both UnstructuredMarqoIndex and SemiStructuredMarqoIndex
            # We intentionally hide the lexical and tensor fields info in SemiStructuredMarqoIndex from customers since
            # this information and the SemiStructured concept are internal implementation details only.
            return cls(
                type=core.IndexType.Unstructured,
                treatUrlsAndPointersAsImages=marqo_index.treat_urls_and_pointers_as_images,
                treatUrlsAndPointersAsMedia=marqo_index.treat_urls_and_pointers_as_media,
                filterStringMaxLength=marqo_index.filter_string_max_length,
                model=marqo_index.model.name,
                modelProperties=marqo_index.model.properties,
                normalizeEmbeddings=marqo_index.normalize_embeddings,
                textPreprocessing=marqo_index.text_preprocessing,
                imagePreprocessing=marqo_index.image_preprocessing,
                videoPreprocessing=marqo_index.video_preprocessing,
                audioPreprocessing=marqo_index.audio_preprocessing,
                vectorNumericType=marqo_index.vector_numeric_type,
                annParameters=AnnParameters(
                    spaceType=marqo_index.distance_metric,
                    parameters=marqo_index.hnsw_config
                )
            )
        elif isinstance(marqo_index, core.StructuredMarqoIndex):
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
                videoPreprocessing=marqo_index.video_preprocessing,
                audioPreprocessing=marqo_index.audio_preprocessing,
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
