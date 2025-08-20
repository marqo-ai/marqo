import time
from typing import Dict, Any, Optional, List, Union

from pydantic.v1 import root_validator

import marqo.api.exceptions as api_exceptions
import marqo.core.models.marqo_index as core
from marqo import version, marqo_docs
from marqo.base_model import StrictBaseModel
from marqo.core.models.marqo_index_request import FieldRequest, MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest, ObjectArrayFieldRequest


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
    collapseFields: Optional[List[core.CollapseField]] = None
    objectArrayFields: Optional[List[ObjectArrayFieldRequest]] = None
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
    videoPreprocessing: Optional[core.VideoPreProcessing] = core.VideoPreProcessing(
        splitLength=20,
        splitOverlap=3,
    )
    audioPreprocessing: Optional[core.AudioPreProcessing] = core.AudioPreProcessing(
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

    @root_validator
    def validate_collapse_fields(cls, values):
        collapse_fields = values.get('collapseFields')
        index_type = values.get('type')
        
        # collapseFields is only supported for SemiStructuredIndex
        if collapse_fields is not None and index_type == core.IndexType.Structured:
            raise api_exceptions.InvalidArgError(
                "collapseFields is only supported for unstructured indexes"
            )
        
        return values

    @root_validator
    def validate_object_array_fields(cls, values):
        object_array_fields = values.get('objectArrayFields')
        index_type = values.get('type')
        
        # objectArrayFields is only supported for SemiStructuredIndex
        if object_array_fields is not None and index_type == core.IndexType.Structured:
            raise api_exceptions.InvalidArgError(
                "objectArrayFields is only supported for unstructured indexes"
            )
        
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
                collapse_fields=self.collapseFields,
                object_array_fields=self._convert_object_array_fields_to_core(),
                marqo_version=version.get_version(),
                created_at=time.time(),
                updated_at=time.time()
            )
        else:
            raise api_exceptions.InternalError(f"Unknown index type: {self.type}")

    def _convert_object_array_fields_to_core(self):
        """Convert ObjectArrayFieldRequest objects to core ObjectArrayField objects"""
        if self.objectArrayFields is None:
            return None
        
        core_object_array_fields = []
        for obj_array_req in self.objectArrayFields:
            # Convert field definitions
            core_field_defs = []
            for field_def_req in obj_array_req.fields:
                core_field_defs.append(
                    core.ObjectArrayFieldDefinition(
                        name=field_def_req.name,
                        type=field_def_req.type
                    )
                )
            
            # Create core ObjectArrayField
            core_obj_array = core.ObjectArrayField(
                name=obj_array_req.name,
                object_array_field_name=obj_array_req.object_array_field_name,
                fields=core_field_defs
            )
            core_object_array_fields.append(core_obj_array)
        
        return core_object_array_fields

    @classmethod
    def _convert_core_object_array_fields_to_request(cls, core_object_array_fields):
        """Convert core ObjectArrayField objects to ObjectArrayFieldRequest objects"""
        if core_object_array_fields is None:
            return None
        
        request_object_array_fields = []
        for core_obj_array in core_object_array_fields:
            # Convert field definitions
            request_field_defs = []
            for core_field_def in core_obj_array.fields:
                from marqo.core.models.marqo_index_request import ObjectArrayFieldDefinitionRequest
                request_field_defs.append(
                    ObjectArrayFieldDefinitionRequest(
                        name=core_field_def.name,
                        type=core_field_def.type
                    )
                )
            
            # Create request ObjectArrayField
            request_obj_array = ObjectArrayFieldRequest(
                name=core_obj_array.name,
                object_array_field_name=core_obj_array.object_array_field_name,
                fields=request_field_defs
            )
            request_object_array_fields.append(request_obj_array)
        
        return request_object_array_fields

    @classmethod
    def from_marqo_index(cls, marqo_index: core.MarqoIndex) -> "IndexSettings":
        if isinstance(marqo_index, core.UnstructuredMarqoIndex):
            # This covers both UnstructuredMarqoIndex and SemiStructuredMarqoIndex
            # We intentionally hide the lexical and tensor fields info in SemiStructuredMarqoIndex from customers since
            # this information and the SemiStructured concept are internal implementation details only.
            
            # Only include collapseFields and objectArrayFields for SemiStructuredMarqoIndex
            collapse_fields = None
            object_array_fields = None
            if isinstance(marqo_index, core.SemiStructuredMarqoIndex):
                collapse_fields = marqo_index.collapse_fields
                object_array_fields = cls._convert_core_object_array_fields_to_request(marqo_index.object_array_fields)
            
            return cls(
                type=core.IndexType.Unstructured,
                treatUrlsAndPointersAsImages=marqo_index.treat_urls_and_pointers_as_images,
                treatUrlsAndPointersAsMedia=marqo_index.treat_urls_and_pointers_as_media,
                filterStringMaxLength=marqo_index.filter_string_max_length,
                collapseFields=collapse_fields,
                objectArrayFields=object_array_fields,
                model=marqo_index.model.name,
                modelProperties=IndexSettings.get_model_properties(marqo_index),
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
                modelProperties=IndexSettings.get_model_properties(marqo_index),
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

    @classmethod
    def get_model_properties(cls, marqo_index):
        if marqo_index.model.properties is None:
            return None

        if marqo_index.model.properties.get('isMarqtuneModel', False):
            # Hide all properties except for isMarqtuneModel
            marqo_index.model.properties.pop('name', None)
            marqo_index.model.properties.pop('dimensions')
            marqo_index.model.properties.pop('model_location')
            marqo_index.model.properties.pop('type')
            marqo_index.model.properties.pop('trustRemoteCode', None)
        return marqo_index.model.properties


class IndexSettingsWithName(IndexSettings):
    indexName: str
