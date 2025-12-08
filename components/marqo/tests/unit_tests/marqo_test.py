import uuid
from typing import List, Optional
from unittest import TestCase
from unittest.mock import patch, Mock
import time

from marqo import version
from marqo.core.models.marqo_index import Field, TensorField, Model, TextPreProcessing, TextSplitMethod, \
    ImagePreProcessing, VideoPreProcessing, AudioPreProcessing, DistanceMetric, VectorNumericType, HnswConfig, \
    StructuredMarqoIndex, CollapseField
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, \
    FieldType, FieldFeature, StringArrayField
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.version import get_version


class MarqoTestCase(TestCase):
    @classmethod
    def configure_request_metrics(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        """
        cls.mock_request = Mock()
        cls.patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

    @classmethod
    def setUpClass(cls) -> None:
        cls.configure_request_metrics()

    @classmethod
    def structured_marqo_index(
            cls,
            name: str,
            schema_name: str,
            fields: List[Field] = (),
            tensor_fields: List[TensorField] = (),
            model: Model = Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings: bool = True,
            text_preprocessing: TextPreProcessing = TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing: ImagePreProcessing = ImagePreProcessing(
                patch_method=None
            ),
            video_preprocessing: VideoPreProcessing = VideoPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            audio_preprocessing: AudioPreProcessing = AudioPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            marqo_version=get_version(),
            schema_template_version=get_version(),
            created_at=int(time.time()),
            updated_at=int(time.time()),
            version=None
    ) -> StructuredMarqoIndex:
        """
        Helper method that provides reasonable defaults for StructuredMarqoIndex.
        """
        return StructuredMarqoIndex(
            name=name,
            schema_name=schema_name,
            model=model,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            video_preprocessing=video_preprocessing,
            audio_preprocessing=audio_preprocessing,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            fields=fields,
            tensor_fields=tensor_fields,
            marqo_version=marqo_version,
            schema_template_version=schema_template_version,
            created_at=created_at,
            updated_at=updated_at,
            version=version
        )

    @classmethod
    def semi_structured_marqo_index(
            cls,
            name: str,
            schema_name: Optional[str] = None,
            typeahead_schema_name: Optional[str] = None,
            model: Model = Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings: bool = True,
            text_preprocessing: TextPreProcessing = TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing: ImagePreProcessing = ImagePreProcessing(
                patch_method=None
            ),
            video_preprocessing: VideoPreProcessing = VideoPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            audio_preprocessing: AudioPreProcessing = AudioPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            marqo_version=get_version(),
            schema_template_version=get_version(),  # Default to current version like marqo_version
            created_at=int(time.time()),
            updated_at=int(time.time()),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True,
            filter_string_max_length=100,
            version=None,
            lexical_field_names=(),
            tensor_field_names=(),
            string_array_field_names=(),
            collapse_fields=None,
    ) -> SemiStructuredMarqoIndex:
        """
        Helper method that provides reasonable defaults for UnstructuredMarqoIndex.
        """
        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=schema_name or name,
            typeahead_schema_name=typeahead_schema_name,
            model=model,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            video_preprocessing=video_preprocessing,
            audio_preprocessing=audio_preprocessing,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            marqo_version=marqo_version,
            schema_template_version=schema_template_version,
            created_at=created_at,
            updated_at=updated_at,
            treat_urls_and_pointers_as_images=treat_urls_and_pointers_as_images,
            treat_urls_and_pointers_as_media=treat_urls_and_pointers_as_media,
            filter_string_max_length=filter_string_max_length,
            version=version,
            lexical_fields=[
                Field(name=field_name, type=FieldType.Text,
                      features=[FieldFeature.LexicalSearch],
                      lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}'
                      ) for field_name in lexical_field_names
            ],  # : List[Field]
            tensor_fields=[
                TensorField(
                    name=field_name,
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                ) for field_name in tensor_field_names
            ],  # : List[TensorField]
            string_array_fields=[
                StringArrayField(
                    name=field_name, type=FieldType.ArrayText, features=[FieldFeature.Filter],
                    string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}'
                ) for field_name in string_array_field_names
            ],
            collapse_fields=collapse_fields,
        )

    @classmethod
    def unstructured_marqo_index_request(
            cls,
            name: Optional[str] = None,
            model: Model = Model(
                name='random/small',
                text_query_prefix="",
                text_chunk_prefix=""
            ),
            normalize_embeddings: bool = True,
            text_preprocessing: TextPreProcessing = TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing: ImagePreProcessing = ImagePreProcessing(
                patch_method=None
            ),
            video_preprocessing: VideoPreProcessing = VideoPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            audio_preprocessing: AudioPreProcessing = AudioPreProcessing(
                split_length=20,
                split_overlap=1,
            ),
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            treat_urls_and_pointers_as_images: bool = False,
            treat_urls_and_pointers_as_media: bool = False,
            filter_string_max_length: int = 50,
            collapse_fields: Optional[List[CollapseField]] = None,
            marqo_version=version.get_version(),
            schema_template_version=version.get_version(),
            created_at=time.time(),
            updated_at=time.time(),
    ) -> UnstructuredMarqoIndexRequest:
        """
        Helper method that provides reasonable defaults for UnstructuredMarqoIndexRequest.
        """

        if not name:
            name = 'a' + str(uuid.uuid4()).replace('-', '')

        return UnstructuredMarqoIndexRequest(
            name=name,
            model=model,
            treat_urls_and_pointers_as_images=treat_urls_and_pointers_as_images,
            treat_urls_and_pointers_as_media=treat_urls_and_pointers_as_media,
            filter_string_max_length=filter_string_max_length,
            collapse_fields=collapse_fields,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            video_preprocessing=video_preprocessing,
            audio_preprocessing=audio_preprocessing,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            marqo_version=marqo_version,
            schema_template_version=schema_template_version,
            created_at=created_at,
            updated_at=updated_at,
        )