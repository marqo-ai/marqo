import time
from typing import List
from unittest import TestCase
from unittest.mock import patch, Mock

from marqo.core.models.marqo_index import Field, TensorField, Model, TextPreProcessing, TextSplitMethod, \
    ImagePreProcessing, VideoPreProcessing, AudioPreProcessing, DistanceMetric, VectorNumericType, HnswConfig, \
    StructuredMarqoIndex
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
            fields: List[Field] = None,
            tensor_fields: List[TensorField] = None,
            model: Model = Model(name='hf/all_datasets_v4_MiniLM-L6'),
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
            created_at=created_at,
            updated_at=updated_at,
            version=version
        )
