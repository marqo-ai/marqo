import unittest
from unittest.mock import patch, Mock

from marqo import config
from marqo.core.models.marqo_index import *
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.vespa_client import VespaClient


class MarqoTestCase(unittest.TestCase):

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
    def tearDownClass(cls):
        cls.patcher.stop()

    @classmethod
    def setUpClass(cls) -> None:
        vespa_client = VespaClient(
            "http://localhost:19071",
            "http://localhost:8080",
            "http://localhost:8080"
        )
        cls.configure_request_metrics()
        cls.vespa_client = vespa_client
        cls.config = config.Config(vespa_client=vespa_client)

    def marqo_index(self,
                    name: str,
                    type: IndexType,
                    fields: Optional[List[Field]] = None,
                    tensor_fields: Optional[List[TensorField]] = None,
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
                    treat_urls_and_pointers_as_images: Optional[bool] = None,
                    distance_metric: DistanceMetric = DistanceMetric.Angular,
                    vector_numeric_type: VectorNumericType = VectorNumericType.Float,
                    hnsw_config: HnswConfig = HnswConfig(
                        ef_construction=128,
                        m=16
                    )
                    ):
        """
        Helper method that provides reasonable defaults for MarqoIndex.
        """
        return MarqoIndex(
            name=name,
            type=type,
            model=model,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            treat_urls_and_pointers_as_images=treat_urls_and_pointers_as_images,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            fields=fields,
            tensor_fields=tensor_fields
        )


class AsyncMarqoTestCase(unittest.IsolatedAsyncioTestCase, MarqoTestCase):
    pass
