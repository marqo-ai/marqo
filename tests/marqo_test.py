import unittest
from unittest.mock import patch, Mock

import vespa.application as pyvespa

from marqo import config
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.vespa_client import VespaClient


class MarqoTestCase(unittest.TestCase):
    indexes_to_delete = []

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
        if cls.indexes_to_delete:
            cls.index_management.batch_delete_indexes(cls.indexes_to_delete)

    @classmethod
    def setUpClass(cls) -> None:
        vespa_client = VespaClient(
            "http://localhost:19071",
            "http://localhost:8080",
            "http://localhost:8080"
        )
        cls.configure_request_metrics()
        cls.vespa_client = vespa_client
        cls.index_management = IndexManagement(cls.vespa_client)
        cls.config = config.Config(vespa_client=vespa_client, index_management=cls.index_management)

        cls.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)
        cls.CONTENT_CLUSTER = 'content_default'

    @classmethod
    def create_indexes(cls, indexes: List[MarqoIndex]):
        cls.index_management.batch_create_indexes(indexes)
        cls.vespa_client.wait_for_application_convergence()
        cls.indexes_to_delete = indexes

    def clear_indexes(self, indexes: List[MarqoIndex]):
        for index in indexes:
            self.pyvespa_client.delete_all_docs(self.CONTENT_CLUSTER, index.name)

    @classmethod
    def marqo_index(cls,
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
