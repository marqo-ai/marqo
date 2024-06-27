import time
import unittest
import uuid
from unittest.mock import patch, Mock
import threading

import vespa.application as pyvespa

from marqo import config, version
from marqo.vespa.zookeeper_client import ZookeeperClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import (StructuredMarqoIndexRequest, UnstructuredMarqoIndexRequest,
                                                   FieldRequest, MarqoIndexRequest)
from marqo.core.monitoring.monitoring import Monitoring
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.vespa_client import VespaClient


class MarqoTestCase(unittest.TestCase):
    indexes = []
    create_lock = threading.Lock()
    delete_lock = threading.Lock()

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
        """
        # Lock method
        if cls.indexes:
            with cls.delete_lock:
                try:
                    cls.index_management.batch_delete_indexes(cls.indexes)
                except Exception as e:
                    print(f"Error deleting indexes: {e}")"""
        max_retries = 5
        retry_wait_time = 90  # seconds
        if cls.indexes:
            for attempt in range(max_retries):
                try:
                    cls.index_management.batch_delete_indexes(cls.indexes)
                    break
                except Exception as e: # TODO: Change this exception to something more specific
                    if attempt < max_retries - 1:
                        time.sleep(retry_wait_time)
                    else:
                        raise e

    @classmethod
    def setUpClass(cls) -> None:
        vespa_client = VespaClient(
            "http://localhost:19071",
            "http://localhost:8080",
            "http://localhost:8080",
            content_cluster_name="content_default",
        )
        zookeeper_client = ZookeeperClient(hosts="localhost:2181", zookeeper_connection_timeout=10)
        cls.configure_request_metrics()
        cls.vespa_client = vespa_client
        cls.zookeeper_client = zookeeper_client
        cls.index_management = IndexManagement(cls.vespa_client, cls.zookeeper_client, enable_index_operations=True)
        cls.monitoring = Monitoring(cls.vespa_client, cls.index_management)
        cls.config = config.Config(vespa_client=vespa_client, default_device="cpu",
                                   zookeeper_client=cls.zookeeper_client)

        cls.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)
        cls.CONTENT_CLUSTER = 'content_default'

    @classmethod
    def create_indexes(cls, index_requests: List[MarqoIndexRequest]) -> List[MarqoIndex]:
        # Retries. If fail, wait for 30 seconds and retry request up to 5 tries
        """
        # Lock method
        with cls.create_lock:
            try:
                indexes = cls.index_management.batch_create_indexes(index_requests)
                cls.indexes = cls.indexes + indexes
                return indexes
            except Exception as e:
                print(f"Error creating indexes: {e}")
                """
        max_retries = 5
        retry_wait_time = 90  # seconds
        for attempt in range(max_retries):
            try:
                indexes = cls.index_management.batch_create_indexes(index_requests)
                cls.indexes = cls.indexes + indexes
                break
            except Exception as e: # TODO: Change this exception to something more specific
                if attempt < max_retries - 1:
                    time.sleep(retry_wait_time)
                else:
                    raise e
        #indexes = cls.index_management.batch_create_indexes(index_requests)
        

        return indexes

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

    def clear_indexes(self, indexes: List[MarqoIndex]):
        for index in indexes:
            self.clear_index_by_name(index.schema_name)

    def clear_index_by_name(self, index_name: str):
        self.pyvespa_client.delete_all_docs(self.CONTENT_CLUSTER, index_name)

    def random_index_name(self) -> str:
        return 'a' + str(uuid.uuid4()).replace('-', '')

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
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            marqo_version=version.get_version(),
            created_at=time.time(),
            updated_at=time.time()
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
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            fields=fields,
            tensor_fields=tensor_fields,
            marqo_version=marqo_version,
            created_at=created_at,
            updated_at=updated_at
        )

    @classmethod
    def structured_marqo_index_request(
            cls,
            fields: List[FieldRequest],
            tensor_fields: List[str],
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
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            marqo_version=version.get_version(),
            created_at=time.time(),
            updated_at=time.time(),
    ) -> StructuredMarqoIndexRequest:
        """
        Helper method that provides reasonable defaults for StructuredMarqoIndexRequest.
        """
        if not name:
            name = 'a' + str(uuid.uuid4()).replace('-', '')

        return StructuredMarqoIndexRequest(
            name=name,
            model=model,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            fields=fields,
            tensor_fields=tensor_fields,
            marqo_version=marqo_version,
            created_at=created_at,
            updated_at=updated_at,
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
            distance_metric: DistanceMetric = DistanceMetric.Angular,
            vector_numeric_type: VectorNumericType = VectorNumericType.Float,
            hnsw_config: HnswConfig = HnswConfig(
                ef_construction=128,
                m=16
            ),
            treat_urls_and_pointers_as_images: bool = False,
            marqo_version='1.0.0',
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
            filter_string_max_length=20,
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            distance_metric=distance_metric,
            vector_numeric_type=vector_numeric_type,
            hnsw_config=hnsw_config,
            marqo_version=marqo_version,
            created_at=created_at,
            updated_at=updated_at,
        )

    class _AssertRaisesContext:
        def __init__(self, expected_exception):
            self.expected_exception = expected_exception

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, tb):
            self.exception = exc_value
            if exc_type is None:
                raise AssertionError(f"No exception raised, expected: '{self.expected_exception.__name__}'")
            if issubclass(exc_type, self.expected_exception) and exc_type is not self.expected_exception:
                raise AssertionError(
                    f"Subclass of '{self.expected_exception.__name__}' "
                    f"raised: '{exc_type.__name__}', expected exact exception.")
            if exc_type is not self.expected_exception:
                raise AssertionError(
                    f"Wrong exception raised: '{exc_type.__name__}', expected: '{self.expected_exception.__name__}'")
            return True

    def assertRaisesStrict(self, expected_exception):
        """
        Assert that a specific exception is raised. Will not pass for subclasses of the expected exception.
        """
        return self._AssertRaisesContext(expected_exception)


class AsyncMarqoTestCase(unittest.IsolatedAsyncioTestCase, MarqoTestCase):
    pass
