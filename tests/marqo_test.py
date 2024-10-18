import contextlib
import socket
import threading
import time
import unittest
import uuid
from typing import Generator
from unittest.mock import patch, Mock

import uvicorn
import vespa.application as pyvespa
from starlette.applications import Starlette

from marqo import config, version, tensor_search
from marqo.tensor_search import index_meta_cache, tensor_search
from marqo.vespa.zookeeper_client import ZookeeperClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import (StructuredMarqoIndexRequest, UnstructuredMarqoIndexRequest,
                                                   FieldRequest, MarqoIndexRequest)
from marqo.core.monitoring.monitoring import Monitoring
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.vespa_client import VespaClient


class TestImageUrls(Enum):
    __test__ = False  # Prevent pytest from collecting this class as a test
    IMAGE0 = 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg'
    IMAGE1 = 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg'
    IMAGE2 = 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg'
    IMAGE3 = 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg'
    IMAGE4 = 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg'
    COCO = 'https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg'
    HIPPO_REALISTIC = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic_small.png'
    HIPPO_REALISTIC_LARGE = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    HIPPO_STATUE = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue_small.png'


class MarqoTestCase(unittest.TestCase):
    indexes = []

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
        if cls.indexes:
            cls.index_management.batch_delete_indexes_by_name([index.name for index in cls.indexes])

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
        cls.index_management.bootstrap_vespa()
        indexes = cls.index_management.batch_create_indexes(index_requests)
        cls.indexes = indexes

        return indexes

    @classmethod
    def add_documents(cls, *args, **kwargs):
        # TODO change to use config.document.add_documents when tensor_search.add_documents is removed
        return tensor_search.add_documents(*args, **kwargs)

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
            marqo_version=version.get_version(),
            created_at=time.time(),
            updated_at=time.time(),
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

    @classmethod
    def unstructured_marqo_index(
            cls,
            name: str,
            schema_name: str,
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
            marqo_version=version.get_version(),
            created_at=time.time(),
            updated_at=time.time(),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True,
            filter_string_max_length=100,
            version=None
    ) -> UnstructuredMarqoIndex:
        """
        Helper method that provides reasonable defaults for UnstructuredMarqoIndex.
        """
        return UnstructuredMarqoIndex(
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
            marqo_version=marqo_version,
            created_at=created_at,
            updated_at=updated_at,
            treat_urls_and_pointers_as_images=treat_urls_and_pointers_as_images,
            treat_urls_and_pointers_as_media=treat_urls_and_pointers_as_media,
            filter_string_max_length=filter_string_max_length,
            version=version
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
            marqo_version=version.get_version(),
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
            normalize_embeddings=normalize_embeddings,
            text_preprocessing=text_preprocessing,
            image_preprocessing=image_preprocessing,
            video_preprocessing=video_preprocessing,
            audio_preprocessing=audio_preprocessing,
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


class MockHttpServer:
    """
    A MockHttpServer that takes a Starlette app as input, start the uvicorn server
    in a thread, and yield the server url (with random port binding). After the test,
    it automatically shuts down the server.

    This can be used in individual tests, or as a test fixture in class or module scope.
    Example usage:

    app = Starlette(routes=[
        Route('/path1', lambda _: Response({"a":"b"}, status_code=200)),
        Route('/image.jpg', lambda _: Response(b'\x00\x00\x00\xff', media_type='image/png')),
    ])

    with MockHttpServer(app).run_in_thread() as base_url:
        run_some_tests
    """
    def __init__(self, app: Starlette):
        self.server = uvicorn.Server(config=uvicorn.Config(app=app))

    @contextlib.contextmanager
    def run_in_thread(self) -> Generator[str, None, None]:
        (sock := socket.socket()).bind(("127.0.0.1", 0))
        thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]})
        thread.start()
        try:
            while not self.server.started:
                time.sleep(1)
            address, port = sock.getsockname()
            yield f'http://{address}:{port}'
        finally:
            self.server.should_exit = True
            thread.join()