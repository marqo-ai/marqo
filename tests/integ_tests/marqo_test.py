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

from marqo import config, version
from marqo.config import Config
from marqo.core.index_management.index_management import IndexManagement
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import (StructuredMarqoIndexRequest, UnstructuredMarqoIndexRequest,
                                                   FieldRequest, MarqoIndexRequest)
from marqo.core.monitoring.monitoring import Monitoring
from marqo.inference.native_inference.load_model import NativeModelManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient


class TestImageUrls(str, Enum):
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


class TestAudioUrls(str, Enum):
    __test__ = False
    AUDIO1 = "https://marqo-ecs-50-audio-test-dataset.s3.us-east-1.amazonaws.com/audios/1-100032-A-0.wav"
    AUDIO2 = "https://marqo-ecs-50-audio-test-dataset.s3.us-east-1.amazonaws.com/audios/1-115545-C-48.wav"
    AUDIO3 = "https://marqo-ecs-50-audio-test-dataset.s3.us-east-1.amazonaws.com/audios/1-119125-A-45.wav"

    MP3_AUDIO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample3.mp3"
    ACC_AUDIO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample3.aac"
    OGG_AUDIO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample3.ogg"

    FLAC_AUDIO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample3.flac"


class TestVideoUrls(str, Enum):
    __test__ = False
    VIDEO1 = "https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
    VIDEO2 = "https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4"
    VIDEO3 = "https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/--mI_-gaZLk_000018_000028.mp4"

    MKV_VIDEO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample_640x360.mkv"
    WEBM_VIDEO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample_640x360.webm"
    AVI_VIDEO1 = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/test-media-types/sample_640x360.avi"


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
        cls.index_management = IndexManagement(cls.vespa_client, cls.zookeeper_client, enable_index_operations=True,
                                               deployment_lock_timeout_seconds=2)
        cls.monitoring = Monitoring(cls.vespa_client, cls.index_management)
        cls.config = config.Config(vespa_client=vespa_client,
                                   inference=NativeInferenceLocal(DeviceManager()),
                                   model_manager=NativeModelManager(),
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
    def add_documents(cls, config: Config, add_docs_params: AddDocsParams) -> MarqoAddDocumentsResponse:
        return config.document.add_documents(add_docs_params)

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

    def clear_indexes(self, indexes: List[MarqoIndex]):
        for index in indexes:
            self.clear_index_by_schema_name(index.schema_name)

    def clear_index_by_index_name(self, index_name: str):
        """Delete all documents in the given index.

        Args:
            index_name: The name of the index to clear.
        """
        schema_name = self.index_management.get_index(index_name).schema_name
        return self.clear_index_by_schema_name(schema_name)

    def clear_index_by_schema_name(self, schema_name: str):
        """Delete all documents in the given index.

        Args:
            schema_name: The schema name of the index to clear. It is not the same as the index name.
        """
        self.pyvespa_client.delete_all_docs(self.CONTENT_CLUSTER, schema_name)

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


EXAMPLE_FASHION_DOCUMENTS = [
  {
    "_id": "1",
    "title": "Slim Fit Denim Jacket",
    "brand": "SnugNest",
    "description": "A timeless piece with a modern slim-fit design, perfect for casual layering.",
    "color": "yellow",
    "size": "S",
    "style": "casual",
    "price": 83.42
  },
  {
    "_id": "2",
    "title": "Classic Cotton Shirt",
    "brand": "SnugNest",
    "description": "Comfortable and breathable cotton shirt suitable for everyday wear.",
    "color": "red",
    "size": "M",
    "style": "partywear",
    "price": 49.03
  },
  {
    "_id": "3",
    "title": "High-Waisted Skirt",
    "brand": "PulseWear",
    "description": "Elegant skirt with a high waistline and flattering silhouette.",
    "color": "coral",
    "size": "L",
    "style": "streetwear",
    "price": 1.2
  },
  {
    "_id": "4",
    "title": "Knitted Winter Sweater",
    "brand": "SprintX",
    "description": "Chunky knit sweater designed for warmth and comfort in cold seasons.",
    "color": "red",
    "size": "Free",
    "style": "loungewear",
    "price": 92.99
  },
  {
    "_id": "5",
    "title": "Casual Linen Trousers",
    "brand": "PulseWear",
    "description": "Relaxed-fit trousers crafted from lightweight linen for maximum comfort.",
    "color": "charcoal",
    "size": "M",
    "style": "partywear",
    "price": 88.14
  },
  {
    "_id": "6",
    "title": "Embroidered Kurta",
    "brand": "RetroHue",
    "description": "Traditional kurta with intricate embroidery for festive occasions.",
    "color": "green",
    "size": "S",
    "style": "streetwear",
    "price": 81.33
  },
  {
    "_id": "7",
    "title": "Floral Summer Dress",
    "brand": "SnugNest",
    "description": "Breezy and lightweight dress ideal for sunny summer days.",
    "color": "green",
    "size": "XS",
    "style": "streetwear",
    "price": 28.71
  },
  {
    "_id": "8",
    "title": "Athletic Running Shorts",
    "brand": "PulseWear",
    "description": "Performance shorts made from moisture-wicking fabric for workouts.",
    "color": "green",
    "size": "Free",
    "style": "biker",
    "price": 73.88
  },
  {
    "_id": "9",
    "title": "Hooded Windbreaker",
    "brand": "CozyCore",
    "description": "Windproof and waterproof jacket with adjustable hood.",
    "color": "charcoal",
    "size": "S",
    "style": "streetwear",
    "price": 55.54
  },
  {
    "_id": "10",
    "title": "Fleece Zip-Up Hoodie",
    "brand": "SnugNest",
    "description": "Super soft fleece hoodie for a relaxed and cozy look.",
    "color": "gray",
    "size": "M",
    "style": "loungewear",
    "price": 49.3
  }
]