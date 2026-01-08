"""

To test these functions locally:

1. Run a Marqo container another terminal (these tests assume there is a running Marqo
    container and then try to kill it, failing if unsuccessful)
2. cd into the root of this repo
3. Run the following command (you can replace MARQO_IMAGE_NAME):

    TESTING_CONFIGURATION=DIND_MARQO_OS \
    MARQO_API_TESTS_ROOT=. \
    MARQO_IMAGE_NAME=marqoai/marqo:latest \
    pytest tests/application_tests/test_env_var_changes.py::TestEnvVarChanges::test_multiple_env_vars


We may test multiple different env vars in the same test case. This is because
 each new test case is expensive, requiring a restart of Marqo. This prevents
 this test suite's runtime from growing too large.
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, List
import time

import requests
from marqo import Client
from tests import marqo_test
from tests import utilities


class TestEnvVarChanges(marqo_test.MarqoTestCase):
    """
        All tests that rerun marqo with different env vars should go here
        Teardown will handle resetting marqo back to base settings
    """

    def _wait_for_container_to_be_ready(self, url: str, timeout: int = 60, container_name: str = "marqo") -> None:
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Container is ready!")
                    return
            except requests.exceptions.RequestException:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Container {container_name} did not become ready within {timeout} seconds.")
            time.sleep(5)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        # Ensures that marqo goes back to default state after these tests
        utilities.rerun_marqo_with_default_config(
            calling_class=cls.__name__
        )
        print("Marqo has been rerun with default env vars!")

    def test_preload_models(self):
        # TODO: Add log test
        """
        Tests rerunning marqo with non-default, custom model.
        Default models are ["hf/all_datasets_v4_MiniLM-L6", "ViT-L/14"]

        Also, this tests log output when log level is not set. The log level should be INFO by default.
        """

        open_clip_model_object = {
            "model": "open-clip-1",
            "modelProperties": {
                "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                "dimensions": 512,
                "type": "open_clip",
                "tritonImageEncoderProperties": {
                    "maxBatchSize": 8,
                    "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-image-encoder",
                    "sources": [
                        "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/image-encoder/model.onnx"
                    ],
                    "input": [
                        {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
                    ],
                    "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
                },
                "tritonTextEncoderProperties": {
                    "maxBatchSize": 16,
                    "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-text-encoder",
                    "sources": [
                        "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/text-encoder/model.onnx",
                    ],
                    "input": [{"name": "input", "dims": [77], "dataType": "TYPE_INT32"}],
                    "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
                },
            },
        }

        index_name = "test_index_for_preload_models" + str(uuid.uuid4())[:4]

        print(f"Attempting to rerun marqo with custom model {open_clip_model_object['model']}")
        utilities.rerun_marqo_with_env_vars(
            env_vars={"MARQO_MODELS_TO_PRELOAD": json.dumps([open_clip_model_object])},
            calling_class=self.__class__.__name__,
            target_service="mioc"
        )

        # check preloaded models (should be custom model)
        custom_models = ["open-clip-1"]
        self.client.create_index(index_name=index_name)
        # Wait for model loading to be ready
        self._wait_for_container_to_be_ready("http://localhost:8884/healthz", container_name="mioc")
        res = self.client.index(index_name).get_loaded_models()
        self.assertTrue(
            res["models"][0]["modelName"].startswith("open-clip-1"),
            f"Expected preloaded model to be {custom_models}, but got {res['models']}"
        )

    def test_inference_cache(self):
        """
            Ensures that inference cache works for search but not add_docs when enabled
        """

        # Restart marqo with new max values
        new_models = ["open_clip/ViT-B-32/laion2b_s34b_b79k"]
        index_name = "test_multiple_env_vars" + str(uuid.uuid4())[:4]
        utilities.rerun_marqo_with_env_vars(
            env_vars={
                "MARQO_MODELS_TO_PRELOAD": json.dumps(new_models),
                "MARQO_INFERENCE_CACHE_SIZE": "10",  # enable cache on inference side
            },
            calling_class=self.__class__.__name__,
            target_service="mioc",
        )

        self._wait_for_container_to_be_ready("http://localhost:8884/healthz", container_name="mioc")

        utilities.rerun_marqo_with_env_vars(
            env_vars={
                "MARQO_API_INFERENCE_CACHE_SIZE": "10",  # enable inference cache on api side
            },
            calling_class=self.__class__.__name__,
            target_service="api",
        )
        self._wait_for_container_to_be_ready("http://localhost:8882/health", container_name="api")


        # Create index with same number of replicas and EF
        self.client.create_index(index_name=index_name, ann_parameters={
            "spaceType": 'prenormalized-angular', "parameters": {"efConstruction": 5000, "m": 16}}
                                 )

        # Assert correct EF const
        assert self.client.index(index_name).get_settings() \
                   ["annParameters"]["parameters"]["efConstruction"] == 5000

        # Assert correct models
        res = self.client.index(index_name).get_loaded_models()
        self.assertIn(
            "open_clip/ViT-B-32/laion2b_s34b_b79k", res["models"][0]["modelName"]
        )

        # Test inference cache
        telemetry_client = Client(**self.client_settings, return_telemetry=True)

        min_inference_time_ms = 5  # inference usually takes at least 5ms
        cache_reading_time_ms = 3  # if it hits cache, the pipeline should take less than 3ms

        # Test search query's embedding is cached when inference cache is enabled
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        image_url = marqo_test.TestImageUrls.HIPPO_STATUE.value
        for query in ["test", {"random": 1, "query": 2}, base64_image, image_url]:
            with self.subTest(f"Search query: {query}"):
                # Single query
                # First search that misses cache should take longer
                r = telemetry_client.index(index_name).search(q=query)
                self.assertTrue(
                    r["telemetry"]["timesMs"]["search.vector_inference_full_pipeline"] > min_inference_time_ms)

                # Run a few more times to make sure we populate it on API side cache as well as inference side cache
                self._run_in_threads(lambda client: client.index(index_name).search(q=query),
                                     max_workers=5, count=50)

                # Following searches should hit cache, average latency should be low
                inference_latency = self._run_in_threads(
                    lambda client: client.index(index_name).search(q=query),
                    max_workers=1, count=10, telemetry_name="search.vector_inference_full_pipeline")

                if query == image_url:
                    # image url is not cached, so avg latency will usually be > min_inference_time_ms
                    self.assertTrue(sum(inference_latency) / 10 > min_inference_time_ms, inference_latency)
                else:
                    # other queries are all cached, so avg latency should be < cache_reading_time_ms
                    self.assertTrue(sum(inference_latency) / 10 < cache_reading_time_ms, inference_latency)

        # Test to ensure inference cache is not working for add_documents:
        with self.subTest("Add document"):
            # we do add doc one at a time to reduce the load on inference so the latency is small, we then verify
            # all the latency telemetry data points are larger than the min_inference_time_ms to verify cache is not
            # involved in this process
            inference_latency = self._run_in_threads(
                lambda client: client.index(index_name).add_documents([{"test": "test"}], tensor_fields=["test"]),
                max_workers=1, count=10, telemetry_name="add_documents.inference.all"
            )
            self.assertTrue(all([latency > min_inference_time_ms for latency in inference_latency]), inference_latency)

    def _run_in_threads(self, operation: Callable[[Client], dict], max_workers: int,
                        count: int, telemetry_name: Optional[str] = None) -> List[float]:
        results = []

        # Using ThreadPoolExecutor to simulate concurrent access, we use a new client every time to avoid
        # connection pooling, so we can hit most api workers in split mode
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(operation, Client(**self.client_settings, return_telemetry=True))
                       for _ in range(count)]

            # Collect results or errors from the futures
            for future in as_completed(futures):
                try:
                    res = future.result()

                    if telemetry_name:
                        results.append(res["telemetry"]["timesMs"][telemetry_name])
                except Exception as e:
                    self.fail(f'Exception raised when collecting results: {e}')

        return results
