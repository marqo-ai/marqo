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
import unittest

from tests import marqo_test
from tests import utilities

from marqo import Client


class TestEnvVarChanges(marqo_test.MarqoTestCase):

    """
        All tests that rerun marqo with different env vars should go here
        Teardown will handle resetting marqo back to base settings
    """
    
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
                "name": "ViT-B-32-quickgelu",
                "dimensions": 512,
                "type": "open_clip",
                "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
            }
        }

        print(f"Attempting to rerun marqo with custom model {open_clip_model_object['model']}")
        utilities.rerun_marqo_with_env_vars(
            env_vars = ['-e', f"MARQO_MODELS_TO_PRELOAD=[{json.dumps(open_clip_model_object)}]"],
            calling_class=self.__class__.__name__
        )

        # check preloaded models (should be custom model)
        custom_models = ["open-clip-1"]
        self.client.create_index("test_index_for_preloaded_models")
        res = self.client.index("test_index_for_preload_models").get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(custom_models)

    @unittest.skip(reason="Temproraliy skips this until inference caching is implemented")
    def test_multiple_env_vars(self):
        # TODO: Add log test
        """
            Ensures that rerun_marqo_with_env_vars can work with several different env vars
            at the same time

            3 things in the same command:
            1. Load models
            2. set max EF
            3. set log level to debug

            Also, asserts that Marqo's debug output is as expected
        """

        # Restart marqo with new max values
        max_ef = 6000
        new_models = ["hf/e5-large-v2"]
        index_name = "test_multiple_env_vars"
        utilities.rerun_marqo_with_env_vars(
            env_vars=[
                "-e", f"MARQO_EF_CONSTRUCTION_MAX_VALUE={max_ef}",
                "-e", f"MARQO_MODELS_TO_PRELOAD={json.dumps(new_models)}",
                "-e", f"MARQO_LOG_LEVEL=debug",
                "-e", f"MARQO_INFERENCE_CACHE_SIZE=10"
            ],
            calling_class=self.__class__.__name__
        )

        # Create index with same number of replicas and EF
        res_0 = self.client.create_index(index_name=index_name, ann_parameters={
            "spaceType": 'prenormalized-angular', "parameters": {"efConstruction": 5000, "m": 16}}
        )

        # Assert correct EF const
        assert self.client.index(index_name).get_settings() \
                   ["annParameters"]["parameters"]["efConstruction"] == 5000

        # Assert correct models
        res = self.client.index(index_name).get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(new_models)

        # Test inference cache
        telemetry_client = Client(**self.client_settings, return_telemetry=True)

        inference_time = 10
        cache_reading_time = 5

        # Single query
        # First search
        r = telemetry_client.index(index_name).search(q="test")
        self.assertTrue(r["telemetry"]["timesMs"]["search.vector_inference_full_pipeline"] > inference_time)
        # Second search
        single_q_cache_time = []
        for _ in range(5):
            r = telemetry_client.index(index_name).search(q="test")
            single_q_cache_time.append(r["telemetry"]["timesMs"]["search.vector_inference_full_pipeline"])
        self.assertTrue(min(single_q_cache_time) < cache_reading_time, single_q_cache_time)

        # Multiple queries
        r = telemetry_client.index(index_name).search(q={"random": 1, "query": 2})
        self.assertTrue(r["telemetry"]["timesMs"]["search.vector_inference_full_pipeline"] > inference_time)
        # Second search
        multiple_q_cache_time = []
        for _ in range(5):
            r = telemetry_client.index(index_name).search(q={"random": 1, "query": 2})
            multiple_q_cache_time.append(r["telemetry"]["timesMs"]["search.vector_inference_full_pipeline"])
        self.assertTrue(min(multiple_q_cache_time) < cache_reading_time, multiple_q_cache_time)

        # Test to ensure inference cache is not working for add_documents:
        for _ in range(3):
            r = telemetry_client.index(index_name).add_documents([{"test": "test"}],
                                                                 tensor_fields=["test"])
            self.assertTrue(r["telemetry"]["timesMs"]["add_documents.create_vectors"] > inference_time)