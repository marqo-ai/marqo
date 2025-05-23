import unittest

import numpy as np

from marqo.core.inference.api import InferenceRequest, Modality, ModelConfig, TextPreprocessingConfig
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal


class TestInferenceCache(unittest.TestCase):
    def setUp(self):
        self.inference_local = NativeInferenceLocal(DeviceManager())

        self.base_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["a", "b"],
            model_config=ModelConfig(
                model_name="hf/all_datasets_v4_MiniLM-L6",
                model_properties={
                    "name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                    "dimensions": 384,
                    "tokens": 128,
                    "type": "hf"
                }
            ),
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
            use_inference_cache=True
        )

    def test_caching_inference_should_return_same_result_as_its_delegate(self):
        caching_inference = CachingInference(self.inference_local, 10, "LRU")

        result_from_local_inference = self.inference_local.vectorise(self.base_request)
        result_from_caching_inference = caching_inference.vectorise(self.base_request)

        model_key = caching_inference.model_cache_key(self.base_request.model_config.model_properties)

        self.assertEqual(len(result_from_local_inference.result), len(result_from_caching_inference.result))
        for i in range(len(result_from_local_inference.result)):
            content1, embedding1 = result_from_local_inference.result[i][0]
            content2, embedding2 = result_from_caching_inference.result[i][0]
            self.assertEqual(content1, content2)
            self.assertTrue(np.array_equal(embedding1, embedding2))

            cached_embedding = caching_inference.inference_cache.get(model_key, content1)
            self.assertTrue(np.array_equal(embedding1, cached_embedding))

    def test_caching_inference_should_not_exceed_max_size(self):
        caching_inference = CachingInference(self.inference_local, 2, "LRU")

        result = caching_inference.vectorise(self.base_request.copy(update={"contents": ["1", "2", "3"]}))

        model_key = caching_inference.model_cache_key(self.base_request.model_config.model_properties)
        self.assertEqual(len(result.result), 3)
        self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)
        self.assertIsNone(caching_inference.inference_cache.get(model_key, "1"))
        self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "2"))
        self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "3"))
