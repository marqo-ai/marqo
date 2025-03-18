from unittest import TestCase

import numpy as np

from marqo.core.inference.api import *
from marqo.inference.native_inference.inference_pipeline import InferencePipeline
from numpy import ndarray
from integ_tests.marqo_test import TestImageUrls


class TestOpenCLIPInferencePipeline(TestCase):

    def test_inference_text_no_chunk_no_prefix(self):
        text_inference_request = InferenceRequest(
            modality="language",
            contents=["text", "very long long long long text"],
            device="cpu",
            model_config=ModelConfig(
                model_name="test",
                model_properties={
                    "type": "open_clip",
                    "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                    "dimensions": 512
                },
                normalize_embeddings=True
            ),
            preprocessing_config=TextPreprocessingConfig(
                should_chunk=False
            )
        )

        results = InferencePipeline(text_inference_request).run_pipeline()

        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((512, ), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((512, ), results_2[0][1].shape)
        self.assertEqual("very long long long long text", results_2[0][0])

    def test_inference_image(self):
        image_inference_request = InferenceRequest(
            modality="image",
            contents = [
                TestImageUrls.IMAGE1.value,
                TestImageUrls.IMAGE2.value
            ],
            device="cpu",
            model_config=ModelConfig(
                model_name="test",
                model_properties={
                    "type": "open_clip",
                    "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                    "dimensions": 512
                },
                normalize_embeddings=True
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False,
                download_timeout_ms=1000,
                download_thread_count=1
            )
        )

        results = InferencePipeline(image_inference_request).run_pipeline()

        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((512, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((512, ), results_2[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE2.value, results_2[0][0])