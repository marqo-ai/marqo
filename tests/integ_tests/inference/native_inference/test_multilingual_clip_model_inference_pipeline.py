import os
from unittest.mock import patch

import numpy as np

from integ_tests.inference.inference_test_case import InferenceTestCase
from integ_tests.marqo_test import TestImageUrls
from marqo.core.inference.api import *
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal


class TestMultilingualCLIPInferencePipeline(InferenceTestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.device_patcher = patch.dict(os.environ, {
            "MARQO_MAX_CPU_MODEL_MEMORY": "15"
        })
        cls.device_patcher.start()
    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.device_patcher.stop()

    def setUp(self):
        self.inference = NativeInferenceLocal(device_manager=DeviceManager())

    def test_inference_text_no_chunk_no_prefix(self):
        """Test that the pipeline returns the embeddings for the two texts without chunking or prefix."""
        text_inference_request = InferenceRequest(
            modality="language",
            contents=["text", "very long long long long text"],
            device="cpu",
            model_config=ModelConfig(
                model_name="multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                model_properties={                    
                    "name": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                    "visual_model": "open_clip/ViT-L-14/openai",
                    "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                    "dimensions": 768,
                    "type": "multilingual_clip",
            },
                normalize_embeddings=True
            ),
            preprocessing_config=TextPreprocessingConfig(
                should_chunk=False
            )
        )

        results = self.inference.vectorise(text_inference_request)

        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768, ), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768, ), results_2[0][1].shape)
        self.assertEqual("very long long long long text", results_2[0][0])

    def test_inference_two_valid_images(self):
        """Test that the pipeline returns the embeddings for the two valid images."""
        image_inference_request = InferenceRequest(
            modality="image",
            contents = [
                TestImageUrls.IMAGE1.value,
                TestImageUrls.IMAGE2.value
            ],
            device="cpu",
            model_config=ModelConfig(
                model_name="multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                model_properties={
                    "name": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                    "visual_model": "open_clip/ViT-L-14/openai",
                    "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                    "dimensions": 768,
                    "type": "multilingual_clip",
                },
                normalize_embeddings=True
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False,
                download_timeout_ms=1000,
                download_thread_count=1
            )
        )

        results = self.inference.vectorise(image_inference_request)

        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768, ), results_2[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE2.value, results_2[0][0])

    def test_inference_image_with_one_image_error(self):
        """Test that the pipeline returns an error for the image that failed to download but
        still returns the embeddings for the image that was successfully downloaded."""
        image_inference_request = InferenceRequest(
            modality="image",
            contents = [
                TestImageUrls.IMAGE1.value,
                TestImageUrls.IMAGE2.value + "invalid"
            ],
            device="cpu",
            model_config=ModelConfig(
                model_name="multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                model_properties={
                    "name": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                    "visual_model": "open_clip/ViT-L-14/openai",
                    "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                    "dimensions": 768,
                    "type": "multilingual_clip",
                },
                normalize_embeddings=True
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False,
                download_timeout_ms=1000,
                download_thread_count=1
            )
        )

        results = self.inference.vectorise(image_inference_request)

        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2 = results.result[1]
        self.assertTrue(isinstance(results_2, InferenceErrorModel))