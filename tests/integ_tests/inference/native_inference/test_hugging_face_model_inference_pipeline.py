import numpy as np

from integ_tests.inference.inference_test_case import InferenceTestCase
from marqo.core.inference.api import *
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from integ_tests.marqo_test import TestImageUrls
import pytest


class TestHuggingfaceModelInferencePipeline(InferenceTestCase):
    
    def setUp(self):
        self.inference = NativeInferenceLocal(device_manager=DeviceManager())
        
    def test_inference_text_no_chunk_no_prefix(self):
        """Test that the pipeline returns the embeddings for the two texts without chunking or prefix."""
        text_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["text", "very long long long long text"],
            device="cpu",
            model_config=ModelConfig(
                model_name="hf/e5-base-v2",
                model_properties={
                    "name": 'intfloat/e5-base-v2',
                    "dimensions": 768,
                    "tokens": 512,
                    "type": "hf",
                    "model_size": 0.438,
                },
                normalize_embeddings=True
            ),
            preprocessing_config=TextPreprocessingConfig(
                should_chunk=False,
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

    def test_inference_pipe_do_not_care_about_modality(self):
        """Ensure that the HuggingFaceModelInferencePipeline can still vectorise the content even if the modality is not
        TEXT."""
        text_inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=[TestImageUrls.IMAGE1.value],
            device="cpu",
            model_config=ModelConfig(
                model_name="hf/e5-base-v2",
                model_properties={
                    "name": 'intfloat/e5-base-v2',
                    "dimensions": 768,
                    "tokens": 512,
                    "type": "hf",
                },
                normalize_embeddings=True
            ),
            preprocessing_config=ImagePreprocessingConfig(
                download_header=dict(),
                download_thread_count=1,
            )
        )

        results = self.inference.vectorise(text_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 1)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

    @pytest.mark.largemodel
    def test_stella_model_also_work(self):
        """Stella models inherit from HuggingFaceModel, so it should work with the same pipeline.
        This is a test to ensure that the pipeline can handle it."""

        text_inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=[TestImageUrls.IMAGE1.value],
            device="cpu",
            model_config=ModelConfig(
                model_name="Marqo/dunzhang-stella_en_400M_v5",
                model_properties={
                    "name": "Marqo/dunzhang-stella_en_400M_v5",
                    "dimensions": 1024,
                    "tokens": 512,
                    "type": "hf_stella",
                    "trustRemoteCode": True,
                    "text_query_prefix": "Instruct: Given a web search query, "
                                         "retrieve relevant passages that answer the query.\nQuery: "
                },
                normalize_embeddings=True
            ),
            preprocessing_config=ImagePreprocessingConfig(
                download_header=dict(),
                download_thread_count=1,
            )
        )

        results = self.inference.vectorise(text_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 1)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((1024, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

