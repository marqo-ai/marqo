import numpy as np

from integ_tests.inference.inference_test_case import InferenceTestCase
from integ_tests.marqo_test import TestImageUrls
from marqo.core.inference.api import *
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal


class TestRandomModelInferencePipeline(InferenceTestCase):

    def setUp(self):
        self.inference = NativeInferenceLocal(device_manager=DeviceManager())

    def test_inference_text_no_chunk_no_prefix(self):
        """Test that the pipeline returns the embeddings for the two texts without chunking or prefix."""
        text_inference_request = InferenceRequest(
            modality="language",
            contents=["text", "very long long long long text"],
            device="cpu",
            model_config=ModelConfig(
                model_name="random/small",
                normalize_embeddings=True,
                model_properties={
                    "name": "random/small",
                    "dimensions": 32,
                    "tokens": 128,
                    "type": "random",
                    "notes": ""
                }
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
        self.assertEqual((32, ), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((32, ), results_2[0][1].shape)
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
                model_name="random/small",
                model_properties={
                    "name": "random/small",
                    "dimensions": 32,
                    "tokens": 128,
                    "type": "random",
                    "notes": ""
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
        self.assertEqual((32, ), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((32, ), results_2[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE2.value, results_2[0][0])

    def test_to_ensure_same_content_generate_same_embeddings(self):
        """Ensure that the same content generates the same embeddings,
        whether it is in a list with other content or on its own."""

        # Common parameters
        device = "cpu"
        model_config = ModelConfig(
            model_name="random/small",
            normalize_embeddings=True,
            model_properties={
                "name": "random/small",
                "dimensions": 32,
                "tokens": 128,
                "type": "random",
                "notes": ""
            }
        )

        # The text we want to check
        target_text = "consistent text"

        # First inference request: single item list
        single_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=[target_text],
            device=device,
            model_config=model_config,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False)
        )

        # Second inference request: target_text in the middle of other inputs
        multi_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["another text", target_text, "yet another text"],
            device=device,
            model_config=model_config,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False)
        )

        # Perform vectorisation
        native_inference = self.inference

        single_result = native_inference.vectorise(single_inference_request)
        multi_result = native_inference.vectorise(multi_inference_request)

        # Extract embeddings from the inference result
        # single_result.result = [[(content, embedding)]]
        single_embedding = single_result.result[0][0][1]  # (content, embedding)
        multi_embedding = multi_result.result[1][0][1]

        other_embedding_1 = multi_result.result[0][0][1]
        other_embedding_2 = multi_result.result[2][0][1]

        self.assertTrue(np.allclose(single_embedding, multi_embedding))
        self.assertFalse(np.allclose(single_embedding, other_embedding_1))
        self.assertFalse(np.allclose(single_embedding, other_embedding_2))