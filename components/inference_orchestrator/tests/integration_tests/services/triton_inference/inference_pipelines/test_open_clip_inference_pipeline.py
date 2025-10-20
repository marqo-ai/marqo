import numpy as np
from numpy import ndarray

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from tests.integration_tests.test_case import InferenceTestCase, TestImageUrls


class TestOpenCLIPInferencePipeline(InferenceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.eject_all_models()

    def test_inference_text_no_chunk_no_prefix(self):
        """Test that the pipeline returns the embeddings for the two texts without chunking or prefix."""
        model_name = "Marqo/marqo-fashionSigLIP"
        text_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["text", "very long long long long text"],
            embedding_model_config=EmbeddingModelConfig(
                model_name=model_name,
                model_properties=self.get_model_properties_from_registry(model_name),
                normalize_embeddings=True,
            ),
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
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
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768,), results_2[0][1].shape)
        self.assertEqual("very long long long long text", results_2[0][0])

    def test_inference_two_valid_images(self):
        """Test that the pipeline returns the embeddings for the two valid images."""
        image_inference_request = InferenceRequest(
            modality="image",
            contents=[TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value],
            embedding_model_config=EmbeddingModelConfig(
                model_name="open_clip/ViT-B-32/laion2b_s34b_b79k",
                model_properties=self.get_model_properties_from_registry(
                    "open_clip/ViT-B-32/laion2b_s34b_b79k"
                ),
                normalize_embeddings=True,
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False, download_timeout_ms=1000, download_thread_count=1
            ),
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
        self.assertEqual((512,), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((512,), results_2[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE2.value, results_2[0][0])

    def test_inference_image_with_one_image_error(self):
        """Test that the pipeline returns an error for the image that failed to download but
        still returns the embeddings for the image that was successfully downloaded."""
        image_inference_request = InferenceRequest(
            modality="image",
            contents=[
                TestImageUrls.IMAGE1.value,
                TestImageUrls.IMAGE2.value + "invalid",
            ],
            embedding_model_config=EmbeddingModelConfig(
                model_name="open_clip/ViT-B-32/laion2b_s34b_b79k",
                model_properties=self.get_model_properties_from_registry(
                    "open_clip/ViT-B-32/laion2b_s34b_b79k"
                ),
                normalize_embeddings=True,
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False, download_timeout_ms=1000, download_thread_count=1
            ),
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
        self.assertEqual((512,), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2 = results.result[1]
        self.assertTrue(isinstance(results_2, InferenceErrorModel))
