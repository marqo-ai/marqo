import numpy as np
from numpy import ndarray

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from tests.integration_tests.test_case import InferenceTestCase, TestImageUrls


class TestHuggingfaceModelInferencePipeline(InferenceTestCase):
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
        model_name = "hf/all-MiniLM-L6-v2"
        text_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["text", "very long long long long text"],
            embedding_model_config=EmbeddingModelConfig(
                model_name=model_name,
                model_properties=self.get_model_properties_from_registry(model_name),
                normalize_embeddings=True,
            ),
            preprocessing_config=TextPreprocessingConfig(
                should_chunk=False,
            ),
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
        self.assertEqual((384,), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((384,), results_2[0][1].shape)
        self.assertEqual("very long long long long text", results_2[0][0])

    def test_inference_pipe_do_not_care_about_modality(self):
        """Ensure that the HuggingFaceModelInferencePipeline can still vectorise the content even if the modality is not
        TEXT."""
        model_name = "hf/e5-base-v2"
        text_inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=[TestImageUrls.IMAGE1.value],
            embedding_model_config=EmbeddingModelConfig(
                model_name=model_name,
                model_properties=self.get_model_properties_from_registry(model_name),
                normalize_embeddings=True,
            ),
            preprocessing_config=ImagePreprocessingConfig(
                download_header=dict(),
                download_thread_count=1,
            ),
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
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])
