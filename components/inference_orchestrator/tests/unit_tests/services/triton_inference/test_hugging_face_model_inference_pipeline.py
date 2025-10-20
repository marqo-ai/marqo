import unittest
from unittest.mock import MagicMock, patch

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.triton_inference.embedding_models.hugging_face import (
    HuggingFaceModel,
)
from inference_orchestrator.services.triton_inference.inference_pipelines import (
    HuggingFaceModelInferencePipeline,
)


class TestHuggingFaceModelInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=HuggingFaceModel)
        self.mock_model_config = MagicMock(spec=EmbeddingModelConfig)
        self.mock_model.get_preprocessor.return_value = MagicMock()

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_content_preprocessing_text_modality(self, mock_split_preprocess):
        """Ensure that the content preprocessing is done correctly for text modalities."""
        inference_request = InferenceRequest(
            contents=["this is a test"],
            modality=Modality.TEXT,
            embedding_model_config=self.mock_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)
        mock_split_preprocess.return_value = [["original", "preprocessed"]]

        result = pipeline._content_preprocessing()

        mock_split_preprocess.assert_called_once_with(
            inference_request.contents,
            self.mock_model.get_preprocessor.return_value,
            inference_request.preprocessing_config,
        )
        self.assertEqual(result, [["original", "preprocessed"]])
