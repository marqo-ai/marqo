import unittest
from unittest.mock import MagicMock, patch

from marqo.core.inference.api import *
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline


class TestHuggingFaceModelInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock(spec=HuggingFaceModel)
        self.mock_model_config = MagicMock(spec=ModelConfig)
        self.mock_model.get_preprocessor.return_value = MagicMock()

    @patch("marqo.inference.native_inference.inference_pipeline."
           "hugging_face_model_inference_pipeline.split_prefix_preprocess_text")
    def test_content_preprocessing_text_modality(self, mock_split_preprocess):
        """Ensure that the content preprocessing is done correctly for text modalities."""
        inference_request = InferenceRequest(
            contents=["this is a test"],
            modality=Modality.TEXT,
            model_config=self.mock_model_config,
            preprocessing_config=MagicMock()
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)
        mock_split_preprocess.return_value = [["original", "preprocessed"]]

        result = pipeline._content_preprocessing()

        mock_split_preprocess.assert_called_once_with(
            inference_request.contents,
            self.mock_model.get_preprocessor.return_value,
            inference_request.preprocessing_config
        )
        self.assertEqual(result, [["original", "preprocessed"]])

    @patch("marqo.inference.native_inference.inference_pipeline."
           "hugging_face_model_inference_pipeline.split_prefix_preprocess_text")
    def test_content_preprocessing_non_text_modalities(self, mock_split_preprocess):
        """Ensure that the content preprocessing is done correctly for non-text modalities."""
        for modality in [Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]:
            with self.subTest(modality=modality):
                inference_request = InferenceRequest(
                    contents=["this is a test"],
                    modality=Modality.TEXT,
                    model_config=self.mock_model_config,
                    preprocessing_config=MagicMock()
                )

                pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)
                mock_split_preprocess.reset_mock()
                mock_split_preprocess.return_value = [["original", "preprocessed"]]

                result = pipeline._content_preprocessing()

                mock_split_preprocess.assert_called_once()
                args, kwargs = mock_split_preprocess.call_args
                self.assertEqual(args[0], inference_request.contents)
                self.assertEqual(args[1], self.mock_model.get_preprocessor.return_value)
                self.assertIsInstance(args[2], TextPreprocessingConfig)
                self.assertEqual(result, [["original", "preprocessed"]])