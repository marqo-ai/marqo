import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from marqo.core.inference.api import *
from marqo.inference.native_inference.embedding_models.multilingual_clip_model import (
    MultilingualCLIPModel, MultilingualCLIPTokenizerWrapper, MultilingualCLIPPreprocessor
)
from marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline import (
    MultilingualCLIPModelInferencePipeline
)
from marqo.inference.type import Modality


class TestMultilingualCLIPInferencePipeline(unittest.TestCase):

    @patch("marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline.split_prefix_preprocess_text")
    @patch("marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline.download_and_preprocess_image")
    def test_content_preprocessing_text(self, mock_download, mock_split):
        model = MagicMock(spec=MultilingualCLIPModel)
        model_config = MagicMock(spec=ModelConfig)
        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["hello world"],
            preprocessing_config=TextPreprocessingConfig(),
            return_individual_error=False,
            model_config=model_config
        )
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        mock_split.return_value = [[("original text", "preprocessed text")]]
        result = pipeline._content_preprocessing()

        mock_split.assert_called_once()
        mock_download.assert_not_called()
        self.assertEqual(result, [[("original text", "preprocessed text")]])

    @patch("marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline.split_prefix_preprocess_text")
    @patch("marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline.download_and_preprocess_image")
    def test_content_preprocessing_image(self, mock_download, mock_split):
        model = MagicMock(spec=MultilingualCLIPModel)
        model_config = MagicMock(spec=ModelConfig)
        inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=["http://some.image/url.jpg"],
            preprocessing_config=ImagePreprocessingConfig(),
            return_individual_error=False,
            model_config=model_config
        )
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        mock_download.return_value = [[("original image", torch.tensor([1.0]))]]
        result = pipeline._content_preprocessing()

        mock_download.assert_called_once()
        mock_split.assert_not_called()
        self.assertEqual(result, [[("original image", torch.tensor([1.0]))]])

    def test_collect_valid_content_to_encode(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        inference_request = MagicMock()
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        preprocessed_content = [
            [("original", torch.tensor([1.0]))],
            [("original2", torch.tensor([2.0]))]
        ]

        collected = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(len(collected), 2)
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in collected))

    def test_collect_valid_content_to_encode_with_inference_error(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        inference_request = MagicMock()
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        preprocessed_content = [
            InferenceErrorModel(error_message="something went wrong"),
            [("original", torch.tensor([1.0]))]
        ]

        collected = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(len(collected), 1)
        self.assertIsInstance(collected[0], torch.Tensor)

    def test_collect_valid_content_to_encode_invalid_type(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        inference_request = MagicMock()
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        preprocessed_content = [
            "invalid_content_type"
        ]

        with self.assertRaises(ValueError) as cm:
            pipeline._collect_valid_content_to_encode(preprocessed_content)
        self.assertIn("Unexpected content type", str(cm.exception))

    def test_encode_processed_content_empty(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        inference_request = MagicMock()
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        embeddings = pipeline._encode_processed_content([])

        self.assertEqual(embeddings, [])

    def test_encode_processed_content_success(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        model.encode.return_value = [np.array([0.1, 0.2])]
        inference_request = MagicMock()
        inference_request.modality = Modality.TEXT
        inference_request.model_config.normalize_embeddings = True

        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        preprocessed_content = [
            [("original", torch.tensor([1.0]))]
        ]

        embeddings = pipeline._encode_processed_content(preprocessed_content)

        model.encode.assert_called_once()
        self.assertEqual(len(embeddings), 1)
        self.assertTrue(isinstance(embeddings[0], np.ndarray))

    def test_encode_processed_content_mismatch(self):
        model = MagicMock(spec=MultilingualCLIPModel)
        model.encode.return_value = [np.array([0.1, 0.2])]  # Return fewer embeddings than inputs
        inference_request = MagicMock()
        inference_request.modality = Modality.TEXT
        inference_request.model_config.normalize_embeddings = True

        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        preprocessed_content = [
            [("original1", torch.tensor([1.0]))],
            [("original2", torch.tensor([2.0]))]
        ]

        with self.assertRaises(ValueError) as cm:
            pipeline._encode_processed_content(preprocessed_content)
        self.assertIn("number of embeddings does not match", str(cm.exception))

    @patch.object(MultilingualCLIPModelInferencePipeline, '_content_preprocessing')
    @patch.object(MultilingualCLIPModelInferencePipeline, '_encode_processed_content')
    def test_run_pipeline(self, mock_encode_processed_content, mock_content_preprocessing):
        model = MagicMock(spec=MultilingualCLIPModel)
        inference_request = MagicMock()
        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        mock_content_preprocessing.return_value = [["content"]]
        mock_encode_processed_content.return_value = [np.array([0.5, 0.5])]
        pipeline.format_results = MagicMock(return_value="formatted result")

        result = pipeline.run_pipeline()

        mock_content_preprocessing.assert_called_once()
        mock_encode_processed_content.assert_called_once()
        pipeline.format_results.assert_called_once()
        self.assertEqual(result, "formatted result")

    def test_split_prefix_preprocess_text_tokenizer_called(self):
        """
        Ensure that the tokenizer wrapper is called when preprocessing text.
        """
        tokenizer_wrapper = MagicMock(spec=MultilingualCLIPTokenizerWrapper)
        tokenizer_wrapper.tokenize.return_value = "tokenized_text"

        model = MagicMock(spec=MultilingualCLIPModel)
        model.get_preprocessor.return_value = tokenizer_wrapper

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["This is a test sentence."],
            preprocessing_config=TextPreprocessingConfig(),
            return_individual_error=False,
            model_config=MagicMock(spec=ModelConfig)
        )

        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        with patch(
                "marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline."
                "split_prefix_preprocess_text"
        ) as mock_split:
            mock_split.side_effect = lambda contents, preprocessor, config: [
                [(contents[0], preprocessor.tokenize(contents[0]))]]
            result = pipeline._content_preprocessing()

        tokenizer_wrapper.tokenize.assert_called_once_with("This is a test sentence.")
        self.assertEqual(result[0][0][1], "tokenized_text")

    def test_download_and_preprocess_image_preprocessor_called(self):
        """
        Ensure that the image preprocessor is called during image preprocessing.
        """
        image_preprocessor = MagicMock(spec=MultilingualCLIPPreprocessor)
        image_preprocessor._preprocess_image.return_value = torch.tensor([1.0])

        model = MagicMock(spec=MultilingualCLIPModel)
        model.get_preprocessor.return_value = image_preprocessor

        inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=["http://some.image/url.jpg"],
            preprocessing_config=ImagePreprocessingConfig(),
            return_individual_error=False,
            model_config=MagicMock(spec=ModelConfig)
        )

        pipeline = MultilingualCLIPModelInferencePipeline(model, inference_request)

        with patch(
                "marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline."
                "download_and_preprocess_image"
        ) as mock_download:
            mock_download.side_effect = lambda contents, preprocessor, config, return_individual_error: [
                [(contents[0], preprocessor._preprocess_image(contents[0]))]]
            result = pipeline._content_preprocessing()

        image_preprocessor._preprocess_image.assert_called_once_with("http://some.image/url.jpg")
        self.assertTrue(torch.equal(result[0][0][1], torch.tensor([1.0])))