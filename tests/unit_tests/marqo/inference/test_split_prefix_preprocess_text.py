import torch
from unittest import TestCase
from unittest.mock import patch
from marqo.inference.chunk_download_preprocess_content import _split_prefix_preprocess_text
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.core.inference.api import TextPreprocessingConfig, TextChunkConfig


class CLIPPreprocessor(AbstractPreprocessor):

    @staticmethod
    def preprocess(content, text_modality):
        return [torch.rand(size=(1, 12)) for _ in range(len(content))]

class TestSplitPrefixPreprocessText(TestCase):

    @patch.object(CLIPPreprocessor, 'preprocess', return_value=[torch.rand(size=(1, 12))])
    def test_split_prefix_preprocess_text_with_prefix(self, mock_preprocess):
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix="this is a prefix"
        )

        results = _split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(t, tuple) for t in results))
        self.assertTrue(all(isinstance(t[0], torch.Tensor) for t in results))
        mock_preprocess.assert_called()
        for call in mock_preprocess.call_args_list:
            args, _ = call
            for text in args[0]:
                self.assertTrue(text.startswith("this is a prefix"))

    @patch.object(CLIPPreprocessor, 'preprocess', return_value=[torch.rand(size=(1, 12))])
    def test_split_prefix_preprocess_text_without_prefix(self, mock_preprocess):
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix=None
        )

        results = _split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(t, tuple) for t in results))
        self.assertTrue(all(isinstance(t[0], torch.Tensor) for t in results))
        mock_preprocess.assert_called()
        for call in mock_preprocess.call_args_list:
            args, _ = call
            for text in args[0]:
                self.assertFalse(text.startswith("this is a prefix"))

    @patch.object(CLIPPreprocessor, 'preprocess', return_value=[torch.rand(size=(1, 12))])
    def test_split_prefix_preprocess_text_no_chunking(self, mock_preprocess):
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=False,
            text_prefix="this is a prefix"
        )

        results = _split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 1)
        self.assertTrue(all(isinstance(t, tuple) for t in results))
        self.assertTrue(all(isinstance(t[0], torch.Tensor) for t in results))
        mock_preprocess.assert_called()
        for call in mock_preprocess.call_args_list:
            args, _ = call
            for text in args[0]:
                self.assertTrue(text.startswith("this is a prefix"))