import torch
from unittest import TestCase
from unittest.mock import patch
from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.core.inference.api import TextPreprocessingConfig, TextChunkConfig


class CLIPPreprocessor(AbstractPreprocessor):
    """A mock preprocessor for testing."""

    def preprocess(self, inputs, modality):
        return [torch.rand(size=(1, 12)) for _ in range(len(inputs))]


def preprocess_side_effect(inputs, modality="language"):
    """
    A side effect function for preprocess mock.
    Returns a list of tensors where each tensor depends on the length of the text.
    """
    return [torch.ones(size=(1, 12)), ] * len(inputs)


def faulty_preprocess_side_effect(inputs, modality="language"):
    """Return fewer tensors than the number of inputs to trigger the ValueError."""
    return [torch.ones(size=(1, 12))] * (len(inputs) - 1)  # One less output


class TestSplitPrefixPreprocessText(TestCase):

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_prefix_preprocess_text_with_prefix(self, mock_preprocess):
        """Check that the text is split and preprocessed correctly with a prefix.

        The preprocessor should take the prefixed text as input, while the chunk
        should not include the prefix.
        """
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix="this is a prefix: "
        )
        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)
        self.assertEqual(len(results), 2)
        result_1 = results[0]
        self.assertTrue(isinstance(result_1, list))
        self.assertEqual(len(result_1), 2)
        self.assertTrue(all(isinstance(t, tuple) for t in result_1))
        self.assertTrue(all(isinstance(t[0], str) for t in result_1))
        self.assertTrue(all(not t.startswith("this is a prefix: ") for t, _ in result_1))
        self.assertEqual("This is a", result_1[0][0])
        self.assertEqual("a test sentence", result_1[1][0])

        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_1))

        result_2 = results[1]
        self.assertTrue(isinstance(result_2, list))
        self.assertEqual(len(result_2), 1)
        self.assertTrue(all(isinstance(t, tuple) for t in result_2))
        self.assertTrue(all(isinstance(t[0], str) for t in result_2))
        self.assertTrue(all(not t.startswith("this is a prefix: ") for t, _ in result_2))
        self.assertEqual("Test", result_2[0][0])
        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_2))

        mock_preprocess.assert_called()
        for call in mock_preprocess.call_kwargs_list:
            args, _ = call
            for text in args[0]:
                self.assertTrue(text.startswith("this is a prefix: "))

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_prefix_preprocess_text_without_prefix(self, mock_preprocess):
        """Check the case where the text prefix is None."""
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix=None
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)
        self.assertEqual(len(results), 2)
        result_1 = results[0]
        self.assertTrue(isinstance(result_1, list))
        self.assertEqual(len(result_1), 2)
        self.assertTrue(all(isinstance(t, tuple) for t in result_1))
        self.assertTrue(all(isinstance(t[0], str) for t in result_1))
        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_1))

        result_2 = results[1]
        self.assertTrue(isinstance(result_2, list))
        self.assertEqual(len(result_2), 1)
        self.assertTrue(all(isinstance(t, tuple) for t in result_2))
        self.assertTrue(all(isinstance(t[0], str) for t in result_2))
        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_2))

        mock_preprocess.assert_called()
        for call in mock_preprocess.call_kwargs_list:
            args, _ = call
            for text in args[0]:
                self.assertFalse(text.startswith("this is a prefix"))

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_prefix_preprocess_text_no_chunking(self, mock_preprocess):
        """Check the case where the text is not chunked and has a prefix."""
        content = ["This is a test sentence", "Test"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=False,
            text_prefix="this is a prefix"
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)
        self.assertEqual(len(results), 2)
        result_1 = results[0]
        self.assertTrue(isinstance(result_1, list))
        self.assertEqual(len(result_1), 1)
        self.assertTrue(all(isinstance(t, tuple) for t in result_1))
        self.assertTrue(all(isinstance(t[0], str) for t in result_1))
        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_1))

        result_2 = results[1]
        self.assertTrue(isinstance(result_2, list))
        self.assertEqual(len(result_2), 1)
        self.assertTrue(all(isinstance(t, tuple) for t in result_2))
        self.assertTrue(all(isinstance(t[0], str) for t in result_2))
        self.assertTrue(all(isinstance(t[1], torch.Tensor) for t in result_2))

        mock_preprocess.assert_called()
        for call in mock_preprocess.call_kwargs_list:
            args, _ = call
            for text in args[0]:
                self.assertTrue(text.startswith("this is a prefix"))

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_prefix_preprocess_text_empty_content(self, mock_preprocess):
        """Check behavior when content is an empty list."""
        content = []
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix="this is a prefix"
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 0)
        mock_preprocess.assert_not_called()

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_prefix_preprocess_text_long_text_many_chunks(self, mock_preprocess):
        """Check long content chunking with small split length."""
        long_text = "word " * 100  # 100 repetitions -> 100 words
        content = [long_text]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=2, split_overlap=0),
            text_prefix="prefix: "
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        # Expecting 100/2 = 50 chunks (without overlap)
        self.assertEqual(len(results), 1)
        chunks = results[0]
        self.assertEqual(len(chunks), 50)

        for chunk in chunks:
            text, tensor = chunk
            self.assertFalse(text.startswith("prefix: "))
            self.assertEqual(text, " ".join(text.split()))
            self.assertIsInstance(tensor, torch.Tensor)

        mock_preprocess.assert_called()

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_split_length_greater_than_text_length(self, mock_preprocess):
        """Check behavior when split length is greater than text length."""
        content = ["Short text"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=10, split_overlap=0),
            text_prefix=None
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 1)
        chunks = results[0]
        # Should return one chunk since text is shorter than split_length
        self.assertEqual(len(chunks), 1)

        for text, tensor in chunks:
            self.assertIsInstance(text, str)
            self.assertIsInstance(tensor, torch.Tensor)

        mock_preprocess.assert_called()

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_empty_string_in_content(self, mock_preprocess):
        """Check behavior with an empty string in the content list."""
        content = ["", "Non-empty text"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=False,
            text_prefix="prefix: "
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 2)

        for chunk_list in results:
            for text, tensor in chunk_list:
                self.assertIsInstance(text, str)
                self.assertFalse(text.startswith("prefix: "))
                self.assertIsInstance(tensor, torch.Tensor)

        mock_preprocess.assert_called()

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def test_prefix_is_empty_string(self, mock_preprocess):
        """Check behavior when text_prefix is an empty string."""
        content = ["Hello world"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=False,
            text_prefix=""
        )

        results = split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertEqual(len(results), 1)
        chunks = results[0]

        self.assertEqual(len(chunks), 1)
        text, tensor = chunks[0]

        self.assertEqual(text, "Hello world")  # Prefix is "", so no change.
        self.assertIsInstance(tensor, torch.Tensor)

        mock_preprocess.assert_called()

    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=faulty_preprocess_side_effect)
    def test_mismatched_preprocessed_output_length(self, mock_preprocess):
        """Check that mismatched lengths raise a ValueError."""
        content = ["Test sentence"]
        preprocessor = CLIPPreprocessor()
        preprocessing_config = TextPreprocessingConfig(
            should_chunk=True,
            chunk_config=TextChunkConfig(split_method="word", split_length=3, split_overlap=1),
            text_prefix=None
        )

        with self.assertRaises(ValueError) as context:
            split_prefix_preprocess_text(content, preprocessor, preprocessing_config)

        self.assertIn("The number of preprocessed texts does not match the number of chunks",
                      str(context.exception))
        mock_preprocess.assert_called()
