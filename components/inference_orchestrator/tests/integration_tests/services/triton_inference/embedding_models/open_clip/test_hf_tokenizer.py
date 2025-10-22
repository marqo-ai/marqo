"""
Integration tests for HFTokenizer.

This module tests the HuggingFace tokenizer wrapper used in OpenCLIP models,
including text cleaning, tokenization, and tensor output.
"""

import unittest

import torch

from inference_orchestrator.services.triton_inference.embedding_models.open_clip.hf_tokenizer import (
    HFTokenizer,
    basic_clean,
    whitespace_clean,
)


class TestHFTokenizer(unittest.TestCase):
    """Test suite for HFTokenizer and text cleaning functions."""

    def test_whitespace_clean_single_spaces(self):
        """Test whitespace_clean with multiple spaces."""
        test_cases = [
            ("hello  world", "hello world"),
            ("hello   world", "hello world"),
            ("  hello world  ", "hello world"),
            ("hello\n\nworld", "hello world"),
            ("hello\t\tworld", "hello world"),
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                result = whitespace_clean(input_text)
                self.assertEqual(result, expected_output)

    def test_basic_clean_html_unescape(self):
        """Test basic_clean with HTML entities."""
        test_cases = [
            ("hello &amp; world", "hello & world"),
            ("hello &lt; world", "hello < world"),
            ("hello &gt; world", "hello > world"),
            ("hello &quot; world", 'hello " world'),
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                result = basic_clean(input_text)
                self.assertEqual(result, expected_output)

    def test_basic_clean_strips_whitespace(self):
        """Test basic_clean strips leading/trailing whitespace."""
        result = basic_clean("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_hf_tokenizer_initialization(self):
        """Test HFTokenizer initializes with a valid tokenizer name."""
        # Use a small, well-known tokenizer for testing
        tokenizer = HFTokenizer("bert-base-uncased")
        self.assertIsNotNone(tokenizer.tokenizer)

    def test_hf_tokenizer_single_string(self):
        """Test HFTokenizer with a single string input."""
        tokenizer = HFTokenizer("bert-base-uncased")
        text = "hello world"
        result = tokenizer(text)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 1)

    def test_hf_tokenizer_list_of_strings(self):
        """Test HFTokenizer with a list of strings."""
        tokenizer = HFTokenizer("bert-base-uncased")
        texts = ["hello world", "foo bar"]
        result = tokenizer(texts)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 2)

    def test_hf_tokenizer_cleans_text(self):
        """Test that HFTokenizer applies text cleaning."""
        tokenizer = HFTokenizer("bert-base-uncased")

        # Text with extra whitespace that should be cleaned
        text_with_spaces = "hello   world"
        result_cleaned = tokenizer(text_with_spaces)

        # The same text without extra spaces
        text_normal = "hello world"
        result_normal = tokenizer(text_normal)

        # Both should produce the same tokens after cleaning
        self.assertTrue(torch.equal(result_cleaned, result_normal))

    def test_hf_tokenizer_output_dtype(self):
        """Test that HFTokenizer returns correct tensor dtype."""
        tokenizer = HFTokenizer("bert-base-uncased")
        text = "hello world"
        result = tokenizer(text)

        # input_ids should be integer type
        self.assertTrue(result.dtype in [torch.long, torch.int64, torch.int32])

    def test_hf_tokenizer_batch_processing(self):
        """Test HFTokenizer handles batch processing correctly."""
        tokenizer = HFTokenizer("bert-base-uncased")
        texts = ["hello", "world", "foo bar baz"]
        result = tokenizer(texts)

        # All sequences should be padded to the same length
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 3)
        # All rows should have the same length due to padding
        self.assertTrue(
            all(len(result[i]) == len(result[0]) for i in range(len(result)))
        )

    def test_hf_tokenizer_empty_string(self):
        """Test HFTokenizer handles empty string."""
        tokenizer = HFTokenizer("bert-base-uncased")
        result = tokenizer("")

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)

    def test_hf_tokenizer_special_characters(self):
        """Test HFTokenizer with special characters."""
        tokenizer = HFTokenizer("bert-base-uncased")
        texts_with_special = [
            "hello! world?",
            "foo@bar.com",
            "test#123",
        ]

        for text in texts_with_special:
            with self.subTest(text=text):
                result = tokenizer(text)
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
