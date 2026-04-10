import unittest

from inference_orchestrator.services.media_download_and_preprocess.split_text import (
    _reconstruct_multi_list,
    _reconstruct_single_list,
    _splitting_functions,
    check_make_string_valid,
    prefix_text_chunks,
    split_text,
)


class TestSplittingFunctions(unittest.TestCase):
    """Tests for _splitting_functions."""

    def test_splitting_functions_non_string_raises_error(self):
        """Test that non-string split_by raises TypeError."""
        with self.assertRaises(TypeError) as context:
            _splitting_functions(123)
        self.assertIn("expected str", str(context.exception))

    def test_splitting_functions_invalid_type_raises_error(self):
        """Test that invalid split_by type raises KeyError."""
        with self.assertRaises(KeyError) as context:
            _splitting_functions("invalid_type")
        self.assertIn("unexpected split_by type", str(context.exception))


class TestReconstructSingleList(unittest.TestCase):
    """Tests for _reconstruct_single_list."""

    def test_reconstruct_single_list_with_default_separator(self):
        """Test reconstructing with default space separator."""
        segmented = ["hello", "world"]
        result = _reconstruct_single_list(segmented)
        self.assertEqual("hello world", result)

    def test_reconstruct_single_list_with_custom_separator(self):
        """Test reconstructing with custom separator."""
        segmented = ["hello", "world"]
        result = _reconstruct_single_list(segmented, seperator="-")
        self.assertEqual("hello-world", result)

    def test_reconstruct_single_list_with_none_values(self):
        """Test that None values are filtered out."""
        segmented = ["hello", None, "world", None]
        result = _reconstruct_single_list(segmented)
        self.assertEqual("hello world", result)


class TestReconstructMultiList(unittest.TestCase):
    """Tests for _reconstruct_multi_list."""

    def test_reconstruct_multi_list(self):
        """Test reconstructing multiple lists."""
        segmented_lists = [["hello", "world"], ["foo", "bar"]]
        result = _reconstruct_multi_list(segmented_lists)
        self.assertEqual(["hello world", "foo bar"], result)

    def test_reconstruct_multi_list_filters_empty(self):
        """Test that empty strings are filtered out."""
        segmented_lists = [["hello"], [], ["world"]]
        result = _reconstruct_multi_list(segmented_lists)
        self.assertEqual(["hello", "world"], result)


class TestCheckMakeStringValid(unittest.TestCase):
    """Tests for check_make_string_valid."""

    def test_check_make_string_valid_with_none_coerced(self):
        """Test that None is coerced to empty string."""
        result = check_make_string_valid(None, coerce=True)
        self.assertEqual(" ", result)

    def test_check_make_string_valid_with_empty_string_coerced(self):
        """Test that empty string is coerced."""
        result = check_make_string_valid("", coerce=True)
        self.assertEqual(" ", result)

    def test_check_make_string_valid_with_whitespace(self):
        """Test that whitespace-only string returns empty string."""
        result = check_make_string_valid("   ", coerce=True)
        self.assertEqual(" ", result)

    def test_check_make_string_valid_with_valid_string(self):
        """Test that valid string is returned as-is."""
        result = check_make_string_valid("hello world", coerce=True)
        self.assertEqual("hello world", result)

    def test_check_make_string_valid_non_string_raises_error(self):
        """Test that non-string type raises AttributeError (when calling .isspace())."""
        # The function tries to call .isspace() on non-strings, raising AttributeError
        with self.assertRaises(AttributeError):
            check_make_string_valid(123, coerce=True)


class TestSplitText(unittest.TestCase):
    """Tests for split_text function."""

    def test_split_text_zero_length_raises_error(self):
        """Test that split_length of 0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            split_text("test", split_length=0)
        self.assertIn("split length must be > 0", str(context.exception))

    def test_split_text_short_text_returns_as_is(self):
        """Test that very short text is returned as-is."""
        result = split_text("a", split_length=2)
        self.assertEqual(["a"], result)

    def test_split_text_with_custom_separator(self):
        """Test split_text with custom separator."""
        text = "hello world test"
        result = split_text(
            text, split_by="word", split_length=2, split_overlap=1, custom_seperator="-"
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # First chunk should use custom separator
        self.assertIn("-", result[0])


class TestPrefixTextChunks(unittest.TestCase):
    """Tests for prefix_text_chunks."""

    def test_prefix_text_chunks_with_prefix(self):
        """Test adding prefix to text chunks."""
        chunks = ["hello", "world"]
        result = prefix_text_chunks(chunks, "PREFIX: ")
        self.assertEqual(["PREFIX: hello", "PREFIX: world"], result)

    def test_prefix_text_chunks_without_prefix(self):
        """Test that empty prefix returns chunks unchanged."""
        chunks = ["hello", "world"]
        result = prefix_text_chunks(chunks, "")
        self.assertEqual(["hello", "world"], result)

    def test_prefix_text_chunks_with_none_prefix(self):
        """Test that None prefix returns chunks unchanged."""
        chunks = ["hello", "world"]
        result = prefix_text_chunks(chunks, None)
        self.assertEqual(["hello", "world"], result)


if __name__ == "__main__":
    unittest.main()
