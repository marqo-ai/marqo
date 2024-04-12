import os
import random
import unittest.mock
from unittest.mock import patch

class TestVectoriseInferenceCache(unittest.TestCase):

    def _import_vectorise_with_inference_cache(self, cache_size:int = 50, cache_type = "LRU"):
        """Import the vectorise function with the specified cache size and type."""
        os.environ["MARQO_INFERENCE_CACHE_TYPE"] = cache_type
        os.environ["MARQO_INFERENCE_CACHE_SIZE"] = str(cache_size)
        from marqo.s2_inference.s2_inference import vectorise
        return vectorise

    def test_vectorise_withInferenceCacheForSingleString_success(self):
        """Test if the vectorise function returns the correct embeddings for a single string."""
        vectorise = self._import_vectorise_with_inference_cache()
        # First call
        original_vector = vectorise(model_name="random/small", content="test", device="cpu", enable_cache=True)
        # following calls
        for _ in range(10):
            with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
                cached_vector = vectorise(model_name="random/small", content="test", device="cpu", enable_cache=True)
                mock_encode.assert_not_called()
                self.assertEqual(original_vector, cached_vector)

    def test_vectorise_withInferenceCacheForListString_success(self):
        """Test if the vectorise function returns the correct embeddings for a list of strings."""
        vectorise = self._import_vectorise_with_inference_cache()
        content = ["test1", "test2"]
        # First call
        original_vector = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
        # following calls
        for _ in range(10):
            with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
                cached_vector = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
                mock_encode.assert_not_called()
                self.assertEqual(original_vector, cached_vector)

    def test_vectorise_enableCaseIsFalse(self):
        """Test if the vectorise function returns the correct embeddings when enable_cache is False."""
        vectorise = self._import_vectorise_with_inference_cache()
        for first_call_enable_cache in [True, False]:
            with self.subTest(f"First call enable_cache={first_call_enable_cache}"):
                # First call
                _ = vectorise(model_name="random/small", content="test", device="cpu",
                              enable_cache=first_call_enable_cache)
                # following calls
                with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
                    _ = vectorise(model_name="random/small", content="test", device="cpu", enable_cache=False)
                    mock_encode.assert_called_once()

    def test_vectorise_listOfStringsPartialCache(self):
        """A test to check if the cache is working correctly when part of the list is already cached."""
        vectorise = self._import_vectorise_with_inference_cache()
        cached_content = ["test1", "test2"]
        # First call
        original_vector = vectorise(model_name="random/small", content=cached_content, device="cpu", enable_cache=True)

        # following calls
        new_content = ["test3", "test4"]
        with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
            _ = vectorise(model_name="random/small", content=cached_content + new_content, device="cpu",
                                      enable_cache=True)
            args, _ = mock_encode.call_args
            self.assertEqual(new_content, args[1])

    def test_vectorise_listOfStringsPartialCacheVectorsCorrect(self):
        """A test to check if the cache is working correctly when part of the list is already cached."""
        iterations = 10
        initial_size = 20
        new_cached_size = 20
        total_size = initial_size + new_cached_size
        for _ in range(iterations):
            vectorise = self._import_vectorise_with_inference_cache(cache_size=total_size)
            cached_content = [f"test{i}" for i in range(initial_size)]
            # First call
            original_vector = vectorise(model_name="random/small", content=cached_content, device="cpu", enable_cache=True)

            # following calls with partially cached content
            new_content = [f"test{i}" for i in range(initial_size, total_size)]
            content = cached_content + new_content
            random.shuffle(content)
            vectors = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
            self.assertEqual(len(vectors), total_size)
            self.assertEqual([vectors[content.index(c)] for c in cached_content], original_vector)

            # following calls with fully cached content
            random.shuffle(content)
            with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
                _ = vectorise(model_name="random/small", content=content, device="cpu",
                                          enable_cache=True)
                mock_encode.assert_not_called()

    def test_vectorise_cacheNotWorkForPILImage(self):
        """Test if the cache does not work for PIL.Image.Image objects."""
