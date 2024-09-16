import importlib
import os
import random
import sys
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from PIL import Image

from marqo.s2_inference.s2_inference import get_marqo_inference_cache, clear_marqo_inference_cache, clear_loaded_models
from tests.marqo_test import TestImageUrls


class TestVectoriseInferenceCache(unittest.TestCase):

    def tearDown(self):
        clear_marqo_inference_cache()
        clear_loaded_models()
        # Remove the specific environment variables and loaded modules
        if 'MARQO_INFERENCE_CACHE_TYPE' in os.environ:
            del os.environ['MARQO_INFERENCE_CACHE_TYPE']
        if 'MARQO_INFERENCE_CACHE_SIZE' in os.environ:
            del os.environ['MARQO_INFERENCE_CACHE_SIZE']
        if 'marqo.s2_inference.s2_inference' in sys.modules:
            importlib.reload(sys.modules['marqo.s2_inference.s2_inference'])
        if "marqo.s2_inference" in sys.modules:
            importlib.reload(sys.modules['marqo.s2_inference'])

    def _import_vectorise_with_inference_cache(self, cache_size: int = 50, cache_type="LRU"):
        """Import the vectorise function with the specified cache size and type."""
        os.environ["MARQO_INFERENCE_CACHE_TYPE"] = cache_type
        os.environ["MARQO_INFERENCE_CACHE_SIZE"] = str(cache_size)
        # Assuming the module has already been potentially imported, reload it:
        if 'marqo.s2_inference.s2_inference' in sys.modules:
            importlib.reload(sys.modules['marqo.s2_inference.s2_inference'])
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
            original_vector = vectorise(model_name="random/small", content=cached_content, device="cpu",
                                        enable_cache=True)

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
        vectorise = self._import_vectorise_with_inference_cache()
        content = [Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)), ]
        # First call
        _ = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
        # following calls
        with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
            _ = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
            mock_encode.assert_called_once()

    def test_vectorise_cacheWorkForImagePath(self):
        """Test if the cache works for image paths."""
        vectorise = self._import_vectorise_with_inference_cache()
        content = [TestImageUrls.IMAGE1.value]
        # First call
        original_vector = vectorise(model_name="open_clip/ViT-B-32/laion2b_s34b_b79k", content=content,
                                    device="cpu", enable_cache=True, infer=True)
        # following calls
        with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
            _ = vectorise(model_name="open_clip/ViT-B-32/laion2b_s34b_b79k", content=content,
                          device="cpu", enable_cache=True, infer=True)
            mock_encode.assert_not_called()
        cached_vector = vectorise(model_name="open_clip/ViT-B-32/laion2b_s34b_b79k", content=content,
                                  device="cpu", enable_cache=True, infer=True)
        self.assertEqual(original_vector, cached_vector)

    def test_vectorise_cacheDifferentModelsSameContent(self):
        """Test if the cache works for different models with the same content."""
        vectorise = self._import_vectorise_with_inference_cache()
        content = "test"
        # First call
        original_vector = vectorise(model_name="random/small", content=content, device="cpu", enable_cache=True)
        # following calls
        with patch("marqo.s2_inference.s2_inference._encode_without_cache") as mock_encode:
            cached_vector = vectorise(model_name="random/large", content=content, device="cpu", enable_cache=True)
            mock_encode.assert_called_once()

    def test_vectorise_cacheConcurrentSafety(self):
        """Test if the cache works concurrently."""
        vectorise = self._import_vectorise_with_inference_cache()
        ITERATIONS = 50_000
        FREQUENT_ACCESS_RATIO = 0.5
        FREQUENT_ACCESS_SUBSET_SIZE = 5000
        TOTAL_QUERY_SET_SIZE = 1_000_000

        # Have a warm-up call to ensure the model is loaded
        _ = vectorise(model_name="random/small", content="test", device="cpu", enable_cache=True)

        def call_vectorise():
            if random.random() < FREQUENT_ACCESS_RATIO:
                text = random.sample(frequent_texts, np.random.randint(1, 30))
            else:
                text = random.sample(frequent_texts, np.random.randint(1, 30))
            return vectorise(model_name="random/small", content=text, device="cpu", enable_cache=True)

        texts = [f"text{i} " * 5 for i in range(TOTAL_QUERY_SET_SIZE)]
        frequent_texts = random.sample(texts, FREQUENT_ACCESS_SUBSET_SIZE)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call_vectorise) for _ in
                       range(ITERATIONS)]
            result = [future.result() for future in futures]
        self.assertEqual(ITERATIONS, len(result))

    def test_inferenceCacheOrder_preserved_successfully(self):
        """Test if the order of embeddings is preserved when part of the content is already cached."""
        NUMBER_OF_ITERATIONS = 100

        for _ in range(NUMBER_OF_ITERATIONS):
            vectorise = self._import_vectorise_with_inference_cache(cache_size=20)
            list_of_contents = [f"test{i}" for i in range(20)]
            intended_cache_size = random.randint(0, 20)
            cached_items = random.sample(list_of_contents, intended_cache_size)

            # First call
            original_vectors = vectorise(model_name="random/small",
                                         content=cached_items, device="cpu", enable_cache=True)

            cache = get_marqo_inference_cache()
            self.assertEqual(cache.currsize, intended_cache_size, "Cache size is not as expected.")

            cached_vectors_mapping = {content: vector for content, vector in zip(cached_items, original_vectors)}

            # Second call with partially cached content
            vectors = vectorise(model_name="random/small",
                                content=list_of_contents, device="cpu", enable_cache=True)

            all_vectors_mapping = {content: vector for content, vector in zip(list_of_contents, vectors)}

            for content in cached_items:
                self.assertEqual(cached_vectors_mapping[content], all_vectors_mapping[content],
                                 f"Order of cached content {content} is not preserved.")