from unittest import TestCase
from unittest.mock import Mock

import blake3
import numpy as np

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    Inference,
    InferenceErrorModel,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextChunkConfig,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.inference_cache.caching_inference import (
    CachingInference,
)


class TestCachingInferenceModelCacheKey(TestCase):
    def setUp(self):
        self.caching_inference = CachingInference(
            delegate=Mock(spec=Inference), cache_size=10, cache_type="LRU"
        )

    def test_model_cache_key_generates_deterministic_key_for_same_properties(self):
        """
        Two semantically identical dicts (different insertion order) should produce the same cache key.
        """
        props1 = {"dimension": 512, "name": "open-clip"}
        props2 = {"name": "open-clip", "dimension": 512}

        key1 = self.caching_inference.model_cache_key(props1)
        key2 = self.caching_inference.model_cache_key(props2)

        # The MD5 digest should be identical for the same content regardless of key order
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 32)
        self.assertEqual(key1, key2)

    def test_model_cache_key_generates_different_key_for_different_properties(self):
        """
        Two dicts differing by at least one value/property should produce different cache keys.
        """
        props1 = {"dimension": 512, "name": "open-clip"}
        props2 = {"dimension": 512, "name": "open-clip/variant"}

        key1 = self.caching_inference.model_cache_key(props1)
        key2 = self.caching_inference.model_cache_key(props2)

        self.assertNotEqual(key1, key2)


class TestCachingInferenceShouldSkip(TestCase):
    def setUp(self):
        self.caching_inference = CachingInference(
            delegate=Mock(spec=Inference), cache_size=10, cache_type="LRU"
        )
        # Build a minimal InferenceRequest template
        mock_model_config = Mock(spec=EmbeddingModelConfig)
        mock_model_config.model_properties = {}
        self.base_request = InferenceRequest(
            contents=["a"],
            embedding_model_config=mock_model_config,
            use_inference_cache=True,
            modality=Modality.TEXT,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
        )

    def test_should_skip_when_use_inference_cache_is_false(self):
        req = self.base_request.model_copy(update={"use_inference_cache": False})
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_skip_when_non_text_image_modality(self):
        for modality in [Modality.VIDEO, Modality.AUDIO]:
            with self.subTest(modality=modality):
                req = self.base_request.model_copy(update={"modality": modality})
                self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_skip_when_chunking_enabled(self):
        req = self.base_request.model_copy(
            update={
                "preprocessing_config": TextPreprocessingConfig(
                    should_chunk=True,
                    chunk_config=TextChunkConfig(
                        split_length=2, split_overlap=1, split_method="word"
                    ),
                )
            }
        )
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_not_skip_when_all_condition_clear(self):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            req = self.base_request.model_copy(update={"modality": modality})
            self.assertFalse(self.caching_inference.should_skip_cache(req))


class TestCachingInferenceVectorise(TestCase):
    def setUp(self):
        self.mock_delegate = Mock(spec=Inference)
        self.ci = CachingInference(
            delegate=self.mock_delegate, cache_size=10, cache_type="LRU"
        )
        # Replace cache with a mock
        self.ci.inference_cache = Mock()
        # Stub model_cache_key to a fixed key
        self.ci.model_cache_key = Mock(return_value="fixed-key")
        # Base request template
        mock_model_config = Mock(spec=EmbeddingModelConfig)
        mock_model_config.model_properties = {}
        self.base_request = InferenceRequest(
            contents=["a", "b"],
            embedding_model_config=mock_model_config,
            use_inference_cache=True,
            modality=Modality.TEXT,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
        )

    def test_vectorise_bypass_cache_when_should_skip_cache_is_true(self):
        # Bypass cache
        self.ci.should_skip_cache = Mock(return_value=True)
        req = self.base_request
        self.ci.vectorise(req)
        # Should call delegate only
        self.mock_delegate.vectorise.assert_called_once_with(req)
        self.ci.inference_cache.get.assert_not_called()
        self.ci.inference_cache.set.assert_not_called()

    def test_vectorise_all_cached(self):
        # All contents cached
        arr1 = np.array([1])
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [arr1, arr2]
        req = self.base_request
        result = self.ci.vectorise(req)
        # Delegate should not be called
        self.mock_delegate.vectorise.assert_not_called()
        # Result should contain both cached embeddings
        self.assertEqual(result.result, [[("a", arr1)], [("b", arr2)]])

    def test_vectorise_all_misses(self):
        # No contents cached
        arr1 = np.array([1])
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None, None]
        # Delegate returns embeddings for both
        delegate_result = InferenceResult(result=[[("a", arr1)], [("b", arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Delegate called with original contents
        self.mock_delegate.vectorise.assert_called_once()
        # Cache.set called for each content
        calls = [(("fixed-key", "a", arr1),), (("fixed-key", "b", arr2),)]
        self.assertEqual(self.ci.inference_cache.set.call_count, 2)
        self.assertEqual(self.ci.inference_cache.set.call_args_list, calls)
        # Result matches delegate
        self.assertIs(result, delegate_result)

    def test_vectorise_partial_hits_and_misses(self):
        # First cached, second miss
        arr1 = np.array([1])
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [arr1, None]
        # Delegate returns for ['b'] only
        delegate_result = InferenceResult(result=[[("b", arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Delegate called with ['b']
        called_request = self.mock_delegate.vectorise.call_args[0][0]
        self.assertEqual(called_request.contents, ["b"])
        # Cache.set called once for 'b'
        self.ci.inference_cache.set.assert_called_once_with("fixed-key", "b", arr2)
        # Final result has 'a' then 'b'
        self.assertEqual(result.result, [[("a", arr1)], [("b", arr2)]])

    def test_vectorise_error_not_cached(self):
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None, None]
        delegate_result = InferenceResult(
            result=[(InferenceErrorModel(error_message="fail")), [("b", arr2)]]
        )
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Only valid embedding cached
        self.ci.inference_cache.set.assert_called_once_with("fixed-key", "b", arr2)
        # Error preserved
        self.assertEqual(result.result[0], InferenceErrorModel(error_message="fail"))
        self.assertEqual(result.result[1], [("b", arr2)])

    def test_vectorise_chunking_unsupported_raises(self):
        # Cache miss
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None]
        # Delegate returns multi-chunk for 'a'
        arr1 = np.array([1])
        arr2 = np.array([2])
        delegate_result = InferenceResult(result=[[("a", arr1), ("a_part2", arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result
        req = self.base_request.model_copy(update={"contents": ["a"]})
        with self.assertRaises(RuntimeError) as ctx:
            self.ci.vectorise(req)
        self.assertIn("does not support chunking", str(ctx.exception))


class TestCachingInferenceBase64Images(TestCase):
    def setUp(self):
        self.mock_delegate = Mock(spec=Inference)
        self.caching_inference = CachingInference(
            delegate=self.mock_delegate, cache_size=10, cache_type="LRU"
        )
        # Replace cache with a mock for better control
        self.caching_inference.inference_cache = Mock()
        # Stub model_cache_key to a fixed key
        self.caching_inference.model_cache_key = Mock(return_value="fixed-key")

        self.base64_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        self.base64_jpeg = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        self.url_image = "https://example.com/image.jpg"

        # Base64 image request template
        mock_model_config = Mock(spec=EmbeddingModelConfig)
        mock_model_config.model_properties = {}
        self.base_request = InferenceRequest(
            contents=[self.base64_png],
            embedding_model_config=mock_model_config,
            use_inference_cache=True,
            device=None,
            modality=Modality.IMAGE,
            preprocessing_config=ImagePreprocessingConfig(should_chunk=False),
        )

    def test_vectorise_caches_base64_images_only(self):
        """Test that only base64 images are cached, URLs are processed normally."""
        # Cache miss for base64, no cache check for URL
        self.caching_inference.inference_cache.get.return_value = None

        # Mock delegate response
        embedding1 = np.array([1.0, 2.0])
        embedding2 = np.array([3.0, 4.0])
        delegate_result = InferenceResult(
            result=[[(self.base64_png, embedding1)], [(self.url_image, embedding2)]]
        )
        self.mock_delegate.vectorise.return_value = delegate_result

        mixed_req = self.base_request.model_copy(
            update={"contents": [self.base64_png, self.url_image]}
        )
        result = self.caching_inference.vectorise(mixed_req)

        # Verify cache operations
        expected_hash = blake3.blake3(self.base64_png.encode()).hexdigest()
        expected_cache_key = f"blake3:{expected_hash}"
        model_key = "fixed-key"  # Mocked in setUp

        # Should check cache only for base64 image
        self.caching_inference.inference_cache.get.assert_called_once_with(
            model_key, expected_cache_key
        )

        # Should cache only base64 image
        self.caching_inference.inference_cache.set.assert_called_once_with(
            model_key, expected_cache_key, embedding1
        )

        # Delegate should be called with both contents
        called_request = self.mock_delegate.vectorise.call_args[0][0]
        self.assertEqual(called_request.contents, [self.base64_png, self.url_image])

        # Result should contain original base64 content and URL
        self.assertEqual(
            result.result[0][0], (self.base64_png, embedding1)
        )  # Original base64 returned
        self.assertEqual(
            result.result[1][0], (self.url_image, embedding2)
        )  # URL unchanged

    def test_vectorise_mixed_cache_hits_and_misses(self):
        """Test mixed scenario: base64 cache hit, URL processed normally."""
        # Cache hit for base64
        cached_embedding = np.array([1.0, 2.0])
        self.caching_inference.inference_cache.get.return_value = cached_embedding

        # Mock delegate response for URL only
        url_embedding = np.array([3.0, 4.0])
        delegate_result = InferenceResult(result=[[(self.url_image, url_embedding)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        mixed_req = self.base_request.model_copy(
            update={"contents": [self.base64_png, self.url_image]}
        )
        result = self.caching_inference.vectorise(mixed_req)

        # Should call delegate with only URL (base64 was cached)
        called_request = self.mock_delegate.vectorise.call_args[0][0]
        self.assertEqual(called_request.contents, [self.url_image])

        # Should not cache URL image
        self.caching_inference.inference_cache.set.assert_not_called()

        # Final result should have original base64 for base64, original URL for URL image
        self.assertEqual(
            result.result,
            [[(self.base64_png, cached_embedding)], [(self.url_image, url_embedding)]],
        )

    def test_vectorise_multiple_base64_images(self):
        """Test that multiple base64 images are all cached."""
        # Cache miss for both
        self.caching_inference.inference_cache.get.return_value = None

        # Mock delegate response
        embedding1 = np.array([1.0, 2.0])
        embedding2 = np.array([3.0, 4.0])
        delegate_result = InferenceResult(
            result=[[(self.base64_png, embedding1)], [(self.base64_jpeg, embedding2)]]
        )
        self.mock_delegate.vectorise.return_value = delegate_result

        multi_base64_req = self.base_request.model_copy(
            update={"contents": [self.base64_png, self.base64_jpeg]}
        )
        result = self.caching_inference.vectorise(multi_base64_req)

        # Should cache both images
        expected_hash1 = blake3.blake3(self.base64_png.encode()).hexdigest()
        expected_hash2 = blake3.blake3(self.base64_jpeg.encode()).hexdigest()
        expected_key1 = f"blake3:{expected_hash1}"
        expected_key2 = f"blake3:{expected_hash2}"
        model_key = "fixed-key"

        self.assertEqual(self.caching_inference.inference_cache.set.call_count, 2)
        set_calls = self.caching_inference.inference_cache.set.call_args_list
        self.assertIn(((model_key, expected_key1, embedding1),), set_calls)
        self.assertIn(((model_key, expected_key2, embedding2),), set_calls)

        # Results should contain original base64 content
        self.assertEqual(
            result.result[0][0], (self.base64_png, embedding1)
        )  # Original PNG base64
        self.assertEqual(
            result.result[1][0], (self.base64_jpeg, embedding2)
        )  # Original JPEG base64
