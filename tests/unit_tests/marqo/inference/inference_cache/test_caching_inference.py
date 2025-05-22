from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from marqo.core.inference.api import Inference, InferenceRequest, ModelConfig, TextPreprocessingConfig, TextChunkConfig, \
    InferenceResult, InferenceErrorModel
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.s2_inference.types import Modality


class TestCachingInferenceModelCacheKey(TestCase):
    def setUp(self):
        self.caching_inference = CachingInference(delegate=Mock(spec=Inference), cache_size=10, cache_type='LRU')

    def test_model_cache_key_generates_deterministic_key_for_same_properties(self):
        """
        Two semantically identical dicts (different insertion order) should produce the same cache key.
        """
        props1 = {'dimension': 512, 'name': 'open-clip'}
        props2 = {'name': 'open-clip', 'dimension': 512}

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
        props1 = {'dimension': 512, 'name': 'open-clip'}
        props2 = {'dimension': 512, 'name': 'open-clip/variant'}

        key1 = self.caching_inference.model_cache_key(props1)
        key2 = self.caching_inference.model_cache_key(props2)

        self.assertNotEqual(key1, key2)


class TestCachingInferenceShouldSkip(TestCase):
    def setUp(self):
        self.caching_inference = CachingInference(delegate=Mock(spec=Inference), cache_size=10, cache_type='LRU')
        # Build a minimal InferenceRequest template
        self.base_request = InferenceRequest(
            contents=['a'],
            model_config=Mock(spec=ModelConfig),
            use_inference_cache=True,
            device=None,
            modality=Modality.TEXT,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False)
        )

    def test_should_skip_when_use_inference_cache_is_false(self):
        req = self.base_request.copy(update={'use_inference_cache': False})
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_skip_when_device_set(self):
        req = self.base_request.copy(update={'device': 'cpu'})
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_skip_when_non_text_modality(self):
        req = self.base_request.copy(update={'modality': Modality.IMAGE})
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_skip_when_chunking_enabled(self):
        req = self.base_request.copy(update={'preprocessing_config': TextPreprocessingConfig(
            should_chunk=True, chunk_config=TextChunkConfig(split_length=2, split_overlap=1, split_method='word'))})
        self.assertTrue(self.caching_inference.should_skip_cache(req))

    def test_should_not_skip_when_all_condition_clear(self):
        req = self.base_request
        self.assertFalse(self.caching_inference.should_skip_cache(req))


class TestCachingInferenceVectorise(TestCase):
    def setUp(self):
        self.mock_delegate = Mock(spec=Inference)
        self.ci = CachingInference(delegate=self.mock_delegate, cache_size=0, cache_type='LRU')
        # Replace cache with a mock
        self.ci.inference_cache = Mock()
        # Stub model_cache_key to a fixed key
        self.ci.model_cache_key = Mock(return_value='fixed-key')
        # Base request template
        self.base_request = InferenceRequest(
            contents=['a', 'b'],
            model_config=Mock(spec=ModelConfig),
            use_inference_cache=True,
            device=None,
            modality=Modality.TEXT,
            preprocessing_config=TextPreprocessingConfig(should_chunk=False)
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
        self.assertEqual(result.result, [[('a', arr1)], [('b', arr2)]])

    def test_vectorise_all_misses(self):
        # No contents cached
        arr1 = np.array([1])
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None, None]
        # Delegate returns embeddings for both
        delegate_result = InferenceResult(result=[[('a', arr1)], [('b', arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Delegate called with original contents
        self.mock_delegate.vectorise.assert_called_once()
        # Cache.set called for each content
        calls = [(('fixed-key', 'a', arr1),), (('fixed-key', 'b', arr2),)]
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
        delegate_result = InferenceResult(result=[[('b', arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Delegate called with ['b']
        called_request = self.mock_delegate.vectorise.call_args[0][0]
        self.assertEqual(called_request.contents, ['b'])
        # Cache.set called once for 'b'
        self.ci.inference_cache.set.assert_called_once_with('fixed-key', 'b', arr2)
        # Final result has 'a' then 'b'
        self.assertEqual(result.result, [[('a', arr1)], [('b', arr2)]])

    def test_vectorise_error_not_cached(self):
        arr2 = np.array([2])
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None, None]
        delegate_result = InferenceResult(result=[(InferenceErrorModel(error_message='fail')), [('b', arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result

        result = self.ci.vectorise(self.base_request)

        # Only valid embedding cached
        self.ci.inference_cache.set.assert_called_once_with('fixed-key', 'b', arr2)
        # Error preserved
        self.assertEqual(result.result[0], InferenceErrorModel(error_message='fail'))
        self.assertEqual(result.result[1], [('b', arr2)])

    def test_vectorise_chunking_unsupported_raises(self):
        # Cache miss
        self.ci.should_skip_cache = Mock(return_value=False)
        self.ci.inference_cache.get.side_effect = [None]
        # Delegate returns multi-chunk for 'a'
        arr1 = np.array([1])
        arr2 = np.array([2])
        delegate_result = InferenceResult(result=[[('a', arr1), ('a_part2', arr2)]])
        self.mock_delegate.vectorise.return_value = delegate_result
        req = self.base_request.copy(update={'contents': ['a']})
        with self.assertRaises(RuntimeError) as ctx:
            self.ci.vectorise(req)
        self.assertIn('does not support chunking', str(ctx.exception))