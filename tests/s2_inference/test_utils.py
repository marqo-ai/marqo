import unittest

from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise, 
    _convert_vectorized_output, 
    available_models,
    clear_loaded_models,
    _create_model_cache_key,
    get_model_properties_from_registry
    )

from torch import FloatTensor, linalg, equal
import numpy as np
import time

class TestOutputs(unittest.TestCase):

    def test_check_output(self):
        # tests for checking the output type standardization
        list_o_list = [[1,2]]
        float_tensor = FloatTensor(list_o_list)
        numpy_array = np.array(list_o_list)
        
        assert not _check_output_type(float_tensor)
        assert not _check_output_type(numpy_array)

        assert _check_output_type(float_tensor.tolist())
        assert _check_output_type(numpy_array.tolist())
        assert _check_output_type(list_o_list)

    def test_create_model_cache_key(self):
        # test the key generating functionailty for inserting into the cache
        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "all-MiniLM-L6-v1"]
        devices = ['cpu', 'cuda', 'cuda:1']

        for name in names:
            for device in devices:
                assert _create_model_cache_key(name, get_model_properties_from_registry(name), device) == (name, device)

    def test_clear_model_cache(self):
        # tests clearing the model cache
        clear_loaded_models()
        device = 'cpu'
        assert available_models == dict()

        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "hf/all-MiniLM-L6-v1"]

        keys = []
        for name in names:
            _ = vectorise(name, 'hello', device=device)
            key = _create_model_cache_key(name, get_model_properties_from_registry(name), device)
            keys.append(key)
            
        print(sorted(set(available_models.keys())), sorted(set(keys)))
        assert sorted(set(available_models.keys())) == sorted(set(keys))

        clear_loaded_models()

        assert available_models == dict()

    def test_model_is_getting_cached(self):
        # test the model is cached on subsequent calls
        clear_loaded_models()
        
        device = 'cpu'
        assert available_models == dict()

        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "all-MiniLM-L6-v1"]

        keys = []
        for name in names:
            
            key = _create_model_cache_key(name, get_model_properties_from_registry(name), device)
            assert key not in list(available_models.keys())
            _ = vectorise(name, 'hello', device=device)
            assert key in list(available_models.keys())

        clear_loaded_models()

    def test_cache_is_quicker(self):
        # test the model is cached on subsequent calls
        clear_loaded_models()
        
        device = 'cpu'
        assert available_models == dict()

        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "all-MiniLM-L6-v1"]

        keys = []
        for name in names:
            
            key = _create_model_cache_key(name, get_model_properties_from_registry(name), device)
            assert key not in list(available_models.keys())
            t0 = time.time()
            _ = vectorise(name, 'hello', device=device)
            t1 = time.time()
            _ = vectorise(name, 'hello', device=device)
            t2 = time.time()

            assert (t1 - t0) > (t2 - t1)

        clear_loaded_models()


    def test_convert_output(self):
        
        list_o_lists = [ [[1,2], [3,4]],
                            [[1,2]]
                        ]
        for list_o_list in list_o_lists:
            float_tensor = FloatTensor(list_o_list)
            numpy_array = np.array(list_o_list)

            assert _convert_vectorized_output(list_o_list) == list_o_list
            assert _convert_vectorized_output(float_tensor) == list_o_list
            assert _convert_vectorized_output(numpy_array) == list_o_list
            
    # def test_normalize(self):
    #     list_o_lists = [ [[1,2], [3,4]],
    #                         [[1,2]], (np.random.rand(100,100)-0.5).tolist()
    #                     ]
    #     eps = 1e-6

    #     for list_o_list in list_o_lists:
    #         float_tensor = FloatTensor(list_o_list)
    #         numpy_array = np.array(list_o_list)

    #         normed_list = normalize_2d(list_o_list)
    #         normed_ft = normalize_2d(float_tensor)
    #         normed_np = normalize_2d(numpy_array)

    #         assert FloatTensor(normed_list).dim() == 2
    #         assert normed_ft.dim() == 2
    #         assert FloatTensor(normed_np).dim() == 2

    #         norm_normed_list = linalg.vector_norm(FloatTensor(normed_np), dim=1, ord=2)
    #         norm_normed_ft = linalg.vector_norm(FloatTensor(normed_ft), dim=1, ord=2)
    #         norm_normed_np = linalg.vector_norm(FloatTensor(normed_np), dim=1, ord=2)

    #         assert max(abs(norm_normed_ft - norm_normed_list)) < eps
    #         assert max(abs(norm_normed_ft - norm_normed_np)) < eps

    #         assert abs(min(norm_normed_list) - 1) < eps and abs(max(norm_normed_list) - 1) < eps
    #         assert abs(min(norm_normed_ft) - 1) < eps and abs(max(norm_normed_ft) - 1) < eps
    #         assert abs(min(norm_normed_np) - 1) < eps and abs(max(norm_normed_np) - 1) < eps