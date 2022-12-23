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
                model_properties = get_model_properties_from_registry(name)
                assert (
                            _create_model_cache_key(name, device, model_properties)
                            == (
                               name + "||"
                               + model_properties.get('name', '') + "||"
                               + str(model_properties.get('dimensions', '')) + "||"
                               + model_properties.get('type', '') + "||"
                               + str(model_properties.get('tokens', '')) + "||"
                               + device)
                )

    def test_clear_model_cache(self):
        # tests clearing the model cache
        clear_loaded_models()
        device = 'cpu'
        assert available_models == dict()

        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "hf/all-MiniLM-L6-v1"]

        keys = []
        for name in names:
            _ = vectorise(name, 'hello', device=device)
            key = _create_model_cache_key(name, device, get_model_properties_from_registry(name))
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

            key = _create_model_cache_key(name, device, get_model_properties_from_registry(name))
            assert key not in list(available_models.keys())
            _ = vectorise(name, 'hello', device=device)
            assert key in list(available_models.keys())

        clear_loaded_models()

    def test_cache_is_quicker(self):
        # test the model is cached on subsequent calls
        clear_loaded_models()

        device = 'cpu'
        assert available_models == dict()

        names = ["RN50", "sentence-transformers/all-MiniLM-L6-v1", "all-MiniLM-L6-v1"]

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            key = _create_model_cache_key(name, device, model_properties)
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
            
