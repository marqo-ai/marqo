import unittest

from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise, 
    _convert_vectorized_output, 
    available_models,
    clear_loaded_models,
    )

from torch import FloatTensor, linalg, equal
import numpy as np

class TestOutputs(unittest.TestCase):

    def setUp(self) -> None:

        pass

        
    def test_check_output(self):

        list_o_list = [[1,2]]
        float_tensor = FloatTensor(list_o_list)
        numpy_array = np.array(list_o_list)
        
        assert not _check_output_type(float_tensor)
        assert not _check_output_type(numpy_array)

        assert _check_output_type(float_tensor.tolist())
        assert _check_output_type(numpy_array.tolist())
        assert _check_output_type(list_o_list)

    def test_clear_model_cache(self):

        clear_loaded_models()
        device = 'cpu'
        assert available_models == dict()

        names = ['RN50', "sentence-transformers/all-MiniLM-L6-v1", "all-MiniLM-L6-v1"]

        keys = []
        for name in names:
            _ = vectorise(name, 'hello', device=device)
            key = (name, device)
            keys.append(key)
            print(key)
        print(sorted(set(available_models.keys())), sorted(set(keys)))
        assert sorted(set(available_models.keys())) == sorted(set(keys))

        clear_loaded_models()

        assert available_models == dict()

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