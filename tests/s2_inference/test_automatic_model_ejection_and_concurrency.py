import unittest
import unittest.mock
from unittest.mock import patch
from marqo.api.exceptions import ModelCacheManagementError
from marqo.tensor_search import tensor_search
from marqo.s2_inference.s2_inference import clear_loaded_models, vectorise, validate_model_properties, \
    _validate_model_into_device, _check_memory_threshold_for_model
from marqo.s2_inference import s2_inference
import threading, queue


def normal_vectorise_call(test_model, test_content, q):
    # Function used to threading test
    _ = vectorise(model_name=test_model, content=test_content, device="cpu")
    q.put("success")


def racing_vectorise_call(test_model, test_content, q):
    # Function used to threading test
    try:
        _ = vectorise(model_name=test_model, content=test_content, device="cpu")
        q.put(AssertionError)
    except ModelCacheManagementError as e:
        q.put(e)
        pass


class TestAutomaticModelEject(unittest.TestCase):
    def setUp(self) -> None:
        clear_loaded_models()

        self.index_name = "test_index"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except Exception:
            pass

    def test_validate_model_into_device(self):
        small_list_of_models = ['open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
                                "sentence-transformers/all-MiniLM-L6-v2",
                                "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                                'open_clip/ViT-B-16/laion2b_s34b_b88k']
        content = "Try to kill the cpu"

        with unittest.mock.patch('marqo.s2_inference.s2_inference._validate_model_into_device') as mock_method:
            for model in small_list_of_models:
                _ = vectorise(model_name=model, content=content, device="cpu")

        checked_models = [call_args[0] for call_args, call_kwargs
                          in mock_method.call_args_list]

        self.assertEqual(small_list_of_models, checked_models)

    def test_check_memory_threshold_for_model(self):
        small_list_of_models = ['open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
                                "sentence-transformers/all-MiniLM-L6-v2",
                                "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                                'open_clip/ViT-B-16/laion2b_s34b_b88k']

        content = "Try to kill the cpu"

        with unittest.mock.patch('marqo.s2_inference.s2_inference._check_memory_threshold_for_model') as mock_method:
            for model in small_list_of_models:
                _ = vectorise(model_name=model, content=content, device="cpu")

            checked_devices = [call_args[0] for call_args, call_kwargs
                               in mock_method.call_args_list]

        self.assertEqual(len(checked_devices), len(small_list_of_models))
        self.assertEqual(set(checked_devices), {"cpu"})

    def test_load_very_large_model(self):
        huge_models = ['open_clip/ViT-g-14/laion2b_s12b_b42k']
        for model in huge_models:
            try:
                _ = vectorise(model_name=model, content='this is a huge model', device="cpu")
                raise AssertionError
            except ModelCacheManagementError as e:
                assert "CANNOT find enough space" in e.message

    def test_get_model_size(self):
        models_and_sizes = {
            "open_clip/ViT-L-14/openai": 1.5,
            'open_clip/ViT-L-14/laion400m_e31': 1.5,
            'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k': 1,
            "sentence-transformers/all-MiniLM-L6-v2": 0.7,
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base": 0.7,
            'open_clip/ViT-B-16/laion2b_s34b_b88k': 1,
            'open_clip/coca_ViT-L-14/laion2b_s13b_b90k': 1.5,
            'open_clip/RN50x64/openai': 1,
            "onnx16/open_clip/ViT-B-32/laion2b_e16": 1,
        }

        for model_name, size in models_and_sizes.items():
            self.assertEqual(s2_inference.get_model_size(model_name, validate_model_properties(model_name, None)),
                             size, msg=model_name)

        generic_model = {
            "model_name": "my_custom_clip",
            "model_properties_1": {
                "name": "ViT-L-14",
                "type": "open_clip",
                "dimensions": 768,
                "model_size": 1.53,
            },
            "model_properties_2": {
                "name": "ViT-L/14",
                "dimensions": 768,
                "type": "clip",
            }
        }

        self.assertEqual(s2_inference.get_model_size(generic_model["model_name"], generic_model["model_properties_1"]),
                         1.53)
        self.assertEqual(s2_inference.get_model_size(generic_model["model_name"], generic_model["model_properties_2"]),
                         1.5)

    def test_model_management(self):
        # Instance should be out of memory without model management
        content = "Try to kill the cpu"
        # These models are tested in encoding test to avoid downloading again.
        list_of_models = ["fp16/ViT-B/32", "open_clip/convnext_base_w/laion2b_s13b_b82k",
                          "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg",
                          "onnx16/open_clip/ViT-B-32/laion400m_e32",
                          'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                          "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1",
                          "hf/all_datasets_v4_MiniLM-L6",
                          "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        for model in list_of_models:
            _ = vectorise(model_name=model, content=content, device="cpu")

    def test_thread_safe_function_call(self):
        model_name = "ViT-B/32"
        validated_model_properties = validate_model_properties(model_name, None)
        device = "cpu"
        model_size = 1

        _validate_model_into_device(model_name, validated_model_properties, device,
                                    calling_func="_update_available_models")
        _validate_model_into_device(model_name, validated_model_properties, device, calling_func="unit_test")

        try:
            _validate_model_into_device(model_name, validated_model_properties, device)
            raise AssertionError
        except RuntimeError:
            pass

        try:
            _validate_model_into_device(model_name, validated_model_properties, device, calling_func="void")
            raise AssertionError
        except RuntimeError:
            pass

        _check_memory_threshold_for_model(device, model_size, calling_func="_validate_model_into_device")
        _check_memory_threshold_for_model(device, model_size, calling_func="unit_test")

        try:
            _check_memory_threshold_for_model(device, model_size)
            raise AssertionError
        except RuntimeError:
            pass

        try:
            _check_memory_threshold_for_model(device, model_size, calling_func="void")
            raise AssertionError
        except RuntimeError:
            pass

    def test_concurrent_vectorise_call_no_cache(self):
        # To test error is thrown if multiple threads want to load the model
        clear_loaded_models()
        test_content = "this is a test"
        test_model = "ViT-B/32"

        threads = []
        q_1 = queue.Queue()
        q_2 = queue.Queue()
        t = threading.Thread(target=normal_vectorise_call, args=(test_model, test_content, q_1))
        threads.append(t)
        t.start()

        num_of_threads = 3
        for i in range(num_of_threads):
            t = threading.Thread(target=racing_vectorise_call, args=(test_model, test_content, q_2))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert q_1.qsize() == 1
        assert q_1.get() == "success"

        assert q_2.qsize() == num_of_threads
        while not q_2.empty():
            assert type(q_2.get()) is type(ModelCacheManagementError(""))

    def test_concurrent_vectorise_call_cached(self):
        # To test error is thrown if multiple threads want to load the model
        clear_loaded_models()
        test_content = "this is a test"
        test_model = "ViT-B/32"

        threads = []
        q_1 = queue.Queue()
        t = threading.Thread(target=normal_vectorise_call, args=(test_model, test_content, q_1))
        t.start()
        t.join()
        # model is loaded into cache
        assert q_1.qsize() == 1
        assert q_1.get() == "success"

        q_2 = queue.Queue()
        num_of_threads = 3
        for i in range(num_of_threads):
            t = threading.Thread(target=normal_vectorise_call, args=(test_model, test_content, q_2))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert q_2.qsize() == num_of_threads
        while not q_2.empty():
            assert q_2.get() == "success"

    def test_concurrent_model_loading_and_vectorise(self):
        clear_loaded_models()
        test_content = "this is a test"
        test_model_1 = "ViT-B/32"
        test_model_2 = "sentence-transformers/all-MiniLM-L6-v2"

        q_1 = queue.Queue()
        t = threading.Thread(target=normal_vectorise_call, args=(test_model_1, test_content, q_1))
        t.start()
        t.join() # load test_model_1 into cache

        # thread 1 needs to load a new model into cache
        t_1 = threading.Thread(target=normal_vectorise_call, args=(test_model_2, test_content, q_1))
        # thread 2 uses the model in cache to vectorise
        t_2 = threading.Thread(target=normal_vectorise_call, args=(test_model_1, test_content, q_1))
        t_1.start()
        t_2.start()

        t_1.join()
        t_2.join()

        assert q_1.qsize() == 3
        while not q_1.empty():
            assert q_1.get() == "success"

