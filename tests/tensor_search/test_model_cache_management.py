import torch.cuda
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.s2_inference import _validate_model_properties,\
    _create_model_cache_key, _update_available_models, available_models, clear_loaded_models
from marqo.tensor_search.tensor_search import eject_model, get_cuda_info, get_loaded_models, get_cpu_info
from marqo.errors import ModelNotInCache, HardwareCompatabilityError
import psutil



def load_model(model_name: str, device: str) -> None:
    validated_model_properties = _validate_model_properties(model_name, None)
    model_cache_key = _create_model_cache_key(model_name, device, validated_model_properties)
    _update_available_models(model_cache_key, model_name, validated_model_properties, device, True)


class TestModelCacheManagement(MarqoTestCase):

    def setUp(self) -> None:
        # We pre-define 3 dummy models for testing purpose
        self.MODEL_1 = "ViT-B/32"
        self.MODEL_2 = "hf/all-MiniLM-L6-v2"
        self.MODEL_LIST = [self.MODEL_1, self.MODEL_2]
        self.CUDA_FLAG = torch.cuda.is_available()


        # load several models into cache for setting up
        for model_name in self.MODEL_LIST:
            load_model(model_name, "cpu")

        if self.CUDA_FLAG:
            for model_name in self.MODEL_LIST:
                load_model(model_name, "cuda")


        # We loaded 6 models (3 in cuda, 3 in cpu) as initial setup
        if self.CUDA_FLAG:
            assert len(available_models) >= 4
        else:
            assert len(available_models) >= 2

    def tearDown(self) -> None:
        clear_loaded_models()



    def test_eject_model_cpu(self):
        for model_name in self.MODEL_LIST:
            eject_model(model_name, "cpu")
            if (model_name, "cpu") in available_models:
                raise AssertionError


        my_test_model_1 = "test-model-1"
        my_test_model_2 = "test-model-2"

        try:
            eject_model(my_test_model_1, "cpu")
            raise AssertionError
        except ModelNotInCache:
            pass

        try:
            eject_model(my_test_model_2, "cpu")
            raise AssertionError
        except ModelNotInCache:
            pass


    def test_eject_model_cuda(self):
        if self.CUDA_FLAG:
        # check if we can eject the models
            for model_name in self.MODEL_LIST:
                eject_model(model_name,"cuda")
                if (model_name, "cuda") in available_models:
                    raise AssertionError
            my_test_model_1 = "test-model-1"
            my_test_model_2 = "test-model-2"

            try:
                eject_model(my_test_model_1, "cuda")
            except ModelNotInCache:
                pass

            try:
                eject_model(my_test_model_2, "cuda")
            except ModelNotInCache:
                pass
        else:
            pass


    def test_cuda_info(self):
        if self.CUDA_FLAG is True:
            res = get_cuda_info()
            if "cuda_devices" not in res:
                raise AssertionError
        else:
            try:
                get_cuda_info()
            except HardwareCompatabilityError:
                pass


    def test_get_cpu_info(self) -> None:
        res = get_cpu_info()

        if "cpu_usage_percent" not in res:
            raise AssertionError

        if "memory_used_percent" not in res:
            raise AssertionError

        if "memory_used_gb" not in res:
            raise AssertionError


    def test_loaded_models(self):

        loaded_models = get_loaded_models()["models"]
        loaded_models_keys = [_create_model_cache_key(dic["model_name"], dic["model_device"],
                                _validate_model_properties(dic["model_name"], None)) for dic in loaded_models]
        assert loaded_models_keys==list(available_models.keys())


    def test_edge_case_cuda(self):
        if self.CUDA_FLAG:
            test_iterations = 10
            # Note this is a time consuming test.

            for i in range(test_iterations):
                eject_model(self.MODEL_1, "cuda")
                load_model(self.MODEL_1, "cuda")

                for _device_id in range(torch.cuda.device_count()):
                    # cuda usage
                    assert torch.cuda.memory_allocated(_device_id) < torch.cuda.get_device_properties(_device_id).total_memory
                # cpu usage
                assert psutil.cpu_percent(1) < 100.0
                # memory usage
                assert psutil.virtual_memory()[2]< 100.0
        else:
            pass


    def test_edge_case_cpu(self):
        test_iterations = 10
        # Note this is a time consuming test.

        for i in range(test_iterations):
            eject_model(self.MODEL_1, "cpu")
            load_model(self.MODEL_1, "cpu")
            # cpu usage
            assert psutil.cpu_percent(1) < 100.0
            # memory usage
            assert psutil.virtual_memory()[2]< 100.0


    def test_overall_eject_and_load_model(self):
        clear_loaded_models()
        if len(available_models) != 0:
            raise AssertionError

        for model_name in self.MODEL_LIST:
            validated_model_properties = _validate_model_properties(model_name, None)
            model_cache_key = _create_model_cache_key(model_name, "cpu", validated_model_properties)
            _update_available_models(model_cache_key, model_name, validated_model_properties, "cpu", True)

            if model_cache_key not in available_models:
                raise AssertionError

            res = get_loaded_models()["models"]
            assert res[model_name] == "cpu"

            eject_model(model_name, "cpu")

            if model_cache_key in available_models:
                raise AssertionError

        if self.CUDA_FLAG is True:
            for model_name in self.MODEL_LIST:
                validated_model_properties = _validate_model_properties(model_name, None)
                model_cache_key = _create_model_cache_key(model_name, "cuda", validated_model_properties)
                _update_available_models(model_cache_key, model_name, validated_model_properties, "cuda", True)

                if model_cache_key not in available_models:
                    raise AssertionError

                res = get_loaded_models()["models"]
                assert res[model_name] == "cuda"

                eject_model(model_name, "cuda")

                if model_cache_key in available_models:
                    raise AssertionError










