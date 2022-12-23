import torch.cuda
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.s2_inference import _validate_model_properties,\
    _create_model_cache_key, _update_available_models, available_models
from marqo.tensor_search.tensor_search import eject_model, get_cuda_info, get_loaded_models
from marqo.errors import ModelNotInCache, HardwareCompatabilityError
import psutil



def load_model(model_name: str, device: str) -> None:
    validated_model_properties = _validate_model_properties(model_name, None)
    model_cache_key = _create_model_cache_key(model_name, device, validated_model_properties)
    _update_available_models(model_cache_key, model_name, validated_model_properties, device, True)


class TestModelCacheManagement(MarqoTestCase):

    def setUp(self) -> None:
        # We pre-define 3 dummy models for testing purpose
        self.MODEL_1 = "ViT-L/14"
        self.MODEL_2 = "open_clip/ViT-L-14/laion400m_e31"
        self.MODEL_3 = "hf/all-MiniLM-L6-v2"
        self.MODEL_LIST = [self.MODEL_1, self.MODEL_2, self.MODEL_3]
        self.CUDA_FLAG = torch.cuda.is_available()


        # load several models into cache for setting up
        for model_name in self.MODEL_LIST:
            load_model(model_name, "cpu")

        if self.CUDA_FLAG:
            for model_name in self.MODEL_LIST:
                load_model(model_name, "cuda")


        # We loaded 6 models (3 in cuda, 3 in cpu) as initial setup
        if self.CUDA_FLAG:
            assert len(available_models) == 6
        else:
            assert len(available_models) == 3


    def test_eject_model_cpu(self):
        for model_name in self.MODEL_LIST:
            eject_model(model_name, "cpu")
            if (model_name, "cpu") in available_models:
                raise AssertionError


        my_test_model_1 = "test-model-1"
        my_test_model_2 = "test-model-2"

        try:
            eject_model(my_test_model_1, "cpu")
        except ModelNotInCache:
            pass

        try:
            eject_model(my_test_model_2, "cpu")
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
        try:
            get_cuda_info()
        except HardwareCompatabilityError:
            pass


    def test_loaded_models(self):

        loaded_models = get_loaded_models()["models"]
        loaded_models_keys = [_create_model_cache_key(dic["model_name"], dic["device"],
                                _validate_model_properties(dic["model_name"], None)) for dic in loaded_models]
        assert loaded_models_keys==list(available_models.keys())


    def test_edge_case_cuda(self):
        if self.CUDA_FLAG:
            test_iterations = 10
            # Note this is a time consuming test.

            for i in range(test_iterations):
                eject_model(self.MODEL_1, "cuda")
                load_model(self.MODEL_1, "cuda")

                for id in range(torch.cuda.device_count()):
                    # cuda usage
                    assert torch.cuda.memory_allocated(id) < torch.cuda.get_device_properties(id).total_memory
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











