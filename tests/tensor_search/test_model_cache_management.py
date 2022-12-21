import json
import pprint
import time
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import unittest
import copy
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.s2_inference import _validate_model_properties,\
    _create_model_cache_key, _update_available_models, available_models
from marqo.tensor_search.tensor_search import eject_model, get_cuda_info, get_loaded_models
from marqo.errors import ModelNotInCache, HardwareCompatabilityError



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


        # load several models into cache for setting up
        for model_name in self.MODEL_LIST:
            load_model(model_name, "cuda")
            load_model(model_name, "cpu")
        print(available_models)
        # We will load 6 models (3 in cuda, 3 in cpu) as initial setup

    def test_eject_model(self):
        print(available_models)

        for model_name in self.MODEL_LIST:
            eject_model(model_name, "cpu")
            if (model_name, "cpu") in available_models:
                raise AssertionError(f"Model= {model_name} device = cpu is not deleted from cache")


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
            eject_model(my_test_model_1, "cpu")
        except ModelNotInCache:
            pass

        try:
            eject_model(my_test_model_2, "cuda")
        except ModelNotInCache:
            pass

        try:
            eject_model(my_test_model_2, "cpu")
        except ModelNotInCache:
            pass

    def test_cuda_info(self):
        try:
            get_cuda_info()
        except HardwareCompatabilityError:
            pass

    def test_loaded_models(self) -> dict:

        loaded_models = get_loaded_models()["models"]
        loaded_models_list = [(key, loaded_models[key]) for key in loaded_models]
        assert loaded_models_list==available_models.keys()










