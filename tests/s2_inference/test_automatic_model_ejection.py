import unittest
import os
import torch
import unittest.mock
from marqo.s2_inference.errors import ModelCacheManageError
from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties
import numpy as np
from marqo.tensor_search import tensor_search
from marqo.s2_inference.s2_inference import clear_loaded_models, vectorise, device_memory_manage,\
                                            check_device_memory_status, _validate_model_properties,\
                                            get_model_size
import marqo.s2_inference.constants


class TestAutomaticModelEject(unittest.TestCase):
    def setUp(self) -> None:
        clear_loaded_models()

        self.inde_name = "test_index"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.inde_name)
        except Exception:
            pass

    def test_device_memory_manage(self):
        def pass_through_device_memory_manage(*arg, **kwargs):
            return device_memory_manage(*arg, **kwargs)

        mock_device_memory_manage = unittest.mock.MagicMock()
        mock_device_memory_manage.side_effect = pass_through_device_memory_manage

        small_list_of_models = ['open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
            "sentence-transformers/all-MiniLM-L6-v2", "flax-sentence-embeddings/all_datasets_v4_mpnet-base", 'open_clip/ViT-B-16/laion2b_s34b_b88k']
        content = "Try to kill the cpu"

        @unittest.mock.patch("marqo.s2_inference.s2_inference.device_memory_manage", mock_device_memory_manage)
        def run():
            for model in small_list_of_models:
                _ = vectorise(model_name=model, content=content, device="cpu")
            checked_models = [call_kwargs["device"] for call_args, call_kwargs
                                                in mock_device_memory_manage.call_args_list]

            self.assertEqual(small_list_of_models, checked_models)
            return True
        assert run

    def test_check_device_memory_status(self):
        def pass_through_check_device_memory_status(*arg, **kwargs):
            return check_device_memory_status(*arg, **kwargs)

        mock_check_device_memory_status = unittest.mock.MagicMock()
        mock_check_device_memory_status.side_effect = pass_through_check_device_memory_status

        small_list_of_models = ['open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
            "sentence-transformers/all-MiniLM-L6-v2", "flax-sentence-embeddings/all_datasets_v4_mpnet-base", 'open_clip/ViT-B-16/laion2b_s34b_b88k']
        content = "Try to kill the cpu"

        @unittest.mock.patch("marqo.s2_inference.s2_inference.check_device_memory_status", mock_check_device_memory_status )
        def run():
            for model in small_list_of_models:
                _ = vectorise(model_name=model, content=content, device="cpu")
            checked_devices = [call_kwargs["device"] for call_args, call_kwargs
                                                in mock_check_device_memory_status.call_args_list]
            self.assertEqual(len(checked_devices), 5)
            self.assertEqual(set(checked_devices), {"cpu"})
            return True
        assert run

    def test_load_very_large_model(self):
        huge_models = ['open_clip/ViT-g-14/laion2b_s12b_b42k']
        for model in huge_models:
            try:
                _ = vectorise(model_name=model, content = 'this is a huge model', device="cpu")
                raise AssertionError
            except ModelCacheManageError as e:
                assert "CANNOT find enough space" in e.message


    def test_get_model_size(self):
        models_and_sizes = {
            "open_clip/ViT-L-14/openai" : 1.5,
            'open_clip/ViT-L-14/laion400m_e31' :1.5,
            'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k': 1,
            "sentence-transformers/all-MiniLM-L6-v2": 0.7,
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base" : 0.7,
            'open_clip/ViT-B-16/laion2b_s34b_b88k': 1,
            'open_clip/coca_ViT-L-14/laion2b_s13b_b90k':1.5,
            'open_clip/RN50x64/openai':1,
            "onnx16/open_clip/ViT-B-32/laion2b_e16":1,
        }

        for model_name, size in models_and_sizes.items():
            self.assertEqual(get_model_size(model_name, _validate_model_properties(model_name, None)), size, msg=model_name)

        generic_model = {
            "model_name" : "my_custom_clip",
            "model_properties_1" : {
                "name" : "ViT-L-14",
                "type":"open_clip",
                "dimensions" : 768,
                "model_size" : 1.53,
            },
            "model_properties_2": {
                "name": "ViT-L/14",
                "dimensions": 768,
                "type": "clip",
            }
        }

        self.assertEqual(get_model_size(generic_model["model_name"],generic_model["model_properties_1"]), 1.53)
        self.assertEqual(get_model_size(generic_model["model_name"], generic_model["model_properties_2"]), 1.5)

    def test_model_management(self):
        # Instance should be out of memory without model management
        content = "Try to kill the cpu"
        list_of_models = [
            "open_clip/ViT-L-14/openai", 'open_clip/ViT-L-14/laion400m_e31', 'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
            "sentence-transformers/all-MiniLM-L6-v2", "flax-sentence-embeddings/all_datasets_v4_mpnet-base", 'open_clip/ViT-B-16/laion2b_s34b_b88k',
            'open_clip/coca_ViT-L-14/laion2b_s13b_b90k', 'open_clip/RN50x64/openai', "onnx16/open_clip/ViT-B-32/laion2b_e16"
        ]
        for model in list_of_models:
            _ = vectorise(model_name = model, content = content, device="cpu")






