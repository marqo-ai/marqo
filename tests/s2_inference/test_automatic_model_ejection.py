import unittest
import os
import torch

from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties
import numpy as np
from marqo.tensor_search import tensor_search
from marqo.s2_inference.s2_inference import clear_loaded_models, vectorise


class TestEncoding(unittest.TestCase):
    def setUp(self) -> None:
        clear_loaded_models()

        self.inde_name = "test_index"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.inde_name)
        except Exception:
            pass

        self.list_of_models = [
            "open_clip/ViT-L-14/openai", 'open_clip/ViT-L-14/laion400m_e31', 'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
            "sentence-transformers/all-MiniLM-L6-v2", "flax-sentence-embeddings/all_datasets_v4_mpnet-base", 'open_clip/ViT-B-16/laion2b_s34b_b88k',
            'open_clip/coca_ViT-L-14/laion2b_s13b_b90k', 'open_clip/RN50x64/openai', "onnx16/open_clip/ViT-B-32/laion2b_e16"
        ]

    def test_model_management(self):
        # Instance should be out of memory without model management
        content = "Try to kill the cpu"
        for model in self.list_of_models:
            _ = vectorise(model_name = model, content = content, device="cpu")


