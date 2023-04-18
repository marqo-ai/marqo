import os
import torch
import pytest
from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry, _convert_tensor_to_numpy
from unittest.mock import patch
import numpy as np
import unittest
from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise,
    _convert_vectorized_output,
)
import functools
from marqo.s2_inference.s2_inference import _load_model as og_load_model
_load_model = functools.partial(og_load_model, calling_func = "unit_test")


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
class TestLargeModelEncoding(unittest.TestCase):

    def setUp(self) -> None:
        self.large_clip_models = ["onnx32/openai/ViT-L/14", "open_clip/ViT-L-14/openai", "ViT-L/14"]

        self.multilingual_models = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14"]

        self.e5_models = ["hf/e5-large", "hf/e5-large-unsupervised"]


    def tearDown(self) -> None:
        clear_loaded_models()


    def test_vectorize(self):
        names = self.large_clip_models + self.e5_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = "cuda"
        eps = 1e-9
        with patch.dict(os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "8"}):
            def run():
                for name in names:
                    model_properties = get_model_properties_from_registry(name)
                    model = _load_model(model_properties['name'], model_properties=model_properties, device=device, )

                    for sentence in sentences:
                        output_v = vectorise(name, sentence, model_properties, device, normalize_embeddings=True)

                        assert _check_output_type(output_v)

                        output_m = model.encode(sentence, normalize=True)

                        # Converting output_m to numpy if it is cuda.
                        if type(output_m) == torch.Tensor:
                            output_m = output_m.cpu().numpy()

                        assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps
                    clear_loaded_models()
                return True

            assert run()


    def test_load_clip_text_model(self):
        names = self.large_clip_models
        device = "cuda"
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        for name in names:

            model = _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)

            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode_text(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode_text([text])).sum() < eps

            clear_loaded_models()


    def test_model_outputs(self):
        names = ["onnx16/open_clip/ViT-B-32/laion400m_e32"] + self.e5_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = "cuda"

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence)
                assert _check_output_type(_convert_vectorized_output(output))

            clear_loaded_models()


    def test_model_normalization(self):
        names = self.large_clip_models + self.e5_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = "cuda"
        eps = 1e-6

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence, normalize=True)
                output = _convert_vectorized_output(output)
                max_output_norm = max(torch.linalg.norm(FloatTensor(output), dim=1))
                min_output_norm = min(torch.linalg.norm(FloatTensor(output), dim=1))

                assert abs(max_output_norm - 1) < eps, f"{name}, {sentence}"
                assert abs(min_output_norm - 1) < eps, f"{name}, {sentence}"

            clear_loaded_models()


    def test_multilingual_clip_performance(self):

        clear_loaded_models()

        names = self.multilingual_models
        device = "cuda"
        texts = [
            "skiing person",
            "滑雪的人",
            "лыжник",
            "persona che scia",
        ]
        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"
        e = 0.1
        with patch.dict(os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "8"}):
            def run():
                for name in names:
                    text_feature = np.array(vectorise(model_name=name, content=texts, normalize_embeddings=True, device=device))
                    image_feature = np.array(vectorise(model_name=name, content=image, normalize_embeddings=True, device=device))

                    clear_loaded_models()
                    similarity_score = (text_feature @ image_feature.T).flatten()

                    assert np.abs(np.max(similarity_score) - np.min(similarity_score)) < e

                    del similarity_score

                return True
            assert run()


    def test_cuda_encode_type(self):
        names = self.large_clip_models + self.e5_models

        names += ["fp16/ViT-B/32", "open_clip/convnext_base_w/laion2b_s13b_b82k",
                 "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg",
                 "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised",     "hf/e5-base-unsupervised"]
        names += names_e5

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cuda'

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output_v = _convert_tensor_to_numpy(model.encode(sentence, normalize=True))
                assert isinstance(output_v, np.ndarray)

            clear_loaded_models()




