import unittest
import os
import torch
import pytest

from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties
import numpy as np

from marqo.s2_inference.s2_inference import (
    _load_model,
    _check_output_type, vectorise,
    _convert_vectorized_output,
)

@pytest.fixture(scope="session")
def enable_flag(pytestconfig):
    return pytestconfig.getoption("largemodel")


@pytest.mark.skip(reason="We skip the large model test")
class TestLargeModelEncoding(unittest.TestCase):

    def setUp(self) -> None:
        self.large_clip_models = ['onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k',"onnx32/openai/ViT-L/14",
                                    "open_clip/ViT-L-14/openai", "ViT-L/14", 'onnx32/open_clip/ViT-H-14/laion2b_s32b_b79k',
                                    'open_clip/ViT-H-14/laion2b_s32b_b79k']

        self.multilingual_models = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                                    "multilingual-clip/XLM-Roberta-Large-Vit-B-32", "multilingual-clip/LABSE-Vit-L-14"]


    def tearDown(self) -> None:
        clear_loaded_models()


    def test_vectorize(self):
        names = self.large_clip_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device, )

            for sentence in sentences:
                output_v = vectorise(name, sentence, model_properties, device, normalize_embeddings=True)

                assert _check_output_type(output_v)

                output_m = model.encode(sentence, normalize=True)

                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps

            clear_loaded_models()


    def test_load_clip_text_model(self):
        names = self.large_clip_models
        device = 'cpu'
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
        names = ["onnx16/open_clip/ViT-B-32/laion400m_e32"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence)
                assert _check_output_type(_convert_vectorized_output(output))

            clear_loaded_models()


    def test_model_normalization(self):
        names = self.large_clip_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
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
        device = 'cpu'
        texts = [
            "skiing person",
            "滑雪的人",
            "лыжник",
            "persona che scia",
        ]
        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"
        e = 0.1
        for name in names:
            text_feature = np.array(vectorise(model_name=name, content=texts, normalize_embeddings=True, device=device))
            image_feature = np.array(vectorise(model_name=name, content=image, normalize_embeddings=True, device=device))

            clear_loaded_models()
            similarity_score = (text_feature @ image_feature.T).flatten()

            assert np.abs(np.max(similarity_score) - np.min(similarity_score)) < e

            del similarity_score




