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
from marqo.s2_inference.configs import ModelCache
import shutil


def remove_cached_clip_files():
    '''
    This function removes all the cached models from the clip cache path to save disk space
    '''
    clip_cache_path = ModelCache.clip_cache_path
    if os.path.exists(clip_cache_path):
        for item in os.listdir(clip_cache_path):
            item_path = os.path.join(clip_cache_path, item)
            # Check if the item is a file or directory
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
class TestLargeModelEncoding(unittest.TestCase):

    def setUp(self) -> None:
        self.large_clip_models = [ "open_clip/ViT-L-14/laion400m_e32",
                                   'open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k',
                                   #'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup',  this model is not currently available in open_clip
                                   'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup',
                                   'open_clip/convnext_large_d/laion2b_s26b_b102k_augreg']

        self.multilingual_models = ["hf/multilingual-e5-small", "hf/multilingual-e5-base", "hf/multilingual-e5-large"]

        self.e5_models = ["hf/e5-large", "hf/e5-large-unsupervised"]

    def tearDown(self) -> None:
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_clip_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_clip_files()

    def test_vectorize(self):
        names = self.large_clip_models + self.e5_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = "cuda"
        eps = 1e-9
        with patch.dict(os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "6"}):
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
                    # delete the model to free up memory,
                    # it is hacked loading from _load_model, so we need to delete it manually
                    del model

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

            del model
            clear_loaded_models()


    def test_model_outputs(self):
        names = self.large_clip_models+ self.e5_models
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = "cuda"

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence)
                assert _check_output_type(_convert_vectorized_output(output))

            del model
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

            del model
            clear_loaded_models()

    def test_multilingual_e5_model_performance(self):
        clear_loaded_models()
        device = "cuda"
        english_text = "skiing person"
        other_language_texts = [
            "滑雪的人",
            "лыжник",
            "persona che scia",
        ]
        e = 1
        with patch.dict(os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "10"}):
            for model_name in self.multilingual_models:
                english_feature = np.array(
                    vectorise(model_name=model_name, content=english_text, normalize_embeddings=True, device=device))
                for other_language_text in other_language_texts:
                    other_language_feature = np.array(vectorise(model_name=model_name, content=other_language_text,
                                                                normalize_embeddings=True, device=device))
                    assert np.allclose(english_feature, other_language_feature, atol=e)

    def test_cuda_encode_type(self):
        names = self.large_clip_models + self.e5_models

        names += ["fp16/ViT-B/32", "open_clip/convnext_base_w/laion2b_s13b_b82k",
                 "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg",
                 "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised"]
        names += names_e5

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cuda'

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output_v = _convert_tensor_to_numpy(model.encode(sentence, normalize=True))
                assert isinstance(output_v, np.ndarray)

            del model
            clear_loaded_models()

    @patch("torch.cuda.amp.autocast")
    def test_autocast_called_in_open_clip(self, mock_autocast):
        names = ["open_clip/ViT-B-32/laion400m_e31"]
        contents = ['this is a test sentence. so is this.',
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg"]
        for model_name in names:
            for content in contents:
                vectorise(model_name=model_name, content=content, device="cuda")
                mock_autocast.assert_called_once()
                mock_autocast.reset_mock()