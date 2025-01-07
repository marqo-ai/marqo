import functools
import os
import torch
import pytest
import json
from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry, _convert_tensor_to_numpy
from unittest.mock import patch
import numpy as np
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise,
    _convert_vectorized_output,
)
from marqo.s2_inference.s2_inference import _load_model as og_load_model
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry, \
    _convert_tensor_to_numpy
from marqo.s2_inference.types import FloatTensor
from tests.marqo_test import TestImageUrls
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.s2_inference.configs import ModelCache
import shutil

_load_model = functools.partial(og_load_model, calling_func="unit_test")


def remove_cached_model_files():
    '''
    This function removes all the cached models from the cache paths to save disk space
    '''
    cache_paths = ModelCache.get_all_cache_paths()
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            for item in os.listdir(cache_path):
                item_path = os.path.join(cache_path, item)
                # Check if the item is a file or directory
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)


def get_absolute_file_path(filename: str) -> str:
    currentdir = os.path.dirname(os.path.abspath(__file__))
    abspath = os.path.join(currentdir, filename)
    return abspath


def run_test_vectorize(models, model_type, compare_hardcoded_embeddings=True):
    # model_type determines the filename with which the embeddings are saved/loaded
    # Ensure that vectorised output from vectorise function matches both the model.encode output and
    # hardcoded embeddings from Python 3.8
    
    sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
    device = "cuda"
    eps = 1e-9

    if compare_hardcoded_embeddings:
        embeddings_reference_file = get_absolute_file_path(
            f"embeddings_reference/embeddings_{model_type}_python_3_8.json"
        )

        # Load in hardcoded embeddings json file
        if os.path.exists(embeddings_reference_file) and os.path.isfile(embeddings_reference_file):
            with open(embeddings_reference_file, "r") as f:
                embeddings_python_3_8 = json.load(f)
        else:
            print(f"Embeddings reference file not found at {embeddings_reference_file}. Skipping hardcoded embeddings test"
                  f" for model type: {model_type}")
            embeddings_python_3_8 = None

    with patch.dict(os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "10"}):
        def run():
            for name in models:
                model_properties = get_model_properties_from_registry(name)
                model = _load_model(model_properties['name'], model_properties=model_properties, device=device, )

                for sentence in sentences:
                    output_v = vectorise(name, sentence, model_properties, device, normalize_embeddings=True)

                    assert _check_output_type(output_v)

                    output_m = model.encode(sentence, normalize=True)

                    # Converting output_m to numpy if it is cuda.
                    if type(output_m) == torch.Tensor:
                        output_m = output_m.cpu().numpy()

                    # Embeddings must match hardcoded python 3.8.20 embeddings
                    if isinstance(sentence, str):
                        try:
                            if compare_hardcoded_embeddings and embeddings_python_3_8:
                                assert np.allclose(output_m, embeddings_python_3_8[name][sentence], atol=1e-6), \
                                    (f"Hardcoded Python 3.8 embeddings do not match for model: {name}, "
                                     f"sentence: {sentence}")
                        except KeyError:
                            raise KeyError(f"Hardcoded Python 3.8 embeddings not found for "
                                           f"model: {name}, sentence: {sentence} in JSON file: "
                                           f"{embeddings_reference_file}")

                    assert np.allclose(output_m, output_v, atol=eps)

                clear_loaded_models()

                # delete the model to free up memory,
                # it is hacked loading from _load_model, so we need to delete it manually
                del model

            return True

        assert run()


def run_test_model_outputs(models):
    sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
    device = "cuda"

    for name in models:
        model_properties = get_model_properties_from_registry(name)
        model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

        for sentence in sentences:
            output = model.encode(sentence)
            assert _check_output_type(_convert_vectorized_output(output))

        del model
        clear_loaded_models()


def run_test_model_normalization(models):
    sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
    device = "cuda"
    eps = 1e-6

    for name in models:
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


def run_test_cuda_encode_type(models):
    sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
    device = 'cuda'

    for name in models:
        model_properties = get_model_properties_from_registry(name)
        model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

        for sentence in sentences:
            output_v = _convert_tensor_to_numpy(model.encode(sentence, normalize=True))
            assert isinstance(output_v, np.ndarray)

        del model
        clear_loaded_models()


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False,
                    reason="We skip the large model test if we don't have cuda support")
class TestLargeClipModels(unittest.TestCase):
    def setUp(self):
        self.models = [
            'open_clip/ViT-L-14/laion400m_e32',
            'Marqo/ViT-L-14.laion400m_e32',
            'open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k',
            'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup',
            'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup',
            'open_clip/convnext_large_d/laion2b_s26b_b102k_augreg',
            'open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k',
            'Marqo/xlm-roberta-base-ViT-B-32.laion5b_s13b_b90k',
            'open_clip/ViT-H-14-378-quickgelu/dfn5b',
            'open_clip/ViT-SO400M-14-SigLIP-384/webli',
            "visheratin/nllb-siglip-mrl-large",
            "visheratin/nllb-clip-large-siglip",
            "visheratin/nllb-siglip-mrl-base",
            "visheratin/nllb-clip-base-siglip"
        ]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="large_open_clip", compare_hardcoded_embeddings=False)
        
    def test_load_clip_text_model(self):
        device = "cuda"
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        for name in self.models:
            with self.subTest(f"Testing model: {name}"):
                model = _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)

                for text in texts:
                    assert abs(model.encode(text) - model.encode([text])).sum() < eps
                    assert abs(model.encode_text(text) - model.encode([text])).sum() < eps
                    assert abs(model.encode(text) - model.encode_text([text])).sum() < eps

                del model
                clear_loaded_models()

    def test_model_outputs(self):
        for model_name in self.models:
            with self.subTest(f"Testing model: {model_name}"):
                run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            with self.subTest(f"Testing model: {model_name}"):
                run_test_model_normalization([model_name])

    def test_cuda_encode_type(self):
        for model_name in self.models:
            with self.subTest(f"Testing model: {model_name}"):
                run_test_cuda_encode_type([model_name])

    @patch("torch.cuda.amp.autocast")
    def test_autocast_called_in_open_clip(self, mock_autocast):
        names = ["open_clip/ViT-B-32/laion400m_e31", "Marqo/ViT-B-32.laion400m_e31"]
        contents = ['this is a test sentence. so is this.',
                    TestImageUrls.IMAGE0.value]
        for model_name in names:
            with self.subTest(f"Testing model: {model_name}"):
                for content in contents:
                    vectorise(model_name=model_name, content=content, device="cuda")
                    mock_autocast.assert_called_once()
                    mock_autocast.reset_mock()


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False,
                    reason="We skip the large model test if we don't have cuda support")
class TestE5Models(unittest.TestCase):
    def setUp(self):
        self.models = ["hf/e5-large", "hf/e5-large-unsupervised"]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="large_e5")

    def test_model_outputs(self):
        for model_name in self.models:
            run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            run_test_model_normalization([model_name])

    def test_cuda_encode_type(self):
        for model_name in self.models:
            run_test_cuda_encode_type([model_name])


@pytest.mark.largemodel
@pytest.mark.skip(reason="Needs further investigation")
class TestBGEModels(unittest.TestCase):
    def setUp(self):
        self.models = ["hf/bge-large-zh-v1.5", "hf/bge-large-en-v1.5"]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="large_bge")

    def test_model_outputs(self):
        for model_name in self.models:
            run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            run_test_model_normalization([model_name])

    def test_cuda_encode_type(self):
        for model_name in self.models:
            run_test_cuda_encode_type([model_name])


@pytest.mark.largemodel
@pytest.mark.skip(reason="Needs further investigation")
class TestSnowflakeModels(unittest.TestCase):
    def setUp(self):
        self.models = ["hf/snowflake-arctic-embed-l"]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="large_snowflake")

    def test_model_outputs(self):
        for model_name in self.models:
            run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            run_test_model_normalization([model_name])

    def test_cuda_encode_type(self):
        for model_name in self.models:
            run_test_cuda_encode_type([model_name])


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False,
                    reason="We skip the large model test if we don't have cuda support")
class TestMultilingualE5Models(unittest.TestCase):
    def setUp(self):
        self.models = [
            "hf/multilingual-e5-small",
            "hf/multilingual-e5-base",
            "hf/multilingual-e5-large",
            "hf/multilingual-e5-large-instruct"
        ]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="large_multilingual_e5")

    def test_model_outputs(self):
        for model_name in self.models:
            run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            run_test_model_normalization([model_name])

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
            for model_name in self.models:
                english_feature = np.array(
                    vectorise(model_name=model_name, content=english_text, normalize_embeddings=True, device=device))
                for other_language_text in other_language_texts:
                    other_language_feature = np.array(vectorise(model_name=model_name, content=other_language_text,
                                                                normalize_embeddings=True, device=device))
                    assert np.allclose(english_feature, other_language_feature, atol=e)

    def test_cuda_encode_type(self):
        run_test_cuda_encode_type(
            self.models + ["fp16/ViT-B/32", "open_clip/convnext_base_w/laion2b_s13b_b82k",
                           "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg",
                           "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1",
                           "hf/all_datasets_v4_MiniLM-L6",]
        )


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False,
                    reason="We skip the large model test if we don't have cuda support")
class TestLanguageBindModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        clear_loaded_models()

    @classmethod
    def tearDownClass(cls) -> None:
        clear_loaded_models()

    def setUp(self):
        self.models = ["LanguageBind/Video_V1.5_FT_Audio_FT_Image"]
        self.device="cuda"

    def _help_test_vectorise(self, model_name, modality, test_content_list):
        for content in test_content_list:
            with self.subTest(model=model_name, content=content, normalized=True):
                normalized_embeddings_list = vectorise(
                    model_name=model_name,
                    content=content, device=self.device, normalize_embeddings=True,
                    modality=modality
                )
                for embeddings in normalized_embeddings_list:
                    self.assertTrue(np.linalg.norm(np.array(embeddings)) - 1 < 1e-6)

            if modality != Modality.TEXT: # Text embeddings are always normalized
                with self.subTest(model=model_name, content=content, normalized=False):
                    unnormalized_embeddings_list = vectorise(
                        model_name=model_name,
                        content=content, device=self.device, normalize_embeddings=False,
                        modality=modality
                    )
                    for embeddings in unnormalized_embeddings_list:
                        # TODO: Record unnormalized embeddings and compare with json
                        self.assertTrue(np.linalg.norm(np.array(embeddings)) - 1 > 1e-2)

    def test_models(self):
        test_cases = {
            Modality.TEXT: ["test", ["test2", "test3"]],
            Modality.AUDIO: [
                "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/4-145081-A-9.wav",
                [
                    "https://marqo-ecs-50-audio-test-dataset.s3.us-east-1.amazonaws.com/audios/1-115920-A-22.wav",
                    "https://marqo-ecs-50-audio-test-dataset.s3.us-east-1.amazonaws.com/audios/1-115920-A-22.wav"
                ]
            ],
            Modality.IMAGE: [
                'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg',
                [
                    'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg',
                    'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg'
                ]
            ],
            Modality.VIDEO: [
                'https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/--bO6XwZ9HI_000041_000051.mp4',
                [
                    'https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/-0MVWb7nJLY_000008_000018.mp4',
                    'https://marqo-k400-video-test-dataset.s3.us-east-1.amazonaws.com/videos/-0oMsq-9b6c_000095_000105.mp4'
                ]
            ]
        }

        for model_name in self.models:
            for modality, test_content_list in test_cases.items():
                with self.subTest(model=model_name, modality=modality):
                    self._help_test_vectorise(model_name, modality, test_content_list)


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False,
                    reason="We skip the large model test if we don't have cuda support")
class TestStellaModels(unittest.TestCase):
    def setUp(self):
        self.models = ["Marqo/dunzhang-stella_en_400M_v5"]

    def tearDown(self):
        clear_loaded_models()

    @classmethod
    def setUpClass(cls) -> None:
        remove_cached_model_files()

    @classmethod
    def tearDownClass(cls) -> None:
        remove_cached_model_files()

    def test_vectorize(self):
        # For GPU Memory Optimization, we shouldn't load all models at once
        run_test_vectorize(models=self.models, model_type="stella")

    def test_model_outputs(self):
        for model_name in self.models:
            run_test_model_outputs([model_name])

    def test_model_normalization(self):
        for model_name in self.models:
            run_test_model_normalization([model_name])

    def test_cuda_encode_type(self):
        for model_name in self.models:
            run_test_cuda_encode_type([model_name])
