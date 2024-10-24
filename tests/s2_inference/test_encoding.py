import unittest
import torch
import json
import numpy as np
from unittest.mock import MagicMock, patch
from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties
from marqo.s2_inference.s2_inference import _convert_tensor_to_numpy
import numpy as np
import functools
import os

from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise,
    _convert_vectorized_output,
)
from marqo.s2_inference.s2_inference import _load_model as og_load_model
from tests.marqo_test import TestImageUrls

_load_model = functools.partial(og_load_model, calling_func = "unit_test")


def get_absolute_file_path(filename: str) -> str:
    currentdir = os.path.dirname(os.path.abspath(__file__))
    abspath = os.path.join(currentdir, filename)
    return abspath


class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        clear_loaded_models()

    def test_vectorize(self):
        """
        Ensure that vectorised output from vectorise function matches both the model.encode output and
        hardcoded embeddings from Python 3.8.20
        """

        names = ["fp16/ViT-B/32", "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "hf/bge-small-en-v1.5", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]

        names_bge = ["hf/bge-small-en-v1.5", "hf/bge-base-en-v1.5"]

        names_snowflake = ["hf/snowflake-arctic-embed-m", "hf/snowflake-arctic-embed-m-v1.5"]

        names = names + names_e5 + names_bge + names_snowflake

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9
        embeddings_file_name = get_absolute_file_path("embeddings_reference/embeddings_python_3_8.json")

        # Load in hardcoded embeddings json file
        with open(embeddings_file_name, "r") as f:
            embeddings_python_3_8 = json.load(f)

        for name in names:
            with self.subTest(name=name):
                # Add hardcoded embeddings into the variable.
                model_properties = get_model_properties_from_registry(name)
                model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

                for sentence in sentences:
                    with self.subTest(sentence=sentence):
                        output_v = vectorise(name, sentence, model_properties, device, normalize_embeddings=True)
                        assert _check_output_type(output_v)

                        output_m = model.encode(sentence, normalize=True)

                        # Embeddings must match hardcoded python 3.8.20 embeddings
                        if isinstance(sentence, str):
                            with self.subTest("Hardcoded Python 3.8 Embeddings Comparison"):
                                try:
                                    self.assertEqual(np.allclose(output_m, embeddings_python_3_8[name][sentence],
                                                                 atol=1e-6),
                                                 True)
                                except KeyError:
                                    raise KeyError(f"Hardcoded Python 3.8 embeddings not found for "
                                                   f"model: {name}, sentence: {sentence} in JSON file: "
                                                   f"{embeddings_file_name}")

                        with self.subTest("Model encode vs vectorize"):
                            self.assertEqual(np.allclose(output_m, output_v, atol=eps), True)

                clear_loaded_models()


    def test_vectorize_normalise(self):
        open_clip_names = ["open_clip/ViT-B-32/laion2b_s34b_b79k"]

        names_bge = ["hf/bge-small-en-v1.5", "hf/bge-base-en-v1.5"]

        names_snowflake = ["hf/snowflake-arctic-embed-m", "hf/snowflake-arctic-embed-m-v1.5"]

        names = open_clip_names + names_bge + names_snowflake
                 
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output_v = vectorise(name, sentence, model_properties, device, normalize_embeddings=True)
                assert _check_output_type(output_v)
                output_m = model.encode(sentence, normalize=True)
                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps
                for vector in output_v:
                    assert abs(np.linalg.norm(np.array(vector)) - 1) < 1e-5

                output_v_unnormalised = vectorise(name, sentence, model_properties, device, normalize_embeddings=False)
                assert _check_output_type(output_v)
                output_m_unnormalised = model.encode(sentence, normalize=False)
                assert abs(torch.FloatTensor(output_v_unnormalised) - torch.FloatTensor(output_m_unnormalised)).sum() < eps
                for vector in output_v_unnormalised:
                    assert abs(np.linalg.norm(np.array(vector)) - 1) > 1e-5

            clear_loaded_models()

    def test_cpu_encode_type(self):
        names = ["fp16/ViT-B/32", "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "hf/bge-small-en-v1.5", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]

        names_bge = ["hf/bge-small-en-v1.5", "hf/bge-base-en-v1.5"]

        names_snowflake = ["hf/snowflake-arctic-embed-m", "hf/snowflake-arctic-embed-m-v1.5"]

        names = names + names_e5 + names_bge + names_snowflake

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output_v = _convert_tensor_to_numpy(model.encode(sentence, normalize=True))
                assert isinstance(output_v, np.ndarray)

            clear_loaded_models()


    def test_load_clip_text_model(self):
        names = ["fp16/ViT-B/32", "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                  'RN50', "ViT-B/16"]

        device = 'cpu'
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        for name in names:

            model =  _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)

            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode_text(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode_text([text])).sum() < eps

            clear_loaded_models()


    def test_load_sbert_text_model(self):
        names = ["all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6"]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

            clear_loaded_models()


    def test_load_hf_text_model(self):
        names = ["hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "hf/bge-small-en-v1.5"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]
        names += names_e5

        device = 'cpu'
        eps = 1e-9

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

            clear_loaded_models()


    def test_load_onnx_sbert_text_model(self):
        names = ["onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

            clear_loaded_models()


    def test_compare_onnx_sbert_text_models(self):
        names_sbert_onnx = [("all-MiniLM-L6-v1", "onnx/all-MiniLM-L6-v1"),
                            ("all_datasets_v4_MiniLM-L6", "onnx/all_datasets_v4_MiniLM-L6")]
        sentences = ['hello', 'this is a test sentence. so is this.']
        device = 'cpu'
        eps = 1e-4

        for name_sbert, name_onnx in names_sbert_onnx:
            for sentence in sentences:
                model_properties_sbert = get_model_properties_from_registry(name_sbert)
                model_sbert = _load_model(model_properties_sbert['name'], model_properties=model_properties_sbert, device=device)

                model_properties_onnx = get_model_properties_from_registry(name_onnx)
                model_onnx = _load_model(model_properties_onnx['name'], model_properties=model_properties_onnx, device=device)

                assert abs(model_onnx.encode(sentence) - model_sbert.encode(sentence)).sum() < eps

            clear_loaded_models()


    def test_model_outputs(self):
        names = ["fp16/ViT-B/32", "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',"all-MiniLM-L6-v1",
                 "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "hf/bge-small-en-v1.5", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]
        names += names_e5
                 
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
        names = ["fp16/ViT-B/32", "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'RN50', "ViT-B/16", "all-MiniLM-L6-v1",
                 "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "hf/bge-small-en-v1.5", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]
        names += names_e5
                 
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


    def test_model_un_normalization(self):
        # note: sbert native seems to provide normalized embeddings even with = False, needs more investigation
        # , 
        names = [ 'RN50', "ViT-B/16", "hf/all-MiniLM-L6-v1",
                 "hf/all_datasets_v4_MiniLM-L6", "hf/bge-small-en-v1.5",
                  "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]

        names_e5 = ["hf/e5-small", "hf/e5-base", "hf/e5-small-unsupervised", "hf/e5-base-unsupervised", "hf/e5-base-v2",
                    "hf/multilingual-e5-small"]
        names += names_e5

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-3

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence, normalize=False)
                output = _convert_vectorized_output(output)
                max_output_norm = max(torch.linalg.norm(FloatTensor(output), dim=1))
                min_output_norm = min(torch.linalg.norm(FloatTensor(output), dim=1))

                assert abs(max_output_norm - 1) > eps, f"{name}, {sentence}"
                assert abs(min_output_norm - 1) > eps, f"{name}, {sentence}"

            clear_loaded_models()


class TestOpenClipModelEncoding(unittest.TestCase):
    '''
    This test is for open clip models as they are heavily used in production.
    '''

    def setUp(self) -> None:
        self.open_clip_test_model = [
            'open_clip/RN50/yfcc15m', 'open_clip/ViT-B-32/laion2b_s34b_b79k',
            'open_clip/ViT-B-16/laion2b_s34b_b88k', 'open_clip/convnext_base/laion400m_s13b_b51k',
            'open_clip/convnext_base_w/laion_aesthetic_s13b_b82k',
            'open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k',
            'open_clip/EVA02-B-16/merged2b_s8b_b131k',
            # "open_clip/MobileCLIP-B/datacompdr_lt",
            # "open_clip/MobileCLIP-S1/datacompdr"
        ]

    def tearDown(self) -> None:
        clear_loaded_models()

    def test_open_clip_vectorize(self):
        names = self.open_clip_test_model

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9
        embeddings_reference_file = get_absolute_file_path("embeddings_reference/embeddings_open_clip_python_3_8.json")

        # Load in hardcoded embeddings json file
        with open(embeddings_reference_file, "r") as f:
            embeddings_python_3_8 = json.load(f)

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                for normalize_embeddings in [True, False]:
                    output_v = vectorise(name, sentence, model_properties, device,
                                         normalize_embeddings=normalize_embeddings)

                    assert _check_output_type(output_v)

                    output_m = model.encode(sentence, normalize=normalize_embeddings)

                # Embeddings must match hardcoded python 3.8.20 embeddings
                if isinstance(sentence, str):
                    with self.subTest("Hardcoded Python 3.8 Embeddings Comparison"):
                        try:
                            self.assertEqual(np.allclose(output_m, embeddings_python_3_8[name][sentence], atol=1e-5),
                                            True, f"For model {name} and sentence {sentence}: "
                                                    f"Calculated embedding is {output_m} but "
                                                  f"hardcoded embedding is {embeddings_python_3_8[name][sentence]}")
                        except KeyError:
                            raise KeyError(f"Hardcoded Python 3.8 embeddings not found for "
                                           f"model: {name}, sentence: {sentence} in JSON file: "
                                           f"{embeddings_reference_file}")

                with self.subTest("Model encode vs vectorize"):
                    self.assertEqual(np.allclose(output_m, output_v, atol=eps), True)

            clear_loaded_models()

    def test_load_clip_text_model(self):
        names = self.open_clip_test_model

        device = 'cpu'
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        for name in names:

            model =  _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)

            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode_text(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode_text([text])).sum() < eps

            clear_loaded_models()

    def test_load_clip_image_model(self):
        names = self.open_clip_test_model

        device = 'cpu'
        eps = 1e-9
        images = [TestImageUrls.IMAGE0.value,
                  TestImageUrls.IMAGE1.value,
                  TestImageUrls.IMAGE2.value]

        for name in names:

            model =  _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)

            for image in images:
                assert abs(model.encode(image) - model.encode([image])).sum() < eps
                assert abs(model.encode_image(image) - model.encode([image])).sum() < eps
                assert abs(model.encode(image) - model.encode_image([image])).sum() < eps

            clear_loaded_models()

    def test_cpu_encode_type(self):
        names = self.open_clip_test_model

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'

        for name in names:
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output_v = _convert_tensor_to_numpy(model.encode(sentence, normalize=True))
                assert isinstance(output_v, np.ndarray)

            clear_loaded_models()

    def test_open_clip_embedding_size(self):
        names = self.open_clip_test_model

        device = "cpu"

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]

        for name in names:
            open_clip_properties = _get_open_clip_properties()

            for sentence in sentences:
                output_v = vectorise(name, sentence, get_model_properties_from_registry(name), device, normalize_embeddings=True)
                registered_dimension = open_clip_properties[name]["dimensions"]
                output_dimension = len(output_v[0])

                assert registered_dimension == output_dimension

            clear_loaded_models()

    def test_model_outputs(self):
        names = self.open_clip_test_model

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
        names = self.open_clip_test_model

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

    def test_model_un_normalization(self):
        names = self.open_clip_test_model

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-3

        for name in names:
            # this model always give normalized output
            if name in ['open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k',]:
                continue
            model_properties = get_model_properties_from_registry(name)
            model = _load_model(model_properties['name'], model_properties=model_properties, device=device)

            for sentence in sentences:
                output = model.encode(sentence, normalize=False)
                output = _convert_vectorized_output(output)
                max_output_norm = max(torch.linalg.norm(FloatTensor(output), dim=1))
                min_output_norm = min(torch.linalg.norm(FloatTensor(output), dim=1))

                assert abs(max_output_norm - 1) > eps, f"{name}, {sentence},{max_output_norm}"
                assert abs(min_output_norm - 1) > eps, f"{name}, {sentence},{min_output_norm}"

            clear_loaded_models()

    @patch("torch.cuda.amp.autocast")
    def test_autocast_called_when_cuda(self, mock_autocast):
        names = self.open_clip_test_model
        contents = ['this is a test sentence. so is this.', TestImageUrls.IMAGE0.value]
        for model_name in names:
            for content in contents:
                vectorise(model_name=model_name, content=content, device="cpu")
                mock_autocast.assert_not_called()
