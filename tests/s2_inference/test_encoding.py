import unittest
import os
import torch

from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models, get_model_properties_from_registry
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties
import numpy as np

from marqo.s2_inference.s2_inference import (
    _load_model,
    _check_output_type, vectorise,
    _convert_vectorized_output,
)


class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def tearDown(self) -> None:
        clear_loaded_models()

    def test_vectorize(self):

        names = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                 "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k',"all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
        names = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                 "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k', 'RN50', "ViT-B/16"]
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
        names = ["hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6"]
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
        names = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k', "onnx32/openai/ViT-L/14", "onnx16/openai/ViT-L/14",
                'open_clip/ViT-B-32/laion400m_e32', "all-MiniLM-L6-v1",
                 "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1",
                 "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
        names = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                 "onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k', "onnx32/openai/ViT-L/14", "onnx16/openai/ViT-L/14",
                 'open_clip/ViT-B-32/laion400m_e32', 'RN50', "ViT-B/16", "all-MiniLM-L6-v1",
                 "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
        names = ['open_clip/ViT-B-32/laion400m_e32', 'RN50', "ViT-B/16", "hf/all-MiniLM-L6-v1",
                 "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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


    def test_open_clip_vectorize(self):

        names = ['open_clip/ViT-B-32/laion400m_e32', 'open_clip/RN50/openai']

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

            clear_loaded_models()


    def test_open_clip_embedding_size(self):

        # This is a full test as the list includes all the models. Note that the training dataset does not affect the
        # embedding size.
        names = ['open_clip/ViT-B-32/laion400m_e32', 'open_clip/RN50/openai']

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


    def test_onnx_clip_vectorise(self):

        names = ["onnx16/open_clip/ViT-B-32/laion400m_e32", 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k', "onnx32/openai/ViT-L/14", "onnx16/openai/ViT-L/14"]

        sentences = ['hello', 'this is a test sentence. so is this.',
                     ['hello', 'this is a test sentence. so is this.']]
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

            clear_loaded_models()


    def test_multilingual_clip_performance(self):

        clear_loaded_models()

        names = ["multilingual-clip/XLM-Roberta-Large-Vit-L-14", "multilingual-clip/XLM-R Large Vit-B/16+",
                 "multilingual-clip/XLM-Roberta-Large-Vit-B-32", "multilingual-clip/LABSE-Vit-L-14"]
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