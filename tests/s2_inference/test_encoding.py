import unittest
import os
import torch

from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models
from marqo.s2_inference.model_registry import load_model_properties, _get_open_clip_properties, _get_XCLIP_properties
import numpy as np
from PIL import Image
from marqo.s2_inference.s2_inference import (
    _load_model,
    _check_output_type, vectorise,
    _convert_vectorized_output,
)


class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def test_vectorize(self):

        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot',
                 "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output_v = vectorise(name, sentence, device, normalize_embeddings=True)

                assert _check_output_type(output_v)

                output_m = model.encode(sentence, normalize=True)

                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps

    def test_load_clip_text_model(self):
        names = ['RN50', "ViT-B/16"]
        device = 'cpu'
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        for name in names:

            model = _load_model(name, device=device)

            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode_text(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode_text([text])).sum() < eps

    def test_load_sbert_text_model(self):
        names = ["sentence-transformers/all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6"]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model = _load_model(name, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

    def test_load_hf_text_model(self):
        names = ["hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6"]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model = _load_model(name, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

    def test_load_onnx_sbert_text_model(self):
        names = ["onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        device = 'cpu'
        eps = 1e-9

        for name in names:
            model = _load_model(name, device=device)
            assert abs(model.encode('hello') - model.encode(['hello'])).sum() < eps

    def test_compare_onnx_sbert_text_models(self):
        names = ["all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.']
        device = 'cpu'
        eps = 1e-4

        for name in names:
            for sentence in sentences:
                model_onnx = _load_model(os.path.join('onnx', name), device=device)

                model_sbert = _load_model(name, device=device)

                assert abs(model_onnx.encode(sentence) - model_sbert.encode(sentence)).sum() < eps

    def test_model_outputs(self):
        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot',
                 'open_clip/ViT-B-32/laion400m_e32', "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6",
                 "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1",
                 "onnx/all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output = model.encode(sentence)
                assert _check_output_type(_convert_vectorized_output(output))

    def test_model_normalization(self):
        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot',
                 'open_clip/ViT-B-32/laion400m_e32', 'RN50', "ViT-B/16", "all-MiniLM-L6-v1",
                 "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6",
                 "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-6

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output = model.encode(sentence, normalize=True)
                output = _convert_vectorized_output(output)
                max_output_norm = max(torch.linalg.norm(FloatTensor(output), dim=1))
                min_output_norm = min(torch.linalg.norm(FloatTensor(output), dim=1))

                assert abs(max_output_norm - 1) < eps, f"{name}, {sentence}"
                assert abs(min_output_norm - 1) < eps, f"{name}, {sentence}"

    def test_model_un_normalization(self):
        # note: sbert native seems to provide normalized embeddings even with = False, needs more investigation
        # , 
        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot',
                 'open_clip/ViT-B-32/laion400m_e32', 'RN50', "ViT-B/16", "hf/all-MiniLM-L6-v1",
                 "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-3

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output = model.encode(sentence, normalize=False)
                output = _convert_vectorized_output(output)
                max_output_norm = max(torch.linalg.norm(FloatTensor(output), dim=1))
                min_output_norm = min(torch.linalg.norm(FloatTensor(output), dim=1))

                assert abs(max_output_norm - 1) > eps, f"{name}, {sentence}"
                assert abs(min_output_norm - 1) > eps, f"{name}, {sentence}"

    def test_open_clip_vectorize(self):

        names = ['open_clip/ViT-B-32/laion400m_e32', 'open_clip/RN50/openai']

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output_v = vectorise(name, sentence, device, normalize_embeddings=True)

                assert _check_output_type(output_v)

                output_m = model.encode(sentence, normalize=True)

                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps

    def test_open_clip_embedding_size(self):

        # This is a full test as the list includes all the models. Note that the training dataset does not affect the
        # embedding size.
        names = ['open_clip/ViT-B-32/laion400m_e32', 'open_clip/RN50/openai']

        device = "cpu"

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]

        for name in names:
            open_clip_properties = _get_open_clip_properties()

            for sentence in sentences:
                output_v = vectorise(name, sentence, device, normalize_embeddings=True)
                registered_dimension = open_clip_properties[name]["dimensions"]
                output_dimension = len(output_v[0])

                assert registered_dimension == output_dimension

    def test_XCLIP_vectorize(self):

        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot']

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'
        eps = 1e-9

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output_v = vectorise(name, sentence, device, normalize_embeddings=True)

                assert _check_output_type(output_v)

                output_m = model.encode(sentence, normalize=True)

                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps

    def test_XCLIP_embedding_size(self):

        # This is a full test as the list includes all the models. Note that the training dataset does not affect the
        # embedding size.
        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot']

        device = "cpu"

        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]

        for name in names:
            x_clip_properties = _get_XCLIP_properties()

            for sentence in sentences:
                output_v = vectorise(name, sentence, device, normalize_embeddings=True)
                registered_dimension = x_clip_properties[name]["dimensions"]
                output_dimension = len(output_v[0])

                assert registered_dimension == output_dimension


    def test_XCLIP_vectorize(self):

        names = ['microsoft/xclip-base-patch16-kinetics-600', 'microsoft/xclip-base-patch16-zero-shot']

        video_array = np.random.rand(32, 224, 224, 3) * 255
        video_random = [Image.fromarray(i.astype("uint8")).convert("RGB") for i in video_array]

        videos = [video_random, video_random, [video_random, video_random]]
        device = 'cpu'
        eps = 1e-9

        for name in names:

            model = _load_model(name, device=device)

            for video in videos:
                output_v = vectorise(name, video, device, normalize_embeddings=True)

                assert _check_output_type(output_v)

                output_m = model.encode(video, normalize=True)

                assert abs(torch.FloatTensor(output_m) - torch.FloatTensor(output_v)).sum() < eps