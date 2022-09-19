import unittest
import os
import torch

from marqo.s2_inference.types import FloatTensor
from marqo.s2_inference.s2_inference import clear_loaded_models

from marqo.s2_inference.s2_inference import (
    _load_model,
    _check_output_type, vectorise, 
    _convert_vectorized_output, 
    )

class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def test_vectorize(self):
        
        names = ["all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
                model_onnx = _load_model(os.path.join('onnx',name), device=device)

                model_sbert = _load_model(name, device=device)

                assert abs(model_onnx.encode(sentence) - model_sbert.encode(sentence)).sum() < eps

    def test_model_outputs(self):
        names = ["all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
        sentences = ['hello', 'this is a test sentence. so is this.', ['hello', 'this is a test sentence. so is this.']]
        device = 'cpu'

        for name in names:

            model = _load_model(name, device=device)

            for sentence in sentences:
                output = model.encode(sentence)
                assert _check_output_type(_convert_vectorized_output(output))

    def test_model_normalization(self):
        names = ['RN50', "ViT-B/16", "all-MiniLM-L6-v1", "all_datasets_v4_MiniLM-L6", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
        names = ['RN50', "ViT-B/16", "hf/all-MiniLM-L6-v1", "hf/all_datasets_v4_MiniLM-L6", "onnx/all-MiniLM-L6-v1", "onnx/all_datasets_v4_MiniLM-L6"]
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
