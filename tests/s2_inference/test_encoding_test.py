import unittest

from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise, 
    _convert_vectorized_output, 
    available_models,
    clear_loaded_models,
    _load_model
    )

from torch import FloatTensor
import numpy as np

class TestTestModelOutputs(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def test_load_test_text_model(self):
        names = ["test"]
        device = 'cpu'
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']
        
        for name in names:
        
            model = _load_model(name, device=device)
            
            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode(text)).sum() < eps
                assert model.encode(text).shape[-1] == model.embedding_dimension

    def test_check_output(self):
        texts = ['a', ['a'], ['a', 'b', 'longer text. with more stuff']]
        model = _load_model('test')

        for text in texts:
            output = model.encode(text)
            assert _check_output_type(_convert_vectorized_output(output))

