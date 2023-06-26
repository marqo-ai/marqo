import unittest
import functools
from marqo.s2_inference.s2_inference import (
    _check_output_type, vectorise, 
    _convert_vectorized_output, 
    available_models,
    clear_loaded_models,
    get_model_properties_from_registry,
    )
from marqo.s2_inference.s2_inference import _load_model as og_load_model
_load_model = functools.partial(og_load_model, calling_func = "unit_test")


from torch import FloatTensor
import numpy as np

class TestRandomOutputs(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def test_load_random_text_model(self):
        names = ['random', 'random/small', 'random/medium', 'random/large']
        device = 'cpu'
        eps = 1e-9
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']
        
        for name in names:
        
            model = _load_model(name, model_properties=get_model_properties_from_registry(name), device=device)
            
            for text in texts:
                assert abs(model.encode(text) - model.encode([text])).sum() < eps
                assert abs(model.encode(text) - model.encode(text)).sum() < eps
                assert model.encode(text).shape[-1] == model.embedding_dimension

    def test_check_output(self):
        texts = ['a', ['a'], ['a', 'b', 'longer text. with more stuff']]
        model = _load_model('random', model_properties=get_model_properties_from_registry('random'), device="cpu")

        for text in texts:
            output = model.encode(text)
            assert _check_output_type(_convert_vectorized_output(output))

