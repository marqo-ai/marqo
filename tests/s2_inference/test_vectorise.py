import PIL
from marqo.s2_inference import clip_utils, types, random_utils, s2_inference
import unittest
from unittest import mock
import requests


class TestVectorise(unittest.TestCase):

    def test_vectorise_in_batches(self):
        from marqo.s2_inference import random_utils
        mock_model = mock.MagicMock()
        mock_model.encode = mock.MagicMock()

        random_model = random_utils.Random(model_name='mock_model', embedding_dim=128)

        def func(*args,**kwargs):
            print('hi')
            return random_model.encode(*args,**kwargs)

        mock_model.encode.side_effect = func
        mock_model_props = {
            "name": "mock_model",
            "dimensions": random_model.embedding_dimension,
            "tokens": 128,
            "type": "sbert"
        }

        mock_available_models = {
            s2_inference._create_model_cache_key(
                model_name='mock_model', device='cpu',
                model_properties=mock_model_props
            ): mock_model
        }

        @mock.patch('marqo.s2_inference.s2_inference.available_models', mock_available_models)
        @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
        def run():
            s2_inference.vectorise(model_name='mock_model', content=['just a single content'],
                                   model_properties=mock_model_props)

            print('mock_model.encode.call_args_list', mock_model.encode.call_args_list)
            return True
        assert run()
