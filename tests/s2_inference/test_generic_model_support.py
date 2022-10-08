import unittest
from marqo.s2_inference.errors import InvalidModelSettingsError
from marqo.s2_inference.model_registry import load_model_properties

from marqo.s2_inference.s2_inference import (
    vectorise,
    _get_model_name,
    _check_model_dict,
    _update_model_dict,
    _update_model_properties
    )

from tests.marqo_test import MarqoTestCase


'''
Temporary test file, will refactor it for unit testing purposes later on
'''

class TestGenericModelSupport(unittest.TestCase):

    def setUp(self):
        #giving an error??
        # mq = Client(**self.client_settings)
        # self.config = mq.config
        # self.client = mq

        self.index_name_1 = "my-test-hf-index-1"


    def test_get_model_name_with_str_input(self):
        model = "sentence-transformers/all-mpnet-base-v2"
        model_name = _get_model_name(model)

        assert model_name == "sentence-transformers/all-mpnet-base-v2"


    def test_get_model_name_with_dict_input(self):
        model = {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert"}

        model_name = _get_model_name(model)

        assert model_name == "sentence-transformers/all-mpnet-base-v2"


    def test_check_model_dict(self):
        model = {# "name": "sentence-transformers/all-mpnet-base-v2",
                # "dimensions": 768,
                "tokens":128,
                "type":"sbert"}


        self.assertRaises(InvalidModelSettingsError, _check_model_dict, model)

        model['dimensions'] = 768
        model['name'] = "sentence-transformers/all-mpnet-base-v2"

        self.assertEqual(_check_model_dict(model), True)


    def test_update_model_dict(self):
        model = {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768
                # "tokens": 128,
                # "type":"sbert"
                }

        updated_model_dict = _update_model_dict(model)
        default_tokens_value = updated_model_dict.get('tokens')
        default_type_value = updated_model_dict.get('type')

        self.assertEqual(default_tokens_value, 128)
        self.assertEqual(default_type_value, "sbert")


    def test_update_model_properties(self):
        model = {"name": "random-model-name",
                "dimensions": 768,
                "tokens": 128,
                "type":"sbert"
                }
        model_name = _get_model_name(model)

        TEST_MODEL_PROPERTIES = load_model_properties()

        _update_model_properties(model, model_name, TEST_MODEL_PROPERTIES)

        self.assertEqual(TEST_MODEL_PROPERTIES['models'][model_name], model)


    def test_vectorise_accepts_dict(self):
        model = {"name": "random-model-name",
                "dimensions": 768,
                "tokens": 128,
                "type":"sbert"}

        result = vectorise(model, "some string")


    # def _delete_index(self, index_name: str):
    #     try:
    #         tensor_search.delete_index(config=self.config, index_name=index_name)
    #         print("Index successfully deleted.")
    #     except Exception as e:
    #         print(e)
    #         print("Index does not exist.")
    #
    #
    # def _create_index(self, index_name: str, settings):
    #     try:
    #         tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, **settings)
    #         print("index created")
    #     except Exception as e:
    #         print(e)
    #         print("index already exists")
