import unittest
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.s2_inference.errors import InvalidModelSettingsError
from marqo.s2_inference.model_registry import load_model_properties
from marqo.client import Client

from marqo.s2_inference.s2_inference import (
    vectorise,
    _get_model_name,
    _check_model_dict,
    _update_model_dict,
    _update_model_properties
    )

from tests.marqo_test import MarqoTestCase


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self):
        mq = Client(**self.client_settings)
        self.endpoint = mq.config.url
        self.config = mq.config
        self.client = mq

        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass


    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass


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
        """_check_model_dict should throw an exception if required keys are not given.
        """
        model = {# "name": "sentence-transformers/all-mpnet-base-v2",
                # "dimensions": 768,
                "tokens":128,
                "type":"sbert"}

        self.assertRaises(InvalidModelSettingsError, _check_model_dict, model)


        """_check_model_dict should not throw an exception if required keys are given.
        """
        model['dimensions'] = 768
        model['name'] = "sentence-transformers/all-mpnet-base-v2"

        self.assertEqual(_check_model_dict(model), True)


    def test_update_model_dict(self):
        """If optional keys are not given, _update_model_dict should add the keys with default values.
        """
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
        """If model info is not in MODEL_PROPERTIES, _update_model_properties should add it
        """
        model = {"name": "random-model-name",
                "dimensions": 768,
                "tokens": 128,
                "type":"sbert"
                }
        model_name = _get_model_name(model)

        TEST_MODEL_PROPERTIES = load_model_properties()

        _update_model_properties(model_name, model, TEST_MODEL_PROPERTIES)

        self.assertEqual(TEST_MODEL_PROPERTIES['models'][model_name], model)


    def test_vectorise_accepts_dict(self):
        model = {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "dimensions": 768,
                "tokens": 128,
                "type":"sbert"}

        result = vectorise(model, "some string")
