from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError, UnknownModelError
from marqo.s2_inference.errors import InvalidModelSettingsError
from marqo.tensor_search import tensor_search
from marqo.client import Client

from marqo.s2_inference.s2_inference import (
    vectorise,
    _validate_model_properties,
)

from tests.marqo_test import MarqoTestCase


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self):
        mq = Client(**self.client_settings)
        self.config = mq.config
        self.index_name_1 = "my-test-create-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

    def tearDown(self) -> None:
        pass

    def test_create_index_with_custom_model_properties(self):
        """index should get created with custom model_properties
        """
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': "test-model",
                    'model_properties': {"name": "sentence-transformers/all-mpnet-base-v2",
                                         "dimensions": 768,
                                         "tokens": 128,
                                         "type": "sbert"}
                }
            }
        )

    def test_add_documents_with_custom_model_properties(self):
        """add_documents should throw error if model is not in registry,
        and model_properties have not been given in index
        """
        model_name = "test-model"
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name
                }
            }
        )

        config = self.config
        index_name = self.index_name_1
        docs = [
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }]
        auto_refresh = True

        self.assertRaises(UnknownModelError, tensor_search.add_documents, config, index_name, docs, auto_refresh)

    def test_create_index_with_model_properties_without_model_name(self):
        """
        bert becomes default name but what if settings are different
        """

    def test_validate_model_properties_missing_required_keys(self):
        """_validate_model_properties should throw an exception if required keys are not given.
        """
        model_name = "test-model"
        model_properties = {  # "name": "sentence-transformers/all-mpnet-base-v2",
            # "dimensions": 768,
            "tokens": 128,
            "type": "sbert"}

        self.assertRaises(InvalidModelSettingsError, _validate_model_properties, model_name, model_properties)

        """_validate_model_properties should not throw an exception if required keys are given.
        """
        model_properties['dimensions'] = 768
        model_properties['name'] = "sentence-transformers/all-mpnet-base-v2"

        validated_model_properties = _validate_model_properties(model_name, model_properties)

        self.assertEqual(validated_model_properties, model_properties)

    def test_validate_model_properties_missing_optional_keys(self):
        """If optional keys are not given, _validate_model_properties should add the keys with default values.
        """
        model_name = 'test-model'
        model_properties = {"name": "sentence-transformers/all-mpnet-base-v2",
                            "dimensions": 768
                            # "tokens": 128,
                            # "type":"sbert"
                            }

        validated_model_properties = _validate_model_properties(model_name, model_properties)
        default_tokens_value = validated_model_properties.get('tokens')
        default_type_value = validated_model_properties.get('type')

        self.assertEqual(default_tokens_value, 128)
        self.assertEqual(default_type_value, "sbert")

    def test_validate_model_properties_missing_properties(self):
        """If model_properties is None _validate_model_properties should use model_registry properties
        """
        model_name = 'test'
        registry_test_model_properties = {"name": "sentence-transformers/all-MiniLM-L6-v1",
                                          "dimensions": 16,
                                          "tokens": 128,
                                          "type": "test",
                                          "notes": ""}

        validated_model_properties = _validate_model_properties(model_name=model_name, model_properties=None)

        self.assertEqual(registry_test_model_properties, validated_model_properties)

    def test_vectorise_accepts_dict(self):
        model_name = "test-model"

        # this model is not in model_registry
        model_properties = {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                            "dimensions": 768,
                            "tokens": 128,
                            "type": "sbert"}

        result = vectorise(model_name=model_name, model_properties=model_properties, content="some string")
