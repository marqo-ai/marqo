import numpy as np

from marqo.errors import IndexNotFoundError
from marqo.s2_inference.errors import InvalidModelPropertiesError, UnknownModelError, ModelLoadError
from marqo.tensor_search import tensor_search

from marqo.s2_inference.s2_inference import (
    available_models,
    vectorise,
    _validate_model_properties,
    _update_available_models
)

from tests.marqo_test import MarqoTestCase


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

    def tearDown(self) -> None:
        pass

    def test_create_index_with_custom_model_properties(self):
        """index should get created with custom model_properties
        """
        model_name = 'test-model'
        model_properties = {"name": "sentence-transformers/all-mpnet-base-v2",
                            "dimensions": 768,
                            "tokens": 128,
                            "type": "sbert"}
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties
                }
            }
        )

    def test_create_index_with_model_properties_without_model_name(self):
        """create_vector_index should throw an error
            if model_properties are given without model_name
        """
        model_properties = {"name": "sentence-transformers/all-mpnet-base-v2",
                            "dimensions": 768,
                            "tokens": 128,
                            "type": "sbert"}

        index_settings = {
            "index_defaults": {
                # 'model': model_name,
                'model_properties': model_properties
            }
        }

        self.assertRaises(UnknownModelError, tensor_search.create_vector_index, config=self.config,
            index_name=self.index_name_1, index_settings=index_settings)

    def test_add_documents(self):
        """if given the right input, add_documents should work without any throwing any errors
        """
        model_name = "test-model"
        model_properties = {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                            "dimensions": 384,
                            "tokens": 128,
                            "type": "sbert"}
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties
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

        tensor_search.add_documents(config=config, index_name=index_name, docs=docs, auto_refresh=auto_refresh)

    def test_validate_model_properties_missing_required_keys(self):
        """_validate_model_properties should throw an exception if required keys are not given.
        """
        model_name = "test-model"
        model_properties = {  # "name": "sentence-transformers/all-mpnet-base-v2",
            # "dimensions": 768,
            "tokens": 128,
            "type": "sbert"}

        self.assertRaises(InvalidModelPropertiesError, _validate_model_properties, model_name, model_properties)

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

    def test_validate_model_properties_unknown_model_error(self):
        """_validate_model_properties should throw an error if model is not in registry,
            and if model_properties have not been given in index
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

        model_properties = None

        self.assertRaises(UnknownModelError, _validate_model_properties, model_name, model_properties)

    def test_update_available_models_model_load_error(self):
        """_update_available_models should throw an error if model_name given in
            model_properties does not exist
        """
        model_cache_key = "sample-cache-key"
        model_name = "test-model"
        model_properties = {"name": "incorect-model-name",
                            "dimensions": 768,
                            "tokens": 128,
                            "type": "sbert"}
        device = "cpu"
        normalize_embeddings = True

        self.assertRaises(ModelLoadError, _update_available_models, model_cache_key,
            model_name, model_properties, device, normalize_embeddings)

    def test_custom_model_gets_loaded(self):
        model_name = "test-model"

        # this model is not in model_registry
        model_properties = {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                            "dimensions": 384,
                            "tokens": 128,
                            "type": "sbert"}

        result = vectorise(model_name=model_name, model_properties=model_properties, content="some string")

        self.assertEqual(np.array(result).shape[-1], model_properties['dimensions'])

    def test_vectorise_with_default_model_different_properties(self):
        """same models with different properties should return different outputs
        """
        model_name = 'sentence-transformers/all-mpnet-base-v2'
        model_properties_default = {"name": "sentence-transformers/all-mpnet-base-v2",
                                    "dimensions": 768,
                                    "tokens": 128,
                                    "type": "sbert"}

        model_properties_custom = {"name": "sentence-transformers/all-mpnet-base-v2",
                                   "dimensions": 768,
                                   "tokens": 4,
                                   "type": "sbert"}

        content = "A path or url to a tensorflow index checkpoint file (e.g, ./tf_model/model.ckpt.index). In this " \
                  "case, from_tf should be set to True and a configuration object should be provided as config " \
                  "argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model " \
                  "using the provided conversion scripts and loading the PyTorch model afterwards. "

        res_default = vectorise(model_name=model_name, model_properties=model_properties_default, content=content)
        res_custom = vectorise(model_name='custom-model', model_properties=model_properties_custom, content=content)

        self.assertNotEqual(res_default, res_custom)
        # self.assertEqual(np.array(res_default).shape[-1], np.array(res_custom).shape[-1])

    def test_modification_of_model_properties(self):
        """available_models should get updated if the model_properties are modified
            and model_name is unchanged
        """
        model_name = 'test-model-in-registry'
        model_properties = {"name": "sentence-transformers/all-mpnet-base-v2",
                            "dimensions": 768,
                            "tokens": 128,
                            "type": "sbert"}
        index_settings = {
            "index_defaults": {
                'model': model_name,
                'model_properties': model_properties
            }
        }

        tensor_search.create_vector_index(index_name=self.index_name_1,
            config=self.config, index_settings=index_settings
        )

        vectorise(model_name=model_name, model_properties=model_properties, content="some string")
        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)

        old_num_of_available_models = len(available_models)
        model_properties['tokens'] = 256

        tensor_search.create_vector_index(index_name=self.index_name_1,
            config=self.config, index_settings=index_settings
        )
        vectorise(model_name=model_name, model_properties=model_properties, content="some string")

        new_num_of_available_models = len(available_models)

        self.assertEqual(new_num_of_available_models, old_num_of_available_models + 1)
