import pprint

import requests
from marqo.neural_search.enums import NeuralSettingsField
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError
from marqo.neural_search import neural_search
import unittest
import copy
from tests.marqo_test import MarqoTestCase


class TestCreateIndex(MarqoTestCase):

    def setUp(self) -> None:
        mq = Client(**self.client_settings)
        self.endpoint = mq.config.url
        self.config = mq.config
        self.client = mq

        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_create_vector_index_default_neural_settings(self):
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass
        # test that index is deleted:
        try:
            neural_search.search(config=self.config, index_name=self.index_name_1, text="some text")
            raise AssertionError
        except MarqoError as e:
            assert "no such index" in str(e)
        self.client.create_index(index_name=self.index_name_1)
        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        assert settings[self.index_name_1]["mappings"]["_meta"][NeuralSettingsField.neural_settings] \
            == neural_search.configs.get_default_neural_index_settings()

    def test_create_vector_index_custom_neural_settings(self):
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass
        # test that index is deleted:
        try:
            neural_search.search(config=self.config, index_name=self.index_name_1, text="some text")
            raise AssertionError
        except MarqoError as e:
            assert "no such index" in str(e)
        custom_settings = {
            NeuralSettingsField.treat_urls_and_pointers_as_images: True,
            NeuralSettingsField.normalize_embeddings: False
        }
        self.client.create_index(index_name=self.index_name_1, **custom_settings)
        settings = requests.get(
            url=self.endpoint + "/" + self.index_name_1 + "/_mapping",
            verify=False
        ).json()
        pprint.pprint(settings)
        retrieved_settings = settings[self.index_name_1]["mappings"]["_meta"][NeuralSettingsField.neural_settings]
        del retrieved_settings[NeuralSettingsField.index_defaults][NeuralSettingsField.model]

        default_settings = neural_search.configs.get_default_neural_index_settings()
        default_text_preprocessing = default_settings[NeuralSettingsField.index_defaults][NeuralSettingsField.text_preprocessing]
        assert retrieved_settings == {
            NeuralSettingsField.index_defaults: {
                NeuralSettingsField.treat_urls_and_pointers_as_images: True,
                NeuralSettingsField.normalize_embeddings: False,
                NeuralSettingsField.text_preprocessing: default_text_preprocessing
        }}

    def test__autofill_neural_settings_fill_missing_text_preprocessing(self):
        modified_settings = neural_search.configs.get_default_neural_index_settings()
        del modified_settings[NeuralSettingsField.index_defaults][NeuralSettingsField.text_preprocessing]\
            [NeuralSettingsField.split_method]
        assert NeuralSettingsField.split_method not in \
               modified_settings[NeuralSettingsField.index_defaults][NeuralSettingsField.text_preprocessing]
        assert neural_search._autofill_neural_settings(modified_settings) \
            == neural_search.configs.get_default_neural_index_settings()
