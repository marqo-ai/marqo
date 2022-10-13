import json
import pprint
import requests
from marqo.tensor_search.enums import IndexSettingsField
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.tensor_search import tensor_search
from marqo.tensor_search import configs
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField


class TestCreateIndex(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_create_vector_index_default_index_settings(self):
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass
        # test that index is deleted:
        try:
            tensor_search.search(config=self.config, index_name=self.index_name_1, text="some text")
            raise AssertionError
        except IndexNotFoundError as e:
            pass
        self.client.create_index(index_name=self.index_name_1)
        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        
        assert settings[self.index_name_1]["mappings"]["_meta"][IndexSettingsField.index_settings] \
            == tensor_search.configs.get_default_index_settings()

    def test_create_vector_index_custom_index_settings(self):
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass
        # test that index is deleted:
        try:
            tensor_search.search(config=self.config, index_name=self.index_name_1, text="some text")
            raise AssertionError
        except IndexNotFoundError as e:
            pass
        custom_settings = {
            IndexSettingsField.treat_urls_and_pointers_as_images: True,
            IndexSettingsField.normalize_embeddings: False
        }
        self.client.create_index(index_name=self.index_name_1, **custom_settings)
        settings = requests.get(
            url=self.endpoint + "/" + self.index_name_1 + "/_mapping",
            verify=False
        ).json()
        pprint.pprint(settings)
        retrieved_settings = settings[self.index_name_1]["mappings"]["_meta"][IndexSettingsField.index_settings]
        del retrieved_settings[IndexSettingsField.index_defaults][IndexSettingsField.model]

        default_settings = tensor_search.configs.get_default_index_settings()
        default_text_preprocessing = default_settings[IndexSettingsField.index_defaults][IndexSettingsField.text_preprocessing]
        default_image_preprocessing = default_settings[IndexSettingsField.index_defaults][IndexSettingsField.image_preprocessing]

        assert retrieved_settings == {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.treat_urls_and_pointers_as_images: True,
                IndexSettingsField.normalize_embeddings: False,
                IndexSettingsField.text_preprocessing: default_text_preprocessing,
                IndexSettingsField.image_preprocessing: default_image_preprocessing
            },
            IndexSettingsField.number_of_shards: default_settings[IndexSettingsField.number_of_shards]
        }

    def test__autofill_index_settings_fill_missing_text_preprocessing(self):
        modified_settings = tensor_search.configs.get_default_index_settings()
        del modified_settings[IndexSettingsField.index_defaults][IndexSettingsField.text_preprocessing]\
            [IndexSettingsField.split_method]
        assert IndexSettingsField.split_method not in \
               modified_settings[IndexSettingsField.index_defaults][IndexSettingsField.text_preprocessing]
        assert tensor_search._autofill_index_settings(modified_settings) \
            == tensor_search.configs.get_default_index_settings()

    def test_default_number_of_shards(self):
        """does an index get created with the default number of shards?"""
        tensor_search.create_vector_index(index_name=self.index_name_1, config=self.config)
        default_shard_count = configs.get_default_index_settings()[NsField.number_of_shards]
        assert default_shard_count is not None
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            verify=False
        )
        assert default_shard_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_shards'])

    def test_autofill_number_of_shards(self):
        """ does it work if other params are filled?"""
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "ViT-B/32",
                }}
        )
        default_shard_count = configs.get_default_index_settings()[NsField.number_of_shards]
        assert default_shard_count is not None
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            headers=self.generic_header,
            verify=False
        )
        assert default_shard_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_shards'])

    def test_set_number_of_shards(self):
        """ does it work if other params are filled?"""
        intended_shard_count = 6
        res_0 = tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "ViT-B/32",
                },
                NsField.number_of_shards: intended_shard_count
            }
        )
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            headers=self.generic_header,
            verify=False
        )
        assert intended_shard_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_shards'])
