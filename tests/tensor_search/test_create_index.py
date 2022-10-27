import pprint
import requests
from marqo.tensor_search.enums import IndexSettingsField, EnvVars
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.tensor_search import tensor_search
from marqo.tensor_search import configs
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField
from unittest import mock
from marqo import errors

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
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_create_vector_index_default_index_settings(self):
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
        # test that index is deleted:
        try:
            tensor_search.search(config=self.config, index_name=self.index_name_1, text="some text")
            raise AssertionError
        except IndexNotFoundError as e:
            pass
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        
        assert settings[self.index_name_1]["mappings"]["_meta"][IndexSettingsField.index_settings] \
            == tensor_search.configs.get_default_index_settings()

    def test_create_vector_index_custom_index_settings(self):
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
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
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings={
                NsField.index_defaults: custom_settings})
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
        default_video_preprocessing = default_settings[IndexSettingsField.index_defaults][IndexSettingsField.video_preprocessing]

        assert retrieved_settings == {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.treat_urls_and_pointers_as_images: True,
                IndexSettingsField.normalize_embeddings: False,
                IndexSettingsField.text_preprocessing: default_text_preprocessing,
                IndexSettingsField.image_preprocessing: default_image_preprocessing,
                IndexSettingsField.video_preprocessing: default_video_preprocessing,
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

    def test_field_limits(self):
        index_limits = [1, 5, 10, 100, 1000]
        for lim in index_limits:
            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as s:
                pass
            mock_read_env_vars = mock.MagicMock()
            mock_read_env_vars.return_value = lim

            @mock.patch("os.environ", {EnvVars.MARQO_MAX_INDEX_FIELDS: str(lim)})
            def run():
                res_1 = tensor_search.add_documents(
                    index_name=self.index_name_1, docs=[
                        {f"f{i}": "some content" for i in range(lim)},
                        {"_id": "1234", **{f"f{i}": "new content" for i in range(lim)}},
                    ],
                    auto_refresh=True, config=self.config
                )
                assert not res_1['errors']
                res_1_2 = tensor_search.add_documents(
                    index_name=self.index_name_1, docs=[
                        {'f0': 'this is fine, but there is no resiliency.'},
                        {f"f{i}": "some content" for i in range(lim // 2 + 1)},
                        {'f0': 'this is fine. Still no resilieny.'}
                    ],
                    auto_refresh=True, config=self.config
                )
                assert not res_1_2['errors']
                try:
                    res_2 = tensor_search.add_documents(
                        index_name=self.index_name_1, docs=[
                            {'fx': "blah"}
                        ], auto_refresh=True, config=self.config
                    )
                    raise AssertionError
                except errors.IndexMaxFieldsError:
                    pass
                return True
            assert run()

    def test_field_limit_non_text_types(self):
        @mock.patch("os.environ", {EnvVars.MARQO_MAX_INDEX_FIELDS: "5"})
        def run():
            docs = [
                {"f1": "fgrrvb", "f2": 1234, "f3": 1.4, "f4": "hello hello", "f5": False, "_id": "hehehehe"},
                {"f1": "erf1f", "f2": 934, "f3": 4.0, "f4": "my name", "f5": True},
                {"f1": "water is healthy", "f5": True},
                {"f2": 49, "f3": 400.4, "f4": "alien message"}
            ]
            res_1 = tensor_search.add_documents(
                index_name=self.index_name_1, docs=docs, auto_refresh=True, config=self.config
            )
            assert not res_1['errors']
            try:
                res_2 = tensor_search.add_documents(
                    index_name=self.index_name_1, docs=[
                        {'fx': "blah"}
                    ], auto_refresh=True, config=self.config
                )
                raise AssertionError
            except errors.IndexMaxFieldsError:
                pass
            return True

        assert run()

    def test_field_Limit_none_env_var(self):
        """When the limit env var is undefined: we need to manually test it,
        as the testing environment may have this env var defined."""
        mock_read_env_vars = mock.MagicMock()
        mock_read_env_vars.return_value = None

        @mock.patch("marqo.tensor_search.utils.read_env_vars_and_defaults", mock_read_env_vars)
        def run():
            docs = [
                {"f1": "fgrrvb", "f2": 1234, "f3": 1.4, "f4": "hello hello", "f5": False},
                {"f1": "erf1f", "f2": 934, "f3": 4.0, "f4": "my name", "f5": True},
                {"f1": "water is healthy", "f5": True},
                {"f2": 49, "f3": 400.4, "f4": "alien message", "_id": "rkjn"}
            ]
            res_1 = tensor_search.add_documents(
                index_name=self.index_name_1, docs=docs, auto_refresh=True, config=self.config
            )
            mapping_info = requests.get(
                self.authorized_url + f"/{self.index_name_1}/_mapping",
                verify=False
            )
            assert not res_1['errors']
            return True
        assert run()

