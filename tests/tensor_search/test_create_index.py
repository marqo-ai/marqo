import pprint
from typing import Any, Dict
import pytest
import os
import requests
from marqo.tensor_search.enums import IndexSettingsField, EnvVars
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.tensor_search import tensor_search, configs, backend
from marqo.tensor_search.utils import read_env_vars_and_defaults
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField
from unittest import mock
from marqo import errors

class TestCreateIndex(MarqoTestCase):

    def setUp(self, custom_index_defaults: Dict[str, Any] = {}) -> None:
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

    def test_create_vector_index__invalid_settings(self):
        custom_index_defaults = [
            {IndexSettingsField.ann_parameters: {
                    IndexSettingsField.ann_method: "fancy-new-ann-method",
            }},
            {IndexSettingsField.ann_parameters: {
                    IndexSettingsField.ann_method: "ivf",
            }},
            {IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_engine: "faiss",
            }},
            {IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_metric: "innerproduct",
            }},
            {IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_method_parameters: {
                IndexSettingsField.hnsw_ef_construction: 0,
                IndexSettingsField.hnsw_m: 16
            }}},
            {IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_method_parameters: {
                IndexSettingsField.hnsw_ef_construction: 128,
                IndexSettingsField.hnsw_m: 101
            }}},
            {IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_method_parameters: {
                IndexSettingsField.hnsw_ef_construction: 1 + int(read_env_vars_and_defaults(EnvVars.MARQO_EF_CONSTRUCTION_MAX_VALUE)),
                IndexSettingsField.hnsw_m: 16
            }}},
            {IndexSettingsField.ann_parameters: {IndexSettingsField.ann_method_parameters: {
                IndexSettingsField.hnsw_ef_construction: 128,
                IndexSettingsField.hnsw_m: -1
            }}},
        ]
        for idx_defaults in custom_index_defaults:
            with self.subTest(custom_index_defaults=idx_defaults):
                try:
                    tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
                except IndexNotFoundError as s:
                    pass
                
                with self.assertRaises(errors.InvalidArgError):
                    print(f"index settings={idx_defaults}")
                    tensor_search.create_vector_index(
                        config=self.config,
                        index_name=self.index_name_1,
                        index_settings={
                            NsField.index_defaults: idx_defaults
                        }
                    )
                    print(tensor_search.get_index_info(self.config, self.index_name_1))

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

        assert retrieved_settings == {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.treat_urls_and_pointers_as_images: True,
                IndexSettingsField.normalize_embeddings: False,
                IndexSettingsField.text_preprocessing: default_text_preprocessing,
                IndexSettingsField.image_preprocessing: default_image_preprocessing,
                IndexSettingsField.ann_parameters: {
                    IndexSettingsField.ann_engine: 'lucene',
                    IndexSettingsField.ann_method_name: 'hnsw',
                    IndexSettingsField.ann_method_parameters: {
                        IndexSettingsField.hnsw_ef_construction: 128,
                        IndexSettingsField.hnsw_m: 16
                    },
                    IndexSettingsField.ann_metric: 'cosinesimil'
                },
            },
            IndexSettingsField.number_of_shards: default_settings[IndexSettingsField.number_of_shards],
            IndexSettingsField.number_of_replicas: default_settings[IndexSettingsField.number_of_replicas]
        }

    def test_create_vector_index_default_knn_settings(self):
        """Do the Marqo-OS settings correspond to the Marqo index settings? For default HNSW params"""
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
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"Title": "wowow"}], auto_refresh=True)
        mappings = requests.get(
            url=self.endpoint + "/" + self.index_name_1 + "/_mapping",
            verify=False
        ).json()
        params = mappings[self.index_name_1]['mappings']['properties']['__chunks']['properties']['__vector_Title']['method']
        assert params['engine'] == 'lucene'
        assert params['space_type'] == 'cosinesimil'
        assert params['parameters'] == {'ef_construction': 128, 'm': 16}

    def test_create_vector_index_custom_knn_settings(self):
        """Do the Marqo-OS settings correspond to the Marqo index settings? For custom HNSW params"""
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
            IndexSettingsField.normalize_embeddings: False,
            IndexSettingsField.ann_parameters: {
                IndexSettingsField.ann_metric: "l2",
                IndexSettingsField.ann_method_parameters: {
                    IndexSettingsField.hnsw_m: 17,
                    IndexSettingsField.hnsw_ef_construction: 133,
                }
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings={
                NsField.index_defaults: custom_settings})
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"Title": "wowow"}], auto_refresh=True)
        mappings = requests.get(
            url=self.endpoint + "/" + self.index_name_1 + "/_mapping",
            verify=False
        ).json()
        params = mappings[self.index_name_1]['mappings']['properties']['__chunks']['properties']['__vector_Title']['method']
        assert params['engine'] == 'lucene'
        assert params['space_type'] == 'l2'
        assert params['parameters'] == {'ef_construction': 133, 'm': 17}

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

    def test_default_number_of_replicas(self):
        tensor_search.create_vector_index(index_name=self.index_name_1, config=self.config)
        default_replicas_count = configs.get_default_index_settings()[NsField.number_of_replicas]
        assert default_replicas_count is not None
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            verify=False
        )
        assert default_replicas_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_replicas'])

    def test_autofill_number_of_replicas(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "ViT-B/32",
                }}
        )
        default_replicas_count = configs.get_default_index_settings()[NsField.number_of_replicas]
        assert default_replicas_count is not None
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            headers=self.generic_header,
            verify=False
        )
        assert default_replicas_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_replicas'])
    
    def test_autofill_index_defaults(self):
        """ Does autofill work as intended when index_defaults is not set, but number_of_shards is?"""
        intended_shard_count = 6
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                NsField.number_of_shards: intended_shard_count        
            }
        )
        
        default_index_defaults = configs.get_default_index_settings()[NsField.index_defaults]
        assert default_index_defaults is not None
        
        index_info = backend.get_index_info(config=self.config, index_name=self.index_name_1)
        test_index_defaults = index_info.index_settings[NsField.index_defaults]

        assert default_index_defaults == test_index_defaults

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

    def test_set_number_of_replicas(self):
        """ does it work if other params are filled?"""
        intended_replicas_count = 4
        res_0 = tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "ViT-B/32",
                },
                NsField.number_of_replicas: intended_replicas_count
            }
        )
        resp = requests.get(
            url=self.authorized_url + f"/{self.index_name_1}",
            headers=self.generic_header,
            verify=False
        )
        assert intended_replicas_count == int(resp.json()[self.index_name_1]['settings']['index']['number_of_replicas'])

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

    def test_create_index_protected_name(self):
        try:
            tensor_search.create_vector_index(config=self.config, index_name='.opendistro_security')
            raise AssertionError
        except errors.InvalidIndexNameError:
            pass

    def test_index_validation_bad(self):
        bad_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": "2",
                    "split_overlap": "0",
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas": 1
        }
        try:
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=bad_settings)
            raise AssertionError
        except errors.InvalidArgError as e:
            pass

    def test_index_validation_good(self):
        good_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas": 1
        }
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=good_settings)
