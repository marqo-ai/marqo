import copy
import json
import pprint
from unittest import mock

import requests
from marqo.tensor_search.enums import TensorField, IndexSettingsField
from marqo.client import Client
from marqo.errors import IndexNotFoundError, InvalidArgError, BadRequestError
from marqo.tensor_search import tensor_search, index_meta_cache
from tests.marqo_test import MarqoTestCase


class TestAddDocuments(MarqoTestCase):

    def setUp(self) -> None:
        mq = Client(**self.client_settings)
        self.endpoint = mq.config.url
        self.config = mq.config
        self.client = mq

        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def _match_all(self, index_name, verbose=True):
        """Helper function"""
        res = requests.get(
            F"{self.endpoint}/{index_name}/_search",
            headers=self.generic_header,
            data=json.dumps({"query": {"match_all": {}}}),
            verify=False
        )
        if verbose:
            pprint.pprint(res.json())
        return res

    def test_add_plain_id_field(self):
        """does a plain 'id' field work (in the doc body)? """
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "id": "abcdefgh",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }], auto_refresh=True)
        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123") == {
                "id": "abcdefgh",
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }

    def test_update_docs_update_chunks(self):
        """Updating a doc needs to update the corresponding chunks"
        """
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }], auto_refresh=True)
        count0_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        )
        count0 = count0_res.json()["count"]
        assert count0 == 1
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }], auto_refresh=True)
        count1_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        )
        count1 = count1_res.json()["count"]
        assert count1 == count0

    def test_implicit_create_index(self):
        r1 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert r1.status_code == 404
        add_doc_res = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        r2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert r2.status_code == 200

    def test_default_index_settings(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "model" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "media_type" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "__field_name" in \
               index_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]

    def test_default_index_settings_implicitly_created(self):
        add_doc_res = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "model" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "media_type" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "__field_name" in \
               index_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]

    def test_add_new_fields_on_the_fly(self):
        add_doc_res = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        cluster_ix_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )

        assert "__vector_abc" in cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
        assert "dimension" in cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]["__vector_abc"]
        add_doc_res = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "1234", "The title book 1": "hahehehe"}], auto_refresh=True
        )
        cluster_ix_info_2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "__vector_abc" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
        assert "__vector_The title book 1" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
        assert "dimension" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]["__vector_The title book 1"]

    def test_add_new_fields_on_the_fly_index_cache_syncs(self):
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        add_doc_res_1 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        index_info_2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].properties[TensorField.chunks]["properties"]["__vector_abc"] \
               == index_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]["__vector_abc"]
        add_doc_res_2 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"cool field": "yep yep", "haha": "heheh"}], auto_refresh=True
        )
        index_info_3 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
               == {k:v for k,v in index_info_3.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"].items()
                   if k.startswith("__vector_")}

    def test_add_multiple_fields(self):
        add_doc_res = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"cool v field": "yep yep", "haha ee": "heheh"}], auto_refresh=True
        )
        cluster_ix_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
               == {k: v for k, v in
                   cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"].items()
                   if k.startswith("__vector_")}
        assert "__vector_cool v field" in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()
        assert "__vector_haha ee" in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()

    def test_add_docs_response_format(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        add_res = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "id": "abcdefgh",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            },
            {
                "_id": "456",
                "id": "abcdefgh",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            },
            {
                "_id": "789",
                "subtitle": [1, 2, 3]
            }
        ], auto_refresh=True)

        assert "errors" in add_res
        assert "processingTimeMs" in add_res
        assert "index_name" in add_res
        assert "items" in add_res

        assert add_res["processingTimeMs"] > 0
        assert isinstance(add_res["errors"], bool)
        assert add_res["index_name"] == self.index_name_1

        for item in add_res["items"]:
            assert "_id" in item
            assert ("result" in item) ^ ("error" in item and "code" in item)
            assert "status" in item

    def test_add_documents_validation(self):
        """These bad docs should raise errors"""
        bad_doc_args = [
            [{
                    "_id": "123",
                    "id": {},
            }],
            [{
                "blahblah": {1243}
            }],
            [{
                "blahblah": None
            }],
            [{
                "blahblah": [None], "hehehe": 123
            },{
                "some other obj": "finnne"
            }],
            [{
                "blahblah": [None], "hehehe": 123
            }, {
                "some other obj": AssertionError  # wtf lad!!!
            }],
            [{
                "blahblah": max  # a func!!!
            }]
        ]
        for bad_doc_arg in bad_doc_args:
            add_res = tensor_search.add_documents(config=self.config, index_name=self.index_name_1,
                                                  docs=bad_doc_arg, auto_refresh=True)
            assert any(['error' in item for item in add_res['items']])


    def test_add_documents_set_device(self):
        """calling search with a specified device overrides device defined in config"""
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(
                config=self.config, index_name=self.index_name_1, device="cuda:411", docs=[{"some": "doc"}],
                auto_refresh=True)
            return True

        assert run()
        assert mock_config.search_device == "cpu"
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:411"

    def test_add_documents_orchestrator_set_device_single_process(self):
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents_orchestrator(
                config=self.config, index_name=self.index_name_1, device="cuda:22", docs=[{"some": "doc"}, {"som other": "doc"}],
                auto_refresh=True, batch_size=1, processes=1)
            return True

        assert run()
        assert mock_config.search_device == "cpu"
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:22"

    def test_add_documents_orchestrator_set_device_empty_batch(self):
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents_orchestrator(
                config=self.config, index_name=self.index_name_1, device="cuda:22", docs=[{"some": "doc"}, {"som other": "doc"}],
                auto_refresh=True, batch_size=0)
            return True

        assert run()
        assert mock_config.search_device == "cpu"
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:22"

    def test_add_documents_empty(self):
        try:
            tensor_search.add_documents(
                config=self.config, index_name=self.index_name_1, docs=[],
                auto_refresh=True)
            raise AssertionError
        except BadRequestError:
            pass

    def test_resilient_add_images(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1,
            index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/16",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True
                }
        })
        docs_results = [
            ([{"_id": "123","image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
             {"_id": "789", "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
             {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", "result"), ("789", "result"), ("456", "error")]
             ),
            ([{"_id": "123",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("123", "result"), ("789", "result"), ("456", "error"), ("111", "error")]
             ),
            ([{"_id": "505", "image_field": "https://www.marqo.ai/this/image/doesnt/exist3.png"},
              {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", "error"), ("456", "error"), ("111", "error")]
             ),
            ([{"_id": "505", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", "error")]
             ),
        ]
        for docs, expected_results in docs_results:
            add_res = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=docs, auto_refresh=True)
            for i, res_dict in enumerate(add_res['items']):
                assert res_dict["_id"] == expected_results[i][0]
                assert expected_results[i][1] in res_dict

    def test_add_documents_resilient_doc_validation(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        docs_results = [
            # problems with
            ([{"_id": "123", "my_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
             {"_id": "789", "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
             {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", "result"), ("789", "result"), ("456", "error")]
             ),
        ]
        for docs, expected_results in docs_results:
            add_res = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=docs, auto_refresh=True)
            for i, res_dict in enumerate(add_res['items']):
                assert res_dict["_id"] == expected_results[i][0]
                assert expected_results[i][1] in res_dict

    def test_add_documents_empty_dicts(self):
        """handle such list of empty dicts well"""

    def test_mappings_arent_updated_bad_image(self):
        """if an image isn't added properly, we need to ensure that
        it's mappings don't get added to index mappings"""

    def test_resilient_invalid_dict(self):
        """handle invalid dicts"""

    def test_mappings_arent_updated_invalid_dict(self):
        """if an dict isn't added properly, we need to ensure that
        it's mappings don't get added to index mappings"""

    def test_resilient_invalid_field(self):
        """handle invalid dicts"""

    def test_mappings_arent_updated_invalid_field(self):
        """if an dict isn't added properly, we need to ensure that
        it's mappings don't get added to index mappings"""

    def test_resilient_invalid_content(self):
        """handle invalid dicts"""

    def test_mappings_arent_updated_invalid_content(self):
        """if an dict isn't added properly, we need to ensure that
        it's mappings don't get added to index mappings"""

    def test_resilient_invalid_ids(self):
        """handle invalid dicts"""

    def test_mappings_arent_updated_invalid_ids(self):
        """if an dict isn't added properly, we need to ensure that
        it's mappings don't get added to index mappings"""