import json
import pprint
import requests
from marqo.neural_search.enums import NeuralField
from marqo.client import Client
from marqo.errors import MarqoApiError
from marqo.neural_search import neural_search, index_meta_cache
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
        except MarqoApiError as s:
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
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        neural_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "id": "abcdefgh",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }], auto_refresh=True)
        assert neural_search.get_document_by_id(
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
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        neural_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
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
        neural_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
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
        add_doc_res = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        r2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert r2.status_code == 200

    def test_default_index_settings(self):
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "model" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "media_type" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "__field_name" in \
               index_info.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]

    def test_default_index_settings_implicitly_created(self):
        add_doc_res = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "model" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "media_type" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
        assert "__field_name" in \
               index_info.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]

    def test_add_new_fields_on_the_fly(self):
        add_doc_res = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        cluster_ix_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )

        assert "__vector_abc" in cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]
        assert "dimension" in cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]["__vector_abc"]
        add_doc_res = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "1234", "The title book 1": "hahehehe"}], auto_refresh=True
        )
        cluster_ix_info_2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert "__vector_abc" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]
        assert "__vector_The title book 1" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]
        assert "dimension" in \
               cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]["__vector_The title book 1"]

    def test_add_new_fields_on_the_fly_index_cache_syncs(self):
        index_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        add_doc_res_1 = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        index_info_2 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].properties[NeuralField.chunks]["properties"]["__vector_abc"] \
               == index_info_2.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"]["__vector_abc"]
        add_doc_res_2 = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"cool field": "yep yep", "haha": "heheh"}], auto_refresh=True
        )
        index_info_3 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
               == {k:v for k,v in index_info_3.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"].items()
                   if k.startswith("__vector_")}

    def test_add_multiple_fields(self):
        add_doc_res = neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"cool v field": "yep yep", "haha ee": "heheh"}], auto_refresh=True
        )
        cluster_ix_info = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
               == {k: v for k, v in
                   cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][NeuralField.chunks]["properties"].items()
                   if k.startswith("__vector_")}
        assert "__vector_cool v field" in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()
        assert "__vector_haha ee" in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()

    def test_add_docs_response_format(self):
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        add_res = neural_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
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
            assert "result" in item
            assert "status" in item
