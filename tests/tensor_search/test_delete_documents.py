import pprint

import marqo.tensor_search.delete_docs
import marqo.tensor_search.tensor_search
from marqo.tensor_search import tensor_search
from marqo.config import Config
from marqo.errors import IndexNotFoundError
from tests.marqo_test import MarqoTestCase
import requests


class TestDeleteDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"

        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_delete_documents(self):
        # first batch:
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder"},
            ], auto_refresh=True)
        count0_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"hooped": "absolutely ridic", "Darling": "A harbour in Sydney", "_id": "455"},
                {"efg": "hheeehehehhe", "_id": "at-at"}
            ], auto_refresh=True)
        count1_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        marqo.tensor_search.tensor_search.delete_documents(config=self.config, index_name=self.index_name_1, doc_ids=["455", "at-at"],
                                                           auto_refresh=True)
        count_post_delete = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        assert count_post_delete == count0_res

    def test_delete_docs_format(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney", "_id": "1234"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder", "_id": "5678"},
            ], auto_refresh=True)

        res = marqo.tensor_search.tensor_search.delete_documents(config=self.config, doc_ids=["5678", "491"], index_name=self.index_name_1
                                                                 , auto_refresh=False)
        assert res["index_name"] == self.index_name_1
        assert res["type"] == "documentDeletion"
        assert res["status"] == "succeeded"
        assert res["details"]["receivedDocumentIds"] == 2
        assert res["details"]["deletedDocuments"] == 1
        assert "PT" in res["duration"]
        assert "Z" in res["startedAt"]
        assert "T" in res["finishedAt"]

