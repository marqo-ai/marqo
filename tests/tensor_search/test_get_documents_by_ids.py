import pprint

from marqo.errors import IndexNotFoundError, InvalidDocumentIdError
from marqo.client import Client
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestGetDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        c = Client(**self.client_settings)
        self.config = c.config
        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_get_documents_by_ids(self):
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"_id": "1", "title 1": "content 1"}, {"_id": "2", "title 1": "content 1"},
            {"_id": "3", "title 1": "content 1"}
        ], auto_refresh=True)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_1, document_ids=['1', '2', '3'],
            show_vectors=True)
        pprint.pprint(res)

    def test_get_documents_by_ids_show_vectors(self):
        """TODO"""

