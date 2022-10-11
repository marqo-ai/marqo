import functools
import pprint
from marqo.tensor_search import enums
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

    def test_get_documents_vectors_format(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        keys = [("title 1", "desc 2", "_id"), ("title 1", "desc 2", "_id")]
        vals = [("content 1", "content 2. blah blah blah", "123"),
                ("some more content", "some cool desk", "5678")]
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            dict(zip(k, v)) for k, v in zip(keys, vals)], auto_refresh=True)
        get_res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_1,
            document_ids=["123", "5678"], show_vectors=True)['results']
        assert len(get_res) == 2
        for i, retrieved_doc in enumerate(get_res):
            assert enums.TensorField.tensor_facets in retrieved_doc
            assert len(retrieved_doc[enums.TensorField.tensor_facets]) == 2
            assert set(keys[i]).union({enums.TensorField.embedding}) - {'_id'} == functools.reduce(
                lambda x, y: x.union(y),
                [set(facet.keys()) for facet in retrieved_doc[enums.TensorField.tensor_facets]]
            )
            for facet in retrieved_doc[enums.TensorField.tensor_facets]:
                assert len(facet) == 2
                if keys[0] in facet:
                    assert facet[keys[0]] == vals[0]
                if keys[1] in facet:
                    assert facet[keys[1]] == vals[1]
                assert enums.TensorField.embedding in facet

    def test_get_documents_by_ids_resilience(self):
        """TODO"""