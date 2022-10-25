import functools
import pprint
from marqo.tensor_search import enums
from marqo.errors import (
    IndexNotFoundError, InvalidDocumentIdError, InvalidArgError
)
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestGetDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
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

    def test_get_document_vectors_non_existent(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        id_reqs = [
            ['123', '456'], ['124']
        ]
        for is_vector_shown in (True, False):
            for i, ids in enumerate(id_reqs):
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=self.index_name_1, document_ids=ids,
                    show_vectors=is_vector_shown
                )
                assert {ii['_id'] for ii in res['results']} == set(id_reqs[i])
                for doc_res in res['results']:
                    assert not doc_res['_found']

    def test_get_document_vectors_resilient(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"_id": '456', "title": "alexandra"}, {'_id': '221', 'message': 'hello'}], auto_refresh=True)
        id_reqs = [
            (['123', '456'], [False, True]), ([['456', '789'], [True, False]]),
            ([['456', '789', '221'], [True, False, True]]), ([['vkj', '456', '4891'], [False, True, False]])
        ]
        for is_vector_shown in (True, False):
            for i, (ids, presence) in enumerate(id_reqs):
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=self.index_name_1, document_ids=ids,
                    show_vectors=is_vector_shown
                )
                assert [ii['_id'] for ii in res['results']] == id_reqs[i][0]
                for j, doc_res in enumerate(res['results']):
                    assert doc_res['_id'] == ids[j]
                    assert doc_res['_found'] == presence[j]
                    if doc_res['_found'] and is_vector_shown:
                        assert enums.TensorField.tensor_facets in doc_res
                        assert 'title' in doc_res or 'message' in doc_res

    def test_get_document_vectors_failures(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        for show_vectors_option in (True, False):
            for bad_get in [[123], [None], [set()], list(), 1.3, dict(),
                            None, 123, ['123', 456], ['123', 45, '445'], [14, '58']]:
                try:
                    res = tensor_search.get_documents_by_ids(
                        config=self.config, index_name=self.index_name_1, document_ids=bad_get,
                        show_vectors=show_vectors_option
                    )
                    raise AssertionError
                except (InvalidDocumentIdError, InvalidArgError):
                    pass