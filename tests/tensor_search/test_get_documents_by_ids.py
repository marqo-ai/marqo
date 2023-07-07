import functools
import pprint
import uuid
from marqo.tensor_search import enums
from marqo.errors import (
    IndexNotFoundError, InvalidDocumentIdError, InvalidArgError,
    IllegalRequestedDocCount
)
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from unittest import mock
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
import os


class TestGetDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"  # standard index created by setUp
        self.index_name_2 = "my-test-index-2"  # for tests that need custom index config
        self._delete_testing_indices()

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_get_documents_by_ids(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[
                {"_id": "1", "title 1": "content 1"}, {"_id": "2", "title 1": "content 1"},
                {"_id": "3", "title 1": "content 1"}
            ], auto_refresh=True, device="cpu")
        )
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_1, document_ids=['1', '2', '3'],
            show_vectors=True)
        pprint.pprint(res)

    def test_get_documents_vectors_format(self):
        keys = [("title 1", "desc 2", "_id"), ("title 1", "desc 2", "_id")]
        vals = [("content 1", "content 2. blah blah blah", "123"),
                ("some more content", "some cool desk", "5678")]
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[dict(zip(k, v)) for k, v in zip(keys, vals)],
            auto_refresh=True, device="cpu"))
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
        tensor_search.add_documents(config=self.config,  add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {"_id": '456', "title": "alexandra"},
                {'_id': '221', 'message': 'hello'}],
            auto_refresh=True, device="cpu")
        )
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

    def test_get_documents_env_limit(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2,
            index_settings={enums.IndexSettingsField.index_defaults: {
                enums.IndexSettingsField.model: enums.MlModel.bert
            }})
        docs = [{"Title": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2,
                docs=docs, auto_refresh=False, device="cpu"
            )
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_2)
        for max_doc in [0, 1, 2, 5, 10, 100, 1000]:
            mock_environ = {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}

            @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
            def run():
                half_search = tensor_search.get_documents_by_ids(
                   config=self.config, index_name=self.index_name_2,
                   document_ids=[docs[i]['_id'] for i in range(max_doc // 2)])
                assert len(half_search['results']) == max_doc // 2
                limit_search = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=self.index_name_2,
                    document_ids=[docs[i]['_id'] for i in range(max_doc)])
                assert len(limit_search['results']) == max_doc
                try:
                    oversized_search = tensor_search.get_documents_by_ids(
                        config=self.config, index_name=self.index_name_2,
                        document_ids=[docs[i]['_id'] for i in range(max_doc + 1)])
                    raise AssertionError
                except IllegalRequestedDocCount:
                    pass
                try:
                    very_oversized_search = tensor_search.get_documents_by_ids(
                         config=self.config, index_name=self.index_name_2,
                         document_ids=[docs[i]['_id'] for i in range(max_doc * 2)])
                    raise AssertionError
                except IllegalRequestedDocCount:
                    pass
                return True
        assert run()

    def test_limit_results_none(self):
        """if env var isn't set or is None"""
        docs = [{"Title": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=docs, auto_refresh=False, device = "cpu"
            ),
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)

        for mock_environ in [dict(),
                             {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: ''}]:
            @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
            def run():
                sample_size = 500
                limit_search = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=self.index_name_1,
                    document_ids=[docs[i]['_id'] for i in range(sample_size)])
                assert len(limit_search['results']) == sample_size
                return True

            assert run()
