import functools
import os
import pprint
import unittest
import uuid
from unittest import mock

from marqo.api.exceptions import IndexNotFoundError
from marqo.api.exceptions import (
    InvalidDocumentIdError, InvalidArgError,
    IllegalRequestedDocCount
)
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from unittest.mock import patch
import os
import pprint
from tests.utils.transition import *


class TestGetDocuments(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_text_index_with_random_model_request = cls.structured_marqo_index_request(
            model=Model(name='random'),
            fields=[
                FieldRequest(name='title1', type=FieldType.Text),
                FieldRequest(name='desc2', type=FieldType.Text),
            ],
            tensor_fields=["title1", "desc2"]
        )
        unstructured_text_index_with_random_model_request = cls.unstructured_marqo_index_request(model=Model(name='random'))

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_text_index_with_random_model_request,
            unstructured_text_index_with_random_model_request
        ])

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_get_documents_by_ids(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [
                    {"_id": "1", "title1": "content 1"}, {"_id": "2", "title1": "content 2"},
                    {"_id": "3", "title1": "content 3"}
                ]
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(index_name=index.name, docs=docs, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None)
                )
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name, document_ids=['1', '2', '3'],
                    show_vectors=True)
                
                # Check that the documents are found and have the correct content
                for i in range(3):
                    self.assertEqual(res['results'][i]['_found'], True)
                    self.assertEqual(res['results'][i]['_id'], docs[i]['_id'])
                    self.assertEqual(res['results'][i]['title1'], docs[i]['title1'])
                    self.assertIn(enums.TensorField.tensor_facets, res['results'][i])
                    self.assertIn(enums.TensorField.embedding, res['results'][i][enums.TensorField.tensor_facets][0])

    def test_get_documents_vectors_format(self):
        keys = [("title1", "desc2", "_id"), ("title1", "desc2", "_id")]
        vals = [("content 1", "content 2. blah blah blah", "123"),
                ("some more content", "some cool desk", "5678")]
        
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[dict(zip(k, v)) for k, v in zip(keys, vals)],
                    device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None))
                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name,
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

        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                for is_vector_shown in (True, False):
                    for i, ids in enumerate(id_reqs):
                        res = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name, document_ids=ids,
                            show_vectors=is_vector_shown
                        )
                        assert {ii['_id'] for ii in res['results']} == set(id_reqs[i])
                        for doc_res in res['results']:
                            assert not doc_res['_found']

    def test_get_document_vectors_resilient(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[
                        {"_id": '456', "title1": "alexandra"},
                        {'_id': '221', 'desc2': 'hello'}],
                    device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None)
                                            )
                id_reqs = [
                    (['123', '456'], [False, True]), ([['456', '789'], [True, False]]),
                    ([['456', '789', '221'], [True, False, True]]), ([['vkj', '456', '4891'], [False, True, False]])
                ]
                for is_vector_shown in (True, False):
                    for i, (ids, presence) in enumerate(id_reqs):
                        res = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name, document_ids=ids,
                            show_vectors=is_vector_shown
                        )
                        assert [ii['_id'] for ii in res['results']] == id_reqs[i][0]
                        for j, doc_res in enumerate(res['results']):
                            assert doc_res['_id'] == ids[j]
                            assert doc_res['_found'] == presence[j]
                            if doc_res['_found'] and is_vector_shown:
                                assert enums.TensorField.tensor_facets in doc_res
                                assert 'title1' in doc_res or 'desc2' in doc_res

    def test_get_document_vectors_failures(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                for show_vectors_option in (True, False):
                    for bad_get in [[123], [None], [set()], list(), 1.3, dict(),
                                    None, 123, ['123', 456], ['123', 45, '445'], [14, '58']]:
                        with self.assertRaises((InvalidDocumentIdError, InvalidArgError)):
                            res = tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name, document_ids=bad_get,
                                show_vectors=show_vectors_option
                            )

    def test_get_documents_env_limit(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [{"title1": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]
                add_docs_batched(
                    config=self.config,
                    index_name=index.name,
                    docs=docs, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None
                )
                for max_doc in [0, 1, 2, 5, 10, 100, 1000]:
                    mock_environ = {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}

                    @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
                    def run():
                        half_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(max_doc // 2)])
                        self.assertEqual(len(half_search['results']),max_doc // 2)
                        limit_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(max_doc)])
                        self.assertEqual(len(limit_search['results']), max_doc)
                        with self.assertRaises(IllegalRequestedDocCount):
                            oversized_search = tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name,
                                document_ids=[docs[i]['_id'] for i in range(max_doc + 1)])
                        with self.assertRaises(IllegalRequestedDocCount):
                            very_oversized_search = tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name,
                                document_ids=[docs[i]['_id'] for i in range(max_doc * 2)])
                        return True
                assert run()

    def test_limit_results_none(self):
        """if env var isn't set or is None"""
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [{"title1": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]

                add_docs_batched(
                    config=self.config, 
                    index_name=index.name,
                    docs=docs, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None
                )

                for mock_environ in [dict(),
                                    {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: ''}]:
                    @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
                    def run():
                        sample_size = 500
                        limit_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(sample_size)])
                        assert len(limit_search['results']) == sample_size
                        return True

                    assert run()
