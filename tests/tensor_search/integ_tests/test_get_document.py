import functools
import unittest

from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search import enums
from marqo.api.exceptions import IndexNotFoundError, InvalidDocumentIdError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from unittest.mock import patch
import os
import pprint


class TestGetDocument(MarqoTestCase):

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

    def test_get_document(self):
        """Also ensures that the _id is returned"""
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(index_name=index.name, docs=[
                        {
                            "_id": "123",
                            "title1": "content 1",
                            "desc2": "content 2. blah blah blah"
                        }], auto_refresh=True, device="cpu",
                        tensor_fields=[] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                )
                self.assertEqual(
                    tensor_search.get_document_by_id(
                        config=self.config, index_name=index.name,
                        document_id="123"
                    ),
                    {
                        "_id": "123",
                        "title1": "content 1",
                        "desc2": "content 2. blah blah blah"
                    }
                )

    def test_get_document_non_existent_index(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                with self.assertRaises(IndexNotFoundError):
                    a = tensor_search.get_document_by_id(
                        config=self.config, index_name="random index name",
                        document_id="123")

    def test_get_document_empty_str(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                with self.assertRaises(InvalidDocumentIdError):
                    a = tensor_search.get_document_by_id(
                        config=self.config, index_name=index.name,
                        document_id="")

    def test_get_document_bad_types(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                for ar in [12.2, 1, [], {}, None]:
                    with self.assertRaises(InvalidDocumentIdError):
                        a = tensor_search.get_document_by_id(
                            config=self.config, index_name=index.name,
                            document_id=ar,
                        )

    def test_get_document_vectors_format(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):

                keys = ("title1", "desc2")
                vals = ("content 1", "content 2. blah blah blah")
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "123", **dict(zip(keys, vals))}],
                    auto_refresh=True, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None)
                )

                res = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="123", show_vectors=True)
                pprint.pprint(res)
                assert enums.TensorField.tensor_facets in res
                assert len(res[enums.TensorField.tensor_facets]) == 2
                assert set(keys) == functools.reduce(
                    lambda x, y: x.union(y),
                    [set(facet.keys()) for facet in res[enums.TensorField.tensor_facets]]
                ) - {enums.TensorField.embedding}

                for facet in res[enums.TensorField.tensor_facets]:
                    assert len(facet) == 2
                    if keys[0] in facet:
                        assert facet[keys[0]] == vals[0]
                    if keys[1] in facet:
                        assert facet[keys[1]] == vals[1]
                    assert enums.TensorField.embedding in facet


