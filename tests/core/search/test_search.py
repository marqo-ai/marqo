import os
import uuid
from unittest import mock

import pytest

from marqo.core.models.marqo_index import *
from marqo.errors import IndexNotFoundError, BadRequestError
from marqo.tensor_search import enums
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount, BadRequestError, InternalError
)
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestSearchStructuredIndex(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # A model that for text only
        text_index = cls.marqo_index(
            name='a' + str(uuid.uuid4()).replace('-', ''),
            type=IndexType.Structured,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            fields=[
                Field(name='title', type=FieldType.Text),
                Field(name='description', type=FieldType.Text),
                Field(name="list", type=FieldType.ArrayText),
            ],
            tensor_fields=[
                TensorField(name='title'), TensorField(name='description')
            ]
        )

        # A model that for text and image
        image_index = cls.marqo_index(
            name='a' + str(uuid.uuid4()).replace('-', ''),
            type=IndexType.Structured,
            model=Model(name='open_clip/ViT-B-32/openai'),
            fields=[
                Field(name='title', type=FieldType.Text),
                Field(name='description', type=FieldType.Text),
                Field(name="image", type=FieldType.ImagePointer),
            ],
            tensor_fields=[
                TensorField(name='title'), TensorField(name='description'),
                TensorField(name="image")
            ]
        )

        cls.text_index_name = text_index.name
        cls.image_index_name = image_index.name
        cls.indexes = [text_index, image_index]
        # These indexes will be deleted in tearDownClass
        cls.create_indexes(cls.indexes)

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def add_documents_helper(self, index_name, docs, **kwargs):
        """A helper function in this class to call aad_documents to reduce duplication"""
        # Default values in this test class
        default_add_docs_params = AddDocsParams(
            index_name=index_name,
            docs=docs,
            auto_refresh=True,
            device="cpu",
            **kwargs
        )
        return tensor_search.add_documents(self.config, add_docs_params=default_add_docs_params)

    def vector_search_helper(self, index_name, text, **kwargs):
        """A helper function in this class to call vector_search to reduce duplication"""
        # Default values in this test class
        vector_search_params = {
            "device": "cpu",
            "result_count": 10,
            "index_name": index_name,
            "text": text,
            "search_method": "TENSOR"
        }
        vector_search_params.update(kwargs)
        return tensor_search.search(self.config, **vector_search_params)

    def tearDown(self):
        self.clear_indexes(self.indexes)
        self.device_patcher.stop()

    def test_each_doc_returned_once(self):
        self.add_documents_helper(index_name=self.text_index_name, docs=[
                {"description": "Exact match hehehe efgh ", "title": "baaadd efgh ", "_id": "5678"},
                {"description": "shouldn't really match ", "title": "Nope.....", "_id": "1234"},
            ])

        search_res = self.vector_search_helper(index_name=self.text_index_name, text=" efgh ")
        assert len(search_res['hits']) == 2

    @mock.patch.dict(os.environ, {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '2'}})
    def test_search_with_excessive_searchable_attributes(self):
        with self.assertRaises(InvalidArgError):
            self.vector_search_helper(index_name=self.text_index_name, text="Exact match hehehe",
                searchable_attributes=["title", "description", "others"]
            )

