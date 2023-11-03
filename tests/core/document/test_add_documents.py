import copy
import functools
import json
import math
import os
import pprint
import re
import uuid
from unittest import mock

import PIL
import pytest
import requests

from marqo.core.models.marqo_index import *
from marqo.errors import IndexNotFoundError, BadRequestError
from marqo.s2_inference import types, s2_inference
from marqo.tensor_search import add_docs
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search, index_meta_cache, backend
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.enums import IndexSettingsField
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase

class TestAddDocumentsStructuredIndex(MarqoTestCase):
    """Test adding_documents functionalities to a structured index."""
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
        
        image_index_chunk = cls.marqo_index(
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
            ],
            image_preprocessing=ImagePreProcessing(patch_method="frcnn")
        )
        
        cls.text_index_name = text_index.name
        cls.image_index_name = image_index.name
        cls.image_index_chunk_name = image_index_chunk.name

        cls.indexes = [text_index, image_index, image_index_chunk]

        # These indexes will be deleted in tearDownClass
        cls.create_indexes(cls.indexes)

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.clear_indexes(self.indexes)
        self.device_patcher.stop()

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


    def test_add_documents_dupe_ids(self):
        """
        Should only use the latest inserted ID. Make sure it doesn't get the first/middle one
        """

        expected_document = {
            "_id": "3",
            "title": "doc 3b"
        }

        self.add_documents_helper(
            index_name=self.text_index_name, docs=[
                {
                    "_id": "1",
                    "title": "doc 1"
                },
                {
                    "_id": "2",
                    "title": "doc 2",
                },
                {
                    "_id": "3",
                    "title": "doc 3a",
                },
                expected_document]
            )

        actual_document = tensor_search.get_document_by_id(config=self.config,
                                                           index_name=self.text_index_name,
                                                           document_id="3")
        self.assertEqual(actual_document, expected_document)

    def test_add_documents_with_missing_index_fails(self):
        rand_index = 'a' + str(uuid.uuid4()).replace('-', '')

        with pytest.raises(IndexNotFoundError):
            self.add_documents_helper(index_name=rand_index, docs=[{"_id": "123", "title": "content 1"}])

    def test_add_docs_response_format(self):
        add_res = self.add_documents_helper(
            index_name=self.text_index_name, docs=[
                {
                    "_id": "123",
                    "title": "content 1",
                    "description": "content 2. blah blah blah"
                },
                {
                    "_id": "456",
                    "title": "content 2",
                    "description": "content 2. blah blah blah"
                },
                {
                    "_id": "911",
                    "title": "content 3",
                    "description": "content 3. blah blah blah"
                }
            ]
        )

        assert "errors" in add_res
        assert "processingTimeMs" in add_res
        assert "index_name" in add_res
        assert "items" in add_res

        assert add_res["processingTimeMs"] > 0
        assert isinstance(add_res["errors"], bool)
        assert add_res["index_name"] == self.text_index_name

        for item in add_res["items"]:
            assert "_id" in item
            assert "status" in item

    def test_add_documents_validation(self):
        """These bad docs should return errors"""
        bad_doc_args = [
            [{"_id": "to_fail_123", "title": {}}],
            # strict checking: only allowed fields:
            [{"_id": "to_fail_123", "title": dict()}],
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]}],
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]},
             {"_id": "to_pass_123", "title": 'some_content'}],
            [{"_id": "to_fail_123", "title": [{"abc": "678"}]}],
            [{"_id": "to_fail_123", "title": {"abc": "234"}}],
            [{"_id": "to_fail_123", "title": {"abc": "234"}},
             {"_id": "to_pass_123", "title": 'some_content'}],
            # other checking:
            [{"title": {1243}, "_id": "to_fail_123"}],
            [{"title": None, "_id": "to_fail_123"}],
            [{"_id": "to_fail_123", "title": [None], "description": 123},
             {"_id": "to_fail_567", "title": "finnne", 123: "heehee"}],
            [{"_id": "to_fail_123", "title": [None], "hehehe": 123},
             {"_id": "to_fail_567", "title": AssertionError}],
            [{"_id": "to_fail_567", "title": max}]
        ]

        for bad_doc_arg in bad_doc_args:
            add_res = self.add_documents_helper(index_name=self.text_index_name, docs=bad_doc_arg)
            assert add_res['errors'] is True
            assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])
            assert all(['error' not in item
                        for item in add_res['items'] if item['_id'].startswith('to_pass')])

    def test_add_documents_id_validation(self):
        """These bad docs should return errors"""
        bad_doc_args = [
            # Wrong data types for ID
            # Tuple: (doc_list, number of docs that should succeed)
            ([{"_id": {}, "title": "zzz"}], 0),
            ([{"_id": dict(), "title": "zzz"}], 0),
            ([{"_id": [1, 2, 3], "title": "zzz"}], 0),
            ([{"_id": 4, "title": "zzz"}], 0),
            ([{"_id": None, "title": "zzz"}], 0),

            ([{"_id": "proper id", "title": "zzz"},
              {"_id": ["bad", "id"], "title": "zzz"},
              {"_id": "proper id 2", "title": "zzz"}], 2)
        ]

        for bad_doc_arg in bad_doc_args:
            add_res = self.add_documents_helper(index_name=self.text_index_name, docs=bad_doc_arg[0])
            assert add_res['errors'] is True
            succeeded_count = 0
            for item in add_res['items']:
                if "status" in item and item["status"] == 200:
                    succeeded_count += 1

            assert succeeded_count == bad_doc_arg[1]

    def test_add_documents_list_non_tensor_validation(self):
        """This doc is valid but should return error because my_field is not marked non-tensor"""
        bad_doc_args = [
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]}],
        ]
        for bad_doc_arg in bad_doc_args:
            add_res = self.add_documents_helper(index_name=self.text_index_name, docs=bad_doc_arg)
            assert add_res['errors'] is True
            assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])

    def test_add_documents_list_success(self):
        good_docs = [
            [{"_id": "to_fail_123", "list": ["wow", "this", "is"]}]
        ]
        for good_doc_arg in good_docs:
            add_res = self.add_documents_helper(index_name=self.text_index_name, docs=good_doc_arg,
                                                non_tensor_fields = ["list"])
            assert add_res['errors'] is False

    def test_add_documents_list_data_type_validation(self):
        """These bad docs should return errors"""
        bad_doc_args = [
            [{"_id": "to_fail_123", "list": ["wow", "this", False]}],
            [{"_id": "to_fail_124", "list": [1, 2, 3]}],
            [{"_id": "to_fail_125", "list": [{}]}]
        ]
        for bad_doc_arg in bad_doc_args:
            add_res = self.add_documents_helper(
                index_name=self.text_index_name, docs=bad_doc_arg, non_tensor_fields=["list"])
        assert add_res['errors'] is True
        assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])

    def test_add_documents_set_device(self):
        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.text_index_name, device="cuda:22", docs=[{"title": "doc"}],
                    auto_refresh=True,
                ),
            )
            return True

        assert run()
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:22"

    def test_add_documents_empty(self):
        try:
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.text_index_name, docs=[],
                    auto_refresh=True, device="cpu")
            )
            raise AssertionError
        except BadRequestError:
            pass

    def test_resilient_add_images(self):
        for index_name in [self.image_index_name, self.image_index_chunk_name]:
            docs_results = [
                ([{"_id": "123",
                   "image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
                  {"_id": "789",
                   "image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
                  {"_id": "456", "image": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
                 [("123", "success"), ("789", "success"), ("456", "error")]
                 ),
                ([{"_id": "123",
                   "image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
                  {"_id": "789",
                   "image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
                  {"_id": "456", "image": "https://www.marqo.ai/this/image/doesnt/exist.png"},
                  {"_id": "111", "image": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
                 [("123", "success"), ("789", "success"), ("456", "error"), ("111", "error")]
                 ),
                ([{"_id": "505", "image": "https://www.marqo.ai/this/image/doesnt/exist3.png"},
                  {"_id": "456", "image": "https://www.marqo.ai/this/image/doesnt/exist.png"},
                  {"_id": "111", "image": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
                 [("505", "error"), ("456", "error"), ("111", "error")]
                 ),
                ([{"_id": "505", "image": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
                 [("505", "error")]
                 ),
            ]
            for docs, expected_results in docs_results:
                add_res = self.add_documents_helper(index_name=index_name, docs=docs)
                assert len(add_res['items']) == len(expected_results)
                print(add_res)
                for i, res_dict in enumerate(add_res['items']):
                    if expected_results[i][1] == "success":
                        assert res_dict["status"] == 200
                    elif expected_results[i][1] == "error":
                        assert "error" in res_dict

    def test_add_documents_id_image_url(self):
        docs = [{
            "_id": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "title": "wow"}
        ]

        with mock.patch('PIL.Image.open') as mock_image_open:
            self.add_documents_helper(index_name=self.image_index_name, docs=docs)
            mock_image_open.assert_not_called()

    def test_doc_too_large(self):
        max_size = 400000
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            add_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.text_index_name, docs=[
                        {"_id": "123", 'title': "edf " * (max_size // 4)},
                        {"_id": "789", "title": "abc " * ((max_size // 4) - 500)},
                        {"_id": "456", "title": "exc " * (max_size // 4)},
                    ],
                    auto_refresh=True, device="cpu"
                ))

            assert len(add_res["items"]) == 3

            assert add_res['items'][0]['status'] != 200
            assert "exceeds the allowed document size limit" in add_res['items'][0]['error']

            assert add_res['items'][1]['status'] == 200

            assert add_res['items'][2]['status'] != 200
            assert "exceeds the allowed document size limit" in add_res['items'][2]['error']
            return True

        assert run()


    def test_doc_too_large_none_env_var(self):
        for env_dict in [dict()]:
            @mock.patch.dict(os.environ, {**os.environ, **env_dict})
            def run():
                add_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.text_index_name, docs=[
                            {"_id": "123", 'title': "Some content"},
                        ],
                        auto_refresh=True, device="cpu"
                    ))
                items = add_res['items']
                assert items[0]["status"] == 200
                return True

            assert run()