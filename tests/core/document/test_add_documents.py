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

    # def test_add_plain_id_field(self):
    #     """does a plain 'id' field work (in the doc body)? """
    #     add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.,
    #             docs=[{
    #                 "_id": "123",
    #                 "id": "abcdefgh",
    #                 "title": "content 1",
    #                 "description": "content 2. blah blah blah"
    #             }],
    #             auto_refresh=True, device="cpu"
    #         )
    #     )
    #     assert tensor_search.get_document_by_id(
    #         config=self.config, index_name=self.index_name_1,
    #         document_id="123") == {
    #                "id": "abcdefgh",
    #                "_id": "123",
    #                "title 1": "content 1",
    #                "desc 2": "content 2. blah blah blah"
    #            }

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

    #
    # def test_update_docs_update_chunks(self):
    #     """Updating a doc needs to update the corresponding chunks"
    #     """
    #     tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[
    #                 {
    #                     "_id": "123",
    #                     "title 1": "content 1",
    #                     "desc 2": "content 2. blah blah blah"
    #                 }],
    #             auto_refresh=True, device="cpu")
    #     )
    #     count0_res = requests.post(
    #         F"{self.endpoint}/{self.index_name_1}/_count",
    #         timeout=self.config.timeout,
    #         verify=False
    #     )
    #     count0 = count0_res.json()["count"]
    #     assert count0 == 1
    #     tensor_search.add_documents(
    #         config=self.config,
    #         add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1,
    #             docs=[{
    #                 "_id": "123",
    #                 "title 1": "content 1",
    #                 "desc 2": "content 2. blah blah blah"
    #             }],
    #             auto_refresh=True, device="cpu"
    #         )
    #     )
    #     count1_res = requests.post(
    #         F"{self.endpoint}/{self.index_name_1}/_count",
    #         timeout=self.config.timeout,
    #         verify=False
    #     )
    #     count1 = count1_res.json()["count"]
    #     assert count1 == count0
    #
    # def test_add_documents_whitespace(self):
    #     """Index fields consisting of only whitespace"""
    #     docs = [
    #         {"title": "", "_id": "0"},
    #         {"title": " ", "_id": "1"},
    #         {"title": "  ", "_id": "2"},
    #         {"title": "\r", "_id": "3"},
    #         {"title": "\r ", "_id": "4"},
    #         {"title": "\r\r", "_id": "5"},
    #         {"title": "\r\t\n", "_id": "6"},
    #     ]
    #     res = self.add_documents_helper(index_name=self.text_index_name, docs=docs)
    #     print(res)
    #     for id, expected_doc in enumerate(docs):
    #         actual_doc = tensor_search.get_document_by_id(config=self.config,
    #                                                       index_name=self.text_index_name,
    #                                                       document_id=str(id))
    #         assert actual_doc == expected_doc
    #
    # def test_default_index_settings(self):
    #     index_info = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     assert "model" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
    #     assert "media_type" in index_info.json()[self.index_name_1]["mappings"]["_meta"]
    #     assert "__field_name" in \
    #            index_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
    #
    # def test_add_new_fields_on_the_fly(self):
    #     add_doc_res = tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True, device="cpu"
    #         )
    #     )
    #     cluster_ix_info = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #
    #     assert TensorField.marqo_knn_field in \
    #            cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
    #     assert "dimension" in \
    #            cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"][
    #                TensorField.marqo_knn_field]
    #     add_doc_res = tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[{"abc": "1234", "The title book 1": "hahehehe"}], auto_refresh=True,
    #             device="cpu"
    #         )
    #     )
    #     cluster_ix_info_2 = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     assert "__vector_abc" not in \
    #            cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
    #     assert "__vector_The title book 1" not in \
    #            cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
    #     assert TensorField.marqo_knn_field in \
    #            cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"]
    #     assert "dimension" in \
    #            cluster_ix_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"][
    #                TensorField.marqo_knn_field]
    #
    # def test_add_new_fields_on_the_fly_index_cache_syncs(self):
    #     index_info = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     add_doc_res_1 = tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True, device="cpu"
    #         )
    #     )
    #     index_info_2 = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     assert index_meta_cache.get_cache()[self.index_name_1].properties[TensorField.chunks]["properties"][
    #                TensorField.marqo_knn_field] \
    #            == index_info_2.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks]["properties"][
    #                TensorField.marqo_knn_field]
    #     add_doc_res_2 = tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[{"cool field": "yep yep", "haha": "heheh"}], auto_refresh=True,
    #             device="cpu"
    #         )
    #     )
    #     index_info_3 = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
    #            == {k: v for k, v in
    #                index_info_3.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks][
    #                    "properties"].items()
    #                if k.startswith("__vector_")}
    #
    # def test_add_multiple_fields(self):
    #     add_doc_res = tensor_search.add_documents(
    #         config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=[{"cool v field": "yep yep", "haha ee": "heheh"}],
    #             auto_refresh=True, device="cpu"
    #         )
    #     )
    #     cluster_ix_info = requests.get(
    #         url=f"{self.endpoint}/{self.index_name_1}",
    #         verify=False
    #     )
    #     assert index_meta_cache.get_cache()[self.index_name_1].get_vector_properties() \
    #            == {k: v for k, v in
    #                cluster_ix_info.json()[self.index_name_1]["mappings"]["properties"][TensorField.chunks][
    #                    "properties"].items()
    #                if k.startswith("__vector_")}
    #
    #     # Only 1 vector field should be created
    #     assert TensorField.marqo_knn_field in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()
    #     assert "__vector_cool v field" not in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()
    #     assert "__vector_haha ee" not in index_meta_cache.get_cache()[self.index_name_1].get_vector_properties()
    #
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
                for i, res_dict in enumerate(add_res['items']):
                    assert res_dict["_id"] == expected_results[i][0]
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
    #
    # def test_add_documents_id_in_tensor_field(self):
    #     index_setting = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: True
    #         }
    #     }
    #     tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2,
    #                                       index_settings=index_setting)
    #     docs = [{
    #         "_id": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    #         "my_field": "wow"}
    #     ]
    #
    #     with mock.patch('marqo.s2_inference.s2_inference.vectorise') as mock_vectorise:
    #         with pytest.raises(BadRequestError, match=re.escape('`_id` field cannot be a tensor field.')):
    #             tensor_search.add_documents(config=self.config,
    #                                         add_docs_params=AddDocsParams(
    #                                             index_name=self.index_name_2, docs=docs, auto_refresh=True,
    #                                             device="cpu", tensor_fields=['my_field', '_id'], non_tensor_fields=None
    #                                         ))
    #         mock_vectorise.assert_not_called()
    #
    # def test_add_documents_id_not_in_non_tensor_field(self):
    #     index_setting = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: False
    #         }
    #     }
    #     tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2,
    #                                       index_settings=index_setting)
    #     docs = [{
    #         "_id": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    #         "my_field": "wow"}
    #     ]
    #
    #     mock_vectorise = mock.MagicMock()
    #     mock_vectorise.side_effect = s2_inference.vectorise
    #
    #     @mock.patch('marqo.s2_inference.s2_inference.vectorise', mock_vectorise)
    #     def run():
    #         tensor_search.add_documents(config=self.config,
    #                                     add_docs_params=AddDocsParams(
    #                                         index_name=self.index_name_2, docs=docs, auto_refresh=True,
    #                                         device="cpu", non_tensor_fields=[]
    #                                     ))
    #         vectorised_content = [call_kwargs['content'] for call_args, call_kwargs
    #                               in mock_vectorise.call_args_list]
    #         expected_content = [['wow']]
    #         mock_vectorise.assert_called_once()
    #         assert vectorised_content == expected_content
    #
    #     run()
    #
    # def test_add_documents_resilient_doc_validation(self):
    #     docs_results = [
    #         # handle empty dicts
    #         ([{"_id": "123", "my_field": "legitimate text"},
    #           {},
    #           {"_id": "456", "my_field": "awesome stuff!"}],
    #          [("123", "result"), (None, 'error'), ('456', 'result')]
    #          ),
    #         ([{}], [(None, 'error')]),
    #         ([{}, {}], [(None, 'error'), (None, 'error')]),
    #         ([{}, {}, {"some_dict": "yep"}], [(None, 'error'), (None, 'error'), (None, 'result')]),
    #         # handle invalid dicts
    #         ([{"this is a set, lmao"}, "this is a string", {"some_dict": "yep"}],
    #          [(None, 'error'), (None, 'error'), (None, 'result')]),
    #         ([1234], [(None, 'error')]), ([None], [(None, 'error')]),
    #         # handle invalid field names
    #         ([{123: "bad"}, {"_id": "cool"}], [(None, 'error'), ("cool", 'result')]),
    #         ([{"__chunks": "bad"}, {"_id": "1511", "__vector_a": "some content"}, {"_id": "cool"},
    #           {"_id": "144451", "__field_content": "some content"}],
    #          [(None, 'error'), ("1511", 'error'), ("cool", 'result'), ("144451", "error")]),
    #         ([{123: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 'error'), ("cool", 'result')]),
    #         ([{None: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 'error'), ("cool", 'result')]),
    #         # handle bad content
    #         ([{"bad": None, "_id": "12345"}, {"_id": "cool"}], [(None, 'error'), ("cool", 'result')]),
    #         ([{"bad": [1, 2, 3, 4], "_id": "12345"}, {"_id": "cool"}], [("12345", 'error'), ("cool", 'result')]),
    #         ([{"bad": ("cat", "dog"), "_id": "12345"}, {"_id": "cool"}], [("12345", 'error'), ("cool", 'result')]),
    #         ([{"bad": set(), "_id": "12345"}, {"_id": "cool"}], [(None, 'error'), ("cool", 'result')]),
    #         ([{"bad": dict(), "_id": "12345"}, {"_id": "cool"}], [(None, 'error'), ("cool", 'result')]),
    #         # handle bad _ids
    #         ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}], [(None, 'error'), ("cool", 'result')]),
    #         ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}, {"bad": "hehehe", "_id": None}, {"field": "yep"},
    #           {"_id": (1, 2), "efgh": "abc"}, {"_id": 1.234, "cool": "wowowow"}],
    #          [(None, 'error'), ("cool", 'result'), (None, 'error'), (None, 'result'), (None, 'error'),
    #           (None, 'error')]),
    #         # mixed
    #         ([{(1, 2, 3): set(), "_id": "12345"}, {"_id": "cool"}, {"bad": [1, 2, 3], "_id": None}, {"field": "yep"},
    #           {}, "abcdefgh"],
    #          [(None, 'error'), ("cool", 'result'), (None, 'error'), (None, 'result'), (None, 'error'),
    #           (None, 'error')]),
    #     ]
    #     for docs, expected_results in docs_results:
    #         add_res = tensor_search.add_documents(
    #             config=self.config, add_docs_params=AddDocsParams(
    #                 index_name=self.index_name_1, docs=docs, auto_refresh=True,
    #                 device="cpu"
    #             )
    #         )
    #         assert len(add_res['items']) == len(expected_results)
    #         for i, res_dict in enumerate(add_res['items']):
    #             # if the expected id is None, then it assumed the id is
    #             # generated and can't be asserted against
    #             if expected_results[i][0] is not None:
    #                 assert res_dict["_id"] == expected_results[i][0]
    #             assert expected_results[i][1] in res_dict
    #
    # def test_mappings_arent_updated(self):
    #     """if a doc isn't added properly, we need to ensure that
    #     it's mappings don't get added to index mappings
    #
    #     Test for:
    #         - invalid images
    #         - invalid dict
    #         - invalid fields
    #         - invalid content
    #         - invalid _ids
    #     """
    #     docs_results = [
    #         # invalid dict
    #         ([{"_id": "24frg", "my_field": "legitimate text"}, {},
    #           {"_id": "srgb4", "my_field": "awesome stuff!"}],
    #          ({"my_field"}, {})
    #          ),
    #         # invalid fields
    #         ([{"_id": "14g", (12, 14): "some content"}, {"_id": "1511", None: "some content"},
    #           {"_id": "1511", "__vector_a": "some content"}, {"_id": "1234f", "__chunks": "some content"},
    #           {"_id": "144451", "__field_content": "some content"},
    #           {"_id": "sv4124", "good_field_3": "some content 2 ", "good_field_4": 3.65}],
    #          ({"good_field_3", "good_field_4"}, {(12, 14), None, "__vector_a", "__chunks", "__field_content"})
    #          ),
    #         # invalid content
    #         ([{"_id": "f24f4", "bad_field_1": []}, {"_id": "4t6g5g5", "bad_field_1": {}},
    #           {"_id": "df3f3", "bad_field_1": (1, 23, 4)},
    #           {"_id": "fr2452", "good_field_1": 000, "good_field_2": 3.65}],
    #          ({"good_field_1", "good_field_2"}, {"bad_field_1"})
    #          ),
    #         # invalid -ids
    #         ([{"_id": 12445, "bad_field_1": "actually decent text"}, {"_id": [], "bad_field_1": "actually decent text"},
    #           {"_id": {}, "bad_field_1": "actually decent text"},
    #           {"_id": "fr2452", "good_field_1": 000, "good_field_2": 3.65}],
    #          ({"good_field_1", "good_field_2"}, {"bad_field_1"})
    #          ),
    #     ]
    #     for docs, (good_fields, bad_fields) in docs_results:
    #         # good_fields should appear in the mapping.
    #         # bad_fields should not
    #         tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #             index_name=self.index_name_1, docs=docs, auto_refresh=True,
    #             device="cpu"
    #         )
    #                                     )
    #         ii = backend.get_index_info(config=self.config, index_name=self.index_name_1)
    #         customer_props = {field_name for field_name in ii.get_text_properties()}
    #         reduced_vector_props = {field_name.replace(TensorField.vector_prefix, '')
    #                                 for field_name in ii.get_text_properties()}
    #         for field in good_fields:
    #             assert field in customer_props
    #             assert field in reduced_vector_props
    #
    #         for field in bad_fields:
    #             assert field not in customer_props
    #             assert field not in reduced_vector_props
    #
    # def test_mappings_arent_updated_images(self):
    #     """if an image isn't added properly, we need to ensure that
    #     it's mappings don't get added to index mappings
    #
    #     Test for:
    #         - images with chunking
    #         - images without chunking
    #     """
    #     image_index_configs = [
    #         # NO CHUNKING
    #         {
    #             IndexSettingsField.index_defaults: {
    #                 IndexSettingsField.model: "ViT-B/16",
    #                 IndexSettingsField.treat_urls_and_pointers_as_images: True
    #             }
    #         },
    #         # WITH CHUNKING
    #         {
    #             IndexSettingsField.index_defaults: {
    #                 IndexSettingsField.model: "ViT-B/16",
    #                 IndexSettingsField.treat_urls_and_pointers_as_images: True,
    #                 IndexSettingsField.normalize_embeddings: True,
    #                 IndexSettingsField.image_preprocessing: {IndexSettingsField.patch_method: "frcnn"},
    #             },
    #         }
    #     ]
    #     for image_index_config in image_index_configs:
    #         tensor_search.create_vector_index(
    #             config=self.config, index_name=self.index_name_2,
    #             index_settings=image_index_config)
    #         docs_results = [
    #             # invalid images
    #             ([{"_id": "123",
    #                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
    #               {"_id": "789",
    #                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
    #               {"_id": "456", "image_field_3": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
    #              ({"image_field_1", "image_field_2"}, {"image_field_3"})
    #              ),
    #         ]
    #         for docs, (good_fields, bad_fields) in docs_results:
    #             # good_fields should appear in the mapping.
    #             # bad_fields should not
    #             tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #                 index_name=self.index_name_2, docs=docs, auto_refresh=True, device="cpu")
    #                                         )
    #             ii = backend.get_index_info(config=self.config, index_name=self.index_name_2)
    #             customer_props = {field_name for field_name in ii.get_text_properties()}
    #             reduced_vector_props = {field_name.replace(TensorField.vector_prefix, '')
    #                                     for field_name in ii.get_text_properties()}
    #             for field in good_fields:
    #                 assert field in customer_props
    #                 assert field in reduced_vector_props
    #
    #             for field in bad_fields:
    #                 assert field not in customer_props
    #                 assert field not in reduced_vector_props
    #         tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
    #
    # def test_put_document_override_non_tensor_field(self):
    #     docs_ = [{"_id": "789", "Title": "Story of Alice Appleseed", "Description": "Alice grew up in Houston, Texas."}]
    #     tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #         index_name=self.index_name_1, docs=docs_, auto_refresh=True, non_tensor_fields=["Title"], device="cpu"))
    #     tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #         index_name=self.index_name_1, docs=docs_, auto_refresh=True, device="cpu"))
    #     resp = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="789",
    #                                             show_vectors=True)
    #
    #     assert len(resp[enums.TensorField.tensor_facets]) == 2
    #     assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][0]
    #     assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][1]
    #     # the order doesn't really matter. We can test for both orderings if this breaks in the future
    #     assert "Title" in resp[enums.TensorField.tensor_facets][0]
    #     assert "Description" in resp[enums.TensorField.tensor_facets][1]
    #
    # def test_add_document_with_non_tensor_field(self):
    #     docs_ = [{"_id": "789", "Title": "Story of Alice Appleseed", "Description": "Alice grew up in Houston, Texas."}]
    #     tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #         index_name=self.index_name_1, docs=docs_, auto_refresh=True, non_tensor_fields=["Title"], device="cpu"
    #     ))
    #     resp = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="789",
    #                                             show_vectors=True)
    #
    #     assert len(resp[enums.TensorField.tensor_facets]) == 1
    #     assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][0]
    #     assert "Title" not in resp[enums.TensorField.tensor_facets][0]
    #     assert "Description" in resp[enums.TensorField.tensor_facets][0]
    #
    # def test_add_document_with_tensor_fields(self):
    #     docs_ = [{"_id": "789", "Title": "Story of Alice Appleseed", "Description": "Alice grew up in Houston, Texas."}]
    #     tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
    #         index_name=self.index_name_1, docs=docs_, auto_refresh=True, tensor_fields=['Title'],
    #         non_tensor_fields=None, device="cpu"
    #     ))
    #     resp = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="789",
    #                                             show_vectors=True)
    #
    #     assert len(resp[enums.TensorField.tensor_facets]) == 1
    #     assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][0]
    #     assert "Title" in resp[enums.TensorField.tensor_facets][0]
    #     assert "Description" not in resp[enums.TensorField.tensor_facets][0]
    #
    # def test_doc_too_large(self):
    #     max_size = 400000
    #     mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}
    #
    #     @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
    #     def run():
    #         update_res = tensor_search.add_documents(
    #             config=self.config, add_docs_params=AddDocsParams(
    #                 index_name=self.index_name_1, docs=[
    #                     {"_id": "123", 'Bad field': "edf " * (max_size // 4)},
    #                     {"_id": "789", "Breaker": "abc " * ((max_size // 4) - 500)},
    #                     {"_id": "456", "Luminosity": "exc " * (max_size // 4)},
    #                 ],
    #                 auto_refresh=True, device="cpu"
    #             ))
    #         items = update_res['items']
    #         assert update_res['errors']
    #         assert 'error' in items[0] and 'error' in items[2]
    #         assert 'doc_too_large' == items[0]['code'] and ('doc_too_large' == items[0]['code'])
    #         assert items[1]['result'] == 'created'
    #         assert 'error' not in items[1]
    #         return True
    #
    #     assert run()
    #
    # def test_doc_too_large_single_doc(self):
    #     max_size = 400000
    #     mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}
    #
    #     @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
    #     def run():
    #         update_res = tensor_search.add_documents(
    #             config=self.config, add_docs_params=AddDocsParams(
    #                 index_name=self.index_name_1, docs=[
    #                     {"_id": "123", 'Bad field': "edf " * (max_size // 4)},
    #                 ],
    #                 auto_refresh=True, use_existing_tensors=True, device="cpu")
    #         )
    #         items = update_res['items']
    #         assert update_res['errors']
    #         assert 'error' in items[0]
    #         assert 'doc_too_large' == items[0]['code']
    #         return True
    #
    #     assert run()
    #
    # def test_doc_too_large_none_env_var(self):
    #     for env_dict in [dict()]:
    #         @mock.patch.dict(os.environ, {**os.environ, **env_dict})
    #         def run():
    #             update_res = tensor_search.add_documents(
    #                 config=self.config, add_docs_params=AddDocsParams(
    #                     index_name=self.index_name_1, docs=[
    #                         {"_id": "123", 'Some field': "Some content"},
    #                     ],
    #                     auto_refresh=True, use_existing_tensors=True, device="cpu"
    #                 ))
    #             items = update_res['items']
    #             assert not update_res['errors']
    #             assert 'error' not in items[0]
    #             assert items[0]['result'] in ['created', 'updated']
    #             return True
    #
    #         assert run()
    #
    # def test_non_tensor_field_list(self):
    #     test_doc = {"_id": "123", "my_list": ["data1", "mydata"], "myfield2": "mydata2"}
    #     tensor_search.add_documents(
    #         self.config,
    #         add_docs_params=AddDocsParams(
    #             docs=[test_doc], auto_refresh=True,
    #             index_name=self.index_name_1, non_tensor_fields=['my_list'], device="cpu"
    #         ))
    #     doc_w_facets = tensor_search.get_document_by_id(
    #         self.config, index_name=self.index_name_1, document_id='123', show_vectors=True)
    #
    #     # check tensor facets:
    #     assert len(doc_w_facets[TensorField.tensor_facets]) == 1
    #     assert 'myfield2' in doc_w_facets[TensorField.tensor_facets][0]
    #     assert doc_w_facets['my_list'] == test_doc['my_list']
    #     assert doc_w_facets['myfield2'] == test_doc['myfield2']
    #
    #     assert 1 == len(doc_w_facets[TensorField.tensor_facets])
    #     assert doc_w_facets[TensorField.tensor_facets][0]["myfield2"] == "mydata2"
    #
    #     # check OpenSearch, to ensure the list got added as a filter field
    #     original_doc = requests.get(
    #         url=F"{self.endpoint}/{self.index_name_1}/_doc/123",
    #         verify=False
    #     ).json()
    #     assert len(original_doc['_source']['__chunks']) == 1
    #     myfield2_chunk = original_doc['_source']['__chunks'][0]
    #     #     check if the chunk represents the tensorsied "mydata2" field
    #     assert myfield2_chunk['__field_name'] == 'myfield2'
    #     assert myfield2_chunk['__field_content'] == 'mydata2'
    #     assert isinstance(myfield2_chunk[TensorField.marqo_knn_field], list)
    #     #      Check if all filter fields are  there (inc. the non tensorised my_list):
    #     assert myfield2_chunk['my_list'] == ['data1', 'mydata']
    #     assert myfield2_chunk['myfield2'] == 'mydata2'
    #
    #     # check index info. my_list needs to be keyword within each chunk
    #     index_info = tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1)
    #     assert index_info.properties['my_list']['type'] == 'text'
    #     assert index_info.properties['myfield2']['type'] == 'text'
    #     assert index_info.properties['__chunks']['properties']['my_list']['type'] == 'keyword'
    #     assert index_info.properties['__chunks']['properties']['myfield2']['type'] == 'keyword'
    #     assert index_info.properties['__chunks']['properties'][TensorField.marqo_knn_field]['type'] == 'knn_vector'
    #
    # def test_no_tensor_field_replace(self):
    #     # test replace and update workflows
    #     tensor_search.add_documents(
    #         self.config, add_docs_params=AddDocsParams(
    #             docs=[{"_id": "123", "myfield": "mydata", "myfield2": "mydata2"}],
    #             auto_refresh=True, index_name=self.index_name_1, device="cpu"
    #         )
    #     )
    #     tensor_search.add_documents(
    #         self.config,
    #         add_docs_params=AddDocsParams(
    #             docs=[{"_id": "123", "myfield": "mydata"}],
    #             auto_refresh=True, index_name=self.index_name_1,
    #             non_tensor_fields=["myfield"], device="cpu"
    #         )
    #     )
    #     doc_w_facets = tensor_search.get_document_by_id(
    #         self.config, index_name=self.index_name_1, document_id='123', show_vectors=True)
    #     assert doc_w_facets[TensorField.tensor_facets] == []
    #     assert 'myfield2' not in doc_w_facets
    #
    # def test_no_tensor_field_on_empty_ix(self):
    #     tensor_search.add_documents(
    #         self.config, add_docs_params=AddDocsParams(
    #             docs=[{"_id": "123", "myfield": "mydata"}],
    #             auto_refresh=True, index_name=self.index_name_1,
    #             non_tensor_fields=["myfield"], device="cpu"
    #         )
    #     )
    #     doc_w_facets = tensor_search.get_document_by_id(
    #         self.config, index_name=self.index_name_1, document_id='123', show_vectors=True)
    #     assert doc_w_facets[TensorField.tensor_facets] == []
    #     assert 'myfield' in doc_w_facets
    #
    # def test_no_tensor_field_on_empty_ix_other_field(self):
    #     tensor_search.add_documents(
    #         self.config, add_docs_params=AddDocsParams(
    #             docs=[{"_id": "123", "myfield": "mydata", "myfield2": "mydata"}],
    #             auto_refresh=True, index_name=self.index_name_1,
    #             non_tensor_fields=["myfield"], device="cpu"
    #         )
    #     )
    #     doc_w_facets = tensor_search.get_document_by_id(
    #         self.config, index_name=self.index_name_1, document_id='123', show_vectors=True)
    #     assert len(doc_w_facets[TensorField.tensor_facets]) == 1
    #     assert 'myfield2' in doc_w_facets[TensorField.tensor_facets][0]
    #     assert 'myfield' not in doc_w_facets[TensorField.tensor_facets][0]
    #     assert 'myfield' in doc_w_facets
    #     assert 'myfield2' in doc_w_facets
    #
    # def test_various_image_count(self):
    #     hippo_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #
    #     def _check_get_docs(doc_count, some_field_value):
    #         approx_half = math.floor(doc_count / 2)
    #         get_res = tensor_search.get_documents_by_ids(
    #             config=self.config, index_name=self.index_name_1,
    #             document_ids=[str(n) for n in (0, approx_half, doc_count - 1)],
    #             show_vectors=True
    #         )
    #         for d in get_res['results']:
    #             assert d['_found'] is True
    #             assert d['some_field'] == some_field_value
    #             assert d['location'] == hippo_url
    #             assert {'_embedding', 'location', 'some_field'} == functools.reduce(lambda x, y: x.union(y),
    #                                                                                 [list(facet.keys()) for facet in
    #                                                                                  d['_tensor_facets']], set())
    #             for facet in d['_tensor_facets']:
    #                 if 'location' in facet:
    #                     assert facet['location'] == hippo_url
    #                 elif 'some_field':
    #                     assert facet['some_field'] == some_field_value
    #                 assert isinstance(facet['_embedding'], list)
    #                 assert len(facet['_embedding']) > 0
    #         return True
    #
    #     doc_counts = 1, 2, 25
    #     for c in doc_counts:
    #         try:
    #             tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
    #         except IndexNotFoundError as s:
    #             pass
    #         tensor_search.create_vector_index(
    #             config=self.config, index_name=self.index_name_1,
    #             index_settings={
    #                 IndexSettingsField.index_defaults: {
    #                     IndexSettingsField.model: "random",
    #                     IndexSettingsField.treat_urls_and_pointers_as_images: True
    #                 }
    #             }
    #         )
    #         res1 = tensor_search.add_documents(
    #             self.config,
    #             add_docs_params=AddDocsParams(
    #                 docs=[{"_id": str(doc_num),
    #                        "location": hippo_url,
    #                        "some_field": "blah"} for doc_num in range(c)],
    #                 auto_refresh=True, index_name=self.index_name_1, device="cpu"
    #             )
    #         )
    #         assert c == tensor_search.get_stats(self.config,
    #                                             index_name=self.index_name_1)['numberOfDocuments']
    #         assert not res1['errors']
    #         assert _check_get_docs(doc_count=c, some_field_value='blah')
    #         res2 = tensor_search.add_documents(
    #             self.config,
    #             add_docs_params=AddDocsParams(
    #                 docs=[{"_id": str(doc_num),
    #                        "location": hippo_url,
    #                        "some_field": "blah2"} for doc_num in range(c)],
    #                 auto_refresh=True, index_name=self.index_name_1,
    #                 non_tensor_fields=["myfield"], device="cpu"
    #             )
    #         )
    #         assert not res2['errors']
    #         assert c == tensor_search.get_stats(self.config,
    #                                             index_name=self.index_name_1)['numberOfDocuments']
    #         assert _check_get_docs(doc_count=c, some_field_value='blah2')
    #
    # def test_images_non_tensor_fields_count(self):
    #     hippo_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #
    #     def _check_get_docs(doc_count, some_field_value):
    #         approx_half = math.floor(doc_count / 2)
    #         get_res = tensor_search.get_documents_by_ids(
    #             config=self.config, index_name=self.index_name_1,
    #             document_ids=[str(n) for n in (0, approx_half, doc_count - 1)],
    #             show_vectors=True
    #         )
    #         for d in get_res['results']:
    #             assert d['_found'] is True
    #             assert d['some_field'] == some_field_value
    #             assert d['location'] == hippo_url
    #             # location is not present:
    #             assert {'_embedding', 'some_field'} == functools.reduce(lambda x, y: x.union(y),
    #                                                                     [list(facet.keys()) for facet in
    #                                                                      d['_tensor_facets']], set())
    #         return True
    #
    #     doc_counts = 1, 20, 23
    #     for c in doc_counts:
    #         try:
    #             tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
    #         except IndexNotFoundError as s:
    #             pass
    #         tensor_search.create_vector_index(
    #             config=self.config, index_name=self.index_name_1,
    #             index_settings={
    #                 IndexSettingsField.index_defaults: {
    #                     IndexSettingsField.model: "random",
    #                     IndexSettingsField.treat_urls_and_pointers_as_images: True
    #                 }
    #             }
    #         )
    #         res1 = tensor_search.add_documents(
    #             self.config,
    #             add_docs_params=AddDocsParams(
    #                 docs=[{"_id": str(doc_num),
    #                        "location": hippo_url,
    #                        "some_field": "blah"} for doc_num in range(c)],
    #                 auto_refresh=True, index_name=self.index_name_1,
    #                 non_tensor_fields=["location"], device="cpu"
    #             ))
    #         assert c == tensor_search.get_stats(self.config,
    #                                             index_name=self.index_name_1)['numberOfDocuments']
    #         assert not res1['errors']
    #         assert _check_get_docs(doc_count=c, some_field_value='blah')
    #         res2 = tensor_search.add_documents(
    #             self.config,
    #             add_docs_params=AddDocsParams(
    #                 docs=[{"_id": str(doc_num),
    #                        "location": hippo_url,
    #                        "some_field": "blah2"} for doc_num in range(c)],
    #                 auto_refresh=True, index_name=self.index_name_1,
    #                 non_tensor_fields=["location"], device="cpu"
    #             )
    #         )
    #         assert not res2['errors']
    #         assert c == tensor_search.get_stats(self.config,
    #                                             index_name=self.index_name_1)['numberOfDocuments']
    #         assert _check_get_docs(doc_count=c, some_field_value='blah2')
    #
    # def test_image_download_timeout(self):
    #     mock_get = mock.MagicMock()
    #     mock_get.side_effect = requests.exceptions.RequestException
    #
    #     @mock.patch('requests.get', mock_get)
    #     def run():
    #         image_repo = dict()
    #         add_docs.threaded_download_images(
    #             allocated_docs=[
    #                 {"Title": "frog", "Desc": "blah"}, {"Title": "Dog", "Loc": "https://google.com/my_dog.png"}],
    #             image_repo=image_repo,
    #             non_tensor_fields=[],
    #             tensor_fields=None,
    #             image_download_headers={}
    #         )
    #         assert list(image_repo.keys()) == ['https://google.com/my_dog.png']
    #         assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
    #         return True
    #
    #     assert run()
    #
    # def test_image_download(self):
    #     image_repo = dict()
    #     good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #     test_doc = {
    #         'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
    #         'field_2': good_url
    #     }
    #
    #     add_docs.threaded_download_images(
    #         allocated_docs=[test_doc],
    #         image_repo=image_repo,
    #         non_tensor_fields=[],
    #         tensor_fields=None,
    #         image_download_headers={}
    #     )
    #     assert len(image_repo) == 2
    #     assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
    #     assert isinstance(image_repo[good_url], types.ImageType)
    #
    # def test_threaded_download_images_non_tensor_field(self):
    #     """Tests add_docs.threaded_download_images(). URLs in non_tensor_fields should not be downloaded """
    #     good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #     bad_url = 'https://google.com/my_dog.png'
    #     examples = [
    #         ([{
    #             'field_1': bad_url,
    #             'field_2': good_url
    #         }], {
    #              bad_url: PIL.UnidentifiedImageError,
    #              good_url: types.ImageType
    #          }),
    #         ([{
    #             'nt_1': bad_url,
    #             'nt_2': good_url
    #         }], {}),
    #         ([{
    #             'field_1': bad_url,
    #             'nt_1': good_url
    #         }], {
    #              bad_url: PIL.UnidentifiedImageError,
    #          }),
    #         ([{
    #             'nt_2': bad_url,
    #             'field_2': good_url
    #         }], {
    #              good_url: types.ImageType
    #          }),
    #     ]
    #     for docs, expected_repo_structure in examples:
    #         image_repo = dict()
    #         add_docs.threaded_download_images(
    #             allocated_docs=docs,
    #             image_repo=image_repo,
    #             non_tensor_fields=['nt_1', 'nt_2'],
    #             tensor_fields=None,
    #             image_download_headers={}
    #         )
    #         assert len(expected_repo_structure) == len(image_repo)
    #         for k in expected_repo_structure:
    #             assert isinstance(image_repo[k], expected_repo_structure[k])
    #
    # def test_download_images_non_tensor_field(self):
    #     """tests add_docs.download_images(). URLs in non_tensor_fields should not be downloaded """
    #     good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #     bad_url = 'https://google.com/my_dog.png'
    #     examples = [
    #         ([{
    #             'field_1': bad_url,
    #             'field_2': good_url
    #         }], {
    #              bad_url: PIL.UnidentifiedImageError,
    #              good_url: types.ImageType
    #          }),
    #         ([{
    #             'nt_1': bad_url,
    #             'nt_2': good_url
    #         }], {}),
    #         ([{
    #             'field_1': bad_url,
    #             'nt_1': good_url
    #         }], {
    #              bad_url: PIL.UnidentifiedImageError,
    #          }),
    #         ([{
    #             'nt_2': bad_url,
    #             'field_2': good_url
    #         }], {
    #              good_url: types.ImageType
    #          }),
    #     ]
    #     with mock.patch('PIL.Image.Image.close') as mock_close:
    #         for docs, expected_repo_structure in examples:
    #             with add_docs.download_images(
    #                     docs=docs,
    #                     thread_count=20,
    #                     non_tensor_fields=('nt_1', 'nt_2'),
    #                     image_download_headers={},
    #                     tensor_fields=None
    #             ) as image_repo:
    #                 assert len(expected_repo_structure) == len(image_repo)
    #                 for k in expected_repo_structure:
    #                     assert isinstance(image_repo[k], expected_repo_structure[k])
    #
    #         # Context manager must have closed all valid images
    #         assert mock_close.call_count == 2



