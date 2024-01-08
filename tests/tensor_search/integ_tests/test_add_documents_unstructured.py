import functools
import math
import os
import uuid
from unittest import mock
from unittest.mock import patch

import PIL
import pytest
import requests

from marqo.api.exceptions import IndexNotFoundError, BadRequestError
from marqo.core.models.marqo_index import *
from marqo.s2_inference import types
from marqo.tensor_search import add_docs
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestAddDocumentsUnstructured(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request()
        default_text_index_encoded_name = cls.unstructured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', '')
        )

        default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_random_model = cls.unstructured_marqo_index_request(
            model=Model(name='random'),
            treat_urls_and_pointers_as_images=True
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
            default_text_index_encoded_name,
            default_image_index,
            image_index_with_chunking,
            image_index_with_random_model
        ])

        cls.default_text_index = default_text_index.name
        cls.default_text_index_encoded_name = default_text_index_encoded_name.name
        cls.default_image_index = default_image_index.name
        cls.image_index_with_chunking = image_index_with_chunking.name
        cls.image_index_with_random_model = image_index_with_random_model.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_add_plain_id_field(self):
        """
        Plain id field works
        """
        tests = [
            (self.default_text_index, 'Standard index name'),
            (self.default_text_index_encoded_name, 'Index name requiring encoding'),
        ]
        for index_name, desc in tests:
            with self.subTest(desc):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index,
                        docs=[{
                            "_id": "123",
                            "title": "content 1",
                            "desc": "content 2. blah blah blah"
                        }],
                        device="cpu", tensor_fields=["title"]
                    )
                )
                self.assertEqual(
                    {
                        "_id": "123",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    tensor_search.get_document_by_id(
                        config=self.config, index_name=self.default_text_index,
                        document_id="123"
                    )
                )

    def test_add_documents_dupe_ids(self):
        """
        Only the latest added document is returned
        """

        # Add once to get vectors
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[{
                    "_id": "1",
                    "title": "doc 123"
                }],
                device="cpu", tensor_fields=["title"]
            )
        )
        tensor_facets = tensor_search.get_document_by_id(
            config=self.config, index_name=self.default_text_index,
            document_id="1", show_vectors=True)['_tensor_facets']

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[
                    {
                        "_id": "2",
                        "title": "doc 000"
                    }
                ],
                device="cpu", tensor_fields=["title"]
            )
        )
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[
                    {
                        "_id": "2",
                        "title": "doc 123"
                    }
                ],
                device="cpu", tensor_fields=["title"]
            )
        )

        expected_doc = {
            "_id": "2",
            "title": "doc 123",
            '_tensor_facets': tensor_facets
        }
        actual_doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.default_text_index,
            document_id="2", show_vectors=True)

        self.assertEqual(expected_doc, actual_doc)

    def test_add_documents_with_missing_index_fails(self):
        rand_index = 'a' + str(uuid.uuid4()).replace('-', '')

        with pytest.raises(IndexNotFoundError):
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=rand_index, docs=[{"abc": "def"}], device="cpu"
                )
            )

    def test_add_documents_whitespace(self):
        """
        Indexing fields consisting of only whitespace works
        """
        docs = [
            {"title": ""},
            {"title": " "},
            {"title": "  "},
            {"title": "\r"},
            {"title": "\r "},
            {"title": "\r\r"},
            {"title": "\r\t\n"},
        ]
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=docs, device="cpu", tensor_fields=[]
            )
        )
        count = self.pyvespa_client.query(
            {"yql": f"select * from sources {self.default_text_index} where true limit 0"}
        ).json["root"]["fields"]["totalCount"]

        assert count == len(docs)

    def test_add_docs_response_format(self):
        add_res = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "123",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    {
                        "_id": "456",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    {
                        "_id": "789",
                        "tags": [1, 'str']  # mixed types, error
                    }
                ],
                device="cpu", tensor_fields=[]
            )
        )
        assert "errors" in add_res
        assert "processingTimeMs" in add_res
        assert "index_name" in add_res
        assert "items" in add_res

        assert add_res["processingTimeMs"] > 0
        assert add_res["errors"] is True
        assert add_res["index_name"] == self.default_text_index

        for item in add_res["items"]:
            assert "_id" in item
            assert "status" in item
            assert (item['status'] == 200) ^ ("error" in item and "code" in item)

        assert [item['status'] for item in add_res["items"]] == [200, 200, 400]

    def test_add_documents_validation(self):
        """
        Invalid documents return errors
        """
        bad_doc_args = [
            [{"_id": "to_fail_123", "title": dict()}],  # dict for non-combination field
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]}],  # tensor field list
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]},  # tensor field list
             {"_id": "to_pass_123", "title": 'some_content'}],
            [{"_id": "to_fail_123", "tags": [{"abc": "678"}]}],  # list of dict
            [{"_id": "to_fail_123", "title": {"abc": "234"}}],  # dict for non-combination field
            [{"_id": "to_fail_123", "title": {"abc": "234"}},  # dict for non-combination field
             {"_id": "to_pass_123", "title": 'some_content'}],
            # other checking:
            [{"title": {1243}, "_id": "to_fail_123"}],  # invalid json
            [{"title": None, "_id": "to_fail_123"}],  # None not a valid type
            [{"_id": "to_fail_123", "title": [None], "desc": "123"},  # None not a valid type
             {"_id": "to_fail_567", "title": "finnne", 123: "heehee"}],  # Field name int
            [{"_id": "to_fail_123", "title": [None], "desc": "123"},  # List of None
             {"_id": "to_fail_567", "title": AssertionError}],  # Pointer as value, invalid json
            [{"_id": "to_fail_567", "tags": max}]  # Invalid json
        ]

        # For replace, check with use_existing_tensors True and False
        for use_existing_tensors_flag in (True, False):
            for bad_doc_arg in bad_doc_args:
                with self.subTest(msg=f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'):
                    add_res = tensor_search.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=self.default_text_index, docs=bad_doc_arg,
                            use_existing_tensors=use_existing_tensors_flag, device="cpu",
                            tensor_fields=["title"]
                        )
                    )
                    assert add_res['errors'] is True
                    assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])
                    assert all([item['status'] == 200
                                for item in add_res['items'] if item['_id'].startswith('to_pass')])

    def test_add_documents_id_validation(self):
        """
        Invalid document IDs return errors
        """
        bad_doc_args = [
            # Wrong data types for ID
            # Tuple: (doc_list, number of docs that should succeed)
            ([{"_id": {}, "title": "yyy"}], 0),
            ([{"_id": dict(), "title": "yyy"}], 0),
            ([{"_id": [1, 2, 3], "title": "yyy"}], 0),
            ([{"_id": 4, "title": "yyy"}], 0),
            ([{"_id": None, "title": "yyy"}], 0),
            ([{"_id": "proper id", "title": "yyy"},
              {"_id": ["bad", "id"], "title": "zzz"},
              {"_id": "proper id 2", "title": "xxx"}], 2)
        ]

        # For replace, check with use_existing_tensors True and False
        for use_existing_tensors_flag in (True, False):
            for bad_doc_arg in bad_doc_args:
                with self.subTest(f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'):
                    add_res = tensor_search.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=self.default_text_index, docs=bad_doc_arg[0],
                            use_existing_tensors=use_existing_tensors_flag, device="cpu", tensor_fields=["title"]
                        )
                    )
                    assert add_res[
                               'errors'] is True, f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'
                    succeeded_count = 0
                    for item in add_res['items']:
                        if item['status'] == 200:
                            succeeded_count += 1
                        else:
                            assert 'Document _id must be a string type' in item['error']

                    assert succeeded_count == bad_doc_arg[1]

    def test_add_documents_list_success(self):
        good_docs = [
            [{"_id": "to_fail_123", "tags": ["wow", "this", "is"]}]
        ]
        for bad_doc_arg in good_docs:
            add_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index,
                    docs=bad_doc_arg,
                    device="cpu",
                    tensor_fields=[],
                )
            )
            assert add_res['errors'] is False

    def test_add_documents_list_data_type_validation(self):
        """These bad docs should return errors"""
        self.tags_ = [
            [{"_id": "to_fail_123", "tags": ["wow", "this", False]}],
            [{"_id": "to_fail_124", "tags": [1, None, 3]}],
            [{"_id": "to_fail_125", "tags": [{}]}]
        ]
        bad_doc_args = self.tags_
        for bad_doc_arg in bad_doc_args:
            with self.subTest(bad_doc_arg):
                add_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index,
                        docs=bad_doc_arg,
                        device="cpu",
                        tensor_fields=[],
                    )
                )
                assert add_res['errors'] is True
                assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])

    def test_add_documents_set_device(self):
        """
        Device is set correctly
        """
        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, device="cuda:22", docs=[{"title": "doc"}, {"title": "doc"}],
                    tensor_fields=["title"]
                ),
            )
            return True

        assert run()
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:22"

    def test_add_documents_empty(self):
        """
        Adding empty documents raises BadRequestError
        """
        try:
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[],
                    device="cpu")
            )
            raise AssertionError
        except BadRequestError:
            pass

    def test_resilient_add_images(self):
        """
        Various image URLs are handled correctly
        """
        image_indexs = [self.default_image_index, self.image_index_with_chunking]

        docs_results = [
            ([{"_id": "123",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", 200), ("789", 200), ("456", 400)]
             ),
            ([{"_id": "123",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("123", 200), ("789", 200), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field": "https://www.marqo.ai/this/image/doesnt/exist3.png"},
              {"_id": "456", "image_field": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400)]
             ),
        ]
        for image_index in image_indexs:
            for docs, expected_results in docs_results:
                with self.subTest(f'{expected_results} - {image_index}'):
                    add_res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_image_index, docs=docs, device="cpu", tensor_fields=["image_field"]))
                    assert len(add_res['items']) == len(expected_results)
                    for i, res_dict in enumerate(add_res['items']):
                        assert res_dict["_id"] == expected_results[i][0]
                        assert res_dict['status'] == expected_results[i][1]

    def test_add_documents_id_image_url(self):
        """
        Image URL as ID is not downloaded
        """
        docs = [{
            "_id": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "title": "wow"}
        ]

        with mock.patch('PIL.Image.open') as mock_image_open:
            tensor_search.add_documents(config=self.config,
                                        add_docs_params=AddDocsParams(
                                            index_name=self.default_image_index, docs=docs,
                                            device="cpu", tensor_fields=["title"]
                                        ))

            mock_image_open.assert_not_called()

    def test_add_documents_resilient_doc_validation(self):
        docs_results = [
            # handle empty dicts
            ([{"_id": "123", "title": "legitimate text"},
              {},
              {"_id": "456", "title": "awesome stuff!"}],
             [("123", 200), (None, 400), ('456', 200)]
             ),
            ([{}], [(None, 400)]),
            ([{}, {}], [(None, 400), (None, 400)]),
            ([{}, {}, {"title": "yep"}], [(None, 400), (None, 400), (None, 200)]),
            # handle invalid dicts
            ([{"this is a set, lmao"}, "this is a string", {"title": "yep"}],
             [(None, 400), (None, 400), (None, 200)]),
            ([1234], [(None, 400)]),
            ([None], [(None, 400)]),
            # handle invalid field names
            ([{123: "bad"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"__chunks": "bad"}, {"_id": "1511", "__vector_a": "some content"}, {"_id": "cool"},
              {"_id": "144451", "__field_content": "some content"}],
             [(None, 200), ("1511", 200), ("cool", 200), ("144451", 200)]),
            ([{123: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{None: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            # handle bad content
            ([{"title": None, "_id": "12345"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"tags": [1, 2, '3', 4], "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{"title": ("cat", "dog"), "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{"title": set(), "_id": "12345"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"title": dict(), "_id": "12345"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            # handle bad _ids
            ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}, {"bad": "hehehe", "_id": None}, {"title": "yep"},
              {"_id": (1, 2), "efgh": "abc"}, {"_id": 1.234, "cool": "wowowow"}],
             [(None, 400), ("cool", 200), (None, 400), (None, 200), (None, 400),
              (None, 400)]),
            # mixed
            ([{(1, 2, 3): set(), "_id": "12345"}, {"_id": "cool"}, {"tags": [1, 2, 3], "_id": None}, {"title": "yep"},
              {}, "abcdefgh"],
             [(None, 400), ("cool", 200), (None, 400), (None, 200), (None, 400),
              (None, 400)]),
        ]
        for docs, expected_results in docs_results:
            with self.subTest(f'{expected_results}'):
                add_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index, docs=docs,
                        device="cpu", tensor_fields=[]
                    )
                )
                self.assertEqual(len(expected_results), len(expected_results))
                for i, res_dict in enumerate(add_res['items']):
                    # if the expected id is None, then it assumed the id is
                    # generated and can't be asserted against
                    if expected_results[i][0] is not None:
                        self.assertEqual(expected_results[i][0], res_dict["_id"])
                    self.assertEqual(expected_results[i][1], res_dict['status'])

    def test_add_document_with_tensor_fields(self):
        """Ensure tensor_fields only works for title but not desc"""
        docs_ = [{"_id": "789", "title": "Story of Alice Appleseed", "desc": "Alice grew up in Houston, Texas."}]
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.default_text_index, docs=docs_, device="cpu", tensor_fields=["title"]
        ))
        resp = tensor_search.get_document_by_id(config=self.config,
                                                index_name=self.default_text_index, document_id="789",
                                                show_vectors=True)

        assert len(resp[enums.TensorField.tensor_facets]) == 1
        assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][0]
        assert "title" in resp[enums.TensorField.tensor_facets][0]
        assert "desc" not in resp[enums.TensorField.tensor_facets][0]

    def test_doc_too_large(self):
        max_size = 400000
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            update_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[
                        {"_id": "123", 'desc': "edf " * (max_size // 4)},
                        {"_id": "789", "desc": "abc " * ((max_size // 4) - 500)},
                        {"_id": "456", "desc": "exc " * (max_size // 4)},
                    ],
                    device="cpu", tensor_fields=["desc"]
                ))
            items = update_res['items']
            assert update_res['errors']
            assert 'error' in items[0] and 'error' in items[2]
            assert 'doc_too_large' == items[0]['code'] and ('doc_too_large' == items[0]['code'])
            assert items[1]['status'] == 200
            assert 'error' not in items[1]
            return True

        assert run()

    def test_doc_too_large_single_doc(self):
        max_size = 400000
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            update_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[
                        {"_id": "123", 'desc': "edf " * (max_size // 4)},
                    ],
                    use_existing_tensors=True, device="cpu", tensor_fields=[])
            )
            items = update_res['items']
            assert update_res['errors']
            assert 'error' in items[0]
            assert 'doc_too_large' == items[0]['code']
            return True

        assert run()

    def test_doc_too_large_none_env_var(self):
        """
        If MARQO_MAX_DOC_BYTES is not set, then the default is used
        """
        # TODO - Consider removing this test as indexing a standard doc is covered by many other tests
        for env_dict in [dict()]:
            @mock.patch.dict(os.environ, {**os.environ, **env_dict})
            def run():
                update_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index, docs=[
                            {"_id": "123", 'desc': "Some content"},
                        ],
                        use_existing_tensors=True, device="cpu", tensor_fields=["desc"]
                    ))
                items = update_res['items']
                assert not update_res['errors']
                assert 'error' not in items[0]
                assert items[0]['status'] == 200
                return True

            assert run()

    def test_remove_tensor_field(self):
        """
        If a document is re-indexed with a tensor field removed, the vectors are removed
        """
        # test replace and update workflows
        tensor_search.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "title": "mydata", "desc": "mydata2"}],
                index_name=self.default_text_index, device="cpu", tensor_fields=["title"]
            )
        )
        tensor_search.add_documents(
            self.config,
            add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "desc": "mydata"}],
                index_name=self.default_text_index,
                device="cpu",
                tensor_fields=[]
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert doc_w_facets[enums.TensorField.tensor_facets] == []
        assert 'title' not in doc_w_facets

    def test_no_tensor_field_on_empty_ix(self):
        """
        If a document is indexed with no tensor fields on an empty index, no vectors are added
        """
        tensor_search.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "desc": "mydata"}],
                index_name=self.default_text_index,
                device="cpu", tensor_fields=[]
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert doc_w_facets[enums.TensorField.tensor_facets] == []
        assert 'desc' in doc_w_facets

    def test_index_doc_on_empty_ix(self):
        """
        If a document is indexed with a tensor field vectors are added for the tensor field
        """
        tensor_search.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "title": "mydata", "desc": "mydata"}],
                index_name=self.default_text_index, tensor_fields=["title"],
                device="cpu"
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert len(doc_w_facets[enums.TensorField.tensor_facets]) == 1
        assert 'title' in doc_w_facets[enums.TensorField.tensor_facets][0]
        assert 'desc' not in doc_w_facets[enums.TensorField.tensor_facets][0]
        assert 'title' in doc_w_facets
        assert 'desc' in doc_w_facets

    def test_various_image_count(self):
        hippo_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'

        def _check_get_docs(doc_count, title_value):
            approx_half = math.floor(doc_count / 2)
            get_res = tensor_search.get_documents_by_ids(
                config=self.config, index_name=self.image_index_with_random_model,
                document_ids=[str(n) for n in (0, approx_half, doc_count - 1)],
                show_vectors=True
            )
            for d in get_res['results']:
                assert d['_found'] is True
                assert d['title'] == title_value
                assert d['location'] == hippo_url
                assert {'_embedding', 'location', 'title'} == functools.reduce(lambda x, y: x.union(y),
                                                                               [list(facet.keys()) for facet in
                                                                                d['_tensor_facets']], set())
                for facet in d['_tensor_facets']:
                    if 'location' in facet:
                        assert facet['location'] == hippo_url
                    elif 'title' in facet:
                        assert facet['title'] == title_value
                    assert isinstance(facet['_embedding'], list)
                    assert len(facet['_embedding']) > 0
            return True

        doc_counts = 1, 2, 25
        for c in doc_counts:
            self.clear_index_by_name(self.image_index_with_random_model)

            res1 = tensor_search.add_documents(
                self.config,
                add_docs_params=AddDocsParams(
                    docs=[{"_id": str(doc_num),
                           "location": hippo_url,
                           "title": "blah"} for doc_num in range(c)],
                    index_name=self.image_index_with_random_model, device="cpu",
                    tensor_fields=["title", "location"]
                )
            )
            self.assertEqual(
                c,
                self.config.monitoring.get_index_stats_by_name(
                    index_name=self.image_index_with_random_model
                ).number_of_documents,
            )
            self.assertFalse(res1['errors'])
            self.assertTrue(_check_get_docs(doc_count=c, title_value='blah'))

    def test_image_download_timeout(self):
        mock_get = mock.MagicMock()
        mock_get.side_effect = requests.exceptions.RequestException

        @mock.patch('requests.get', mock_get)
        def run():
            image_repo = dict()
            add_docs.threaded_download_images(
                allocated_docs=[
                    {"Title": "frog", "Desc": "blah"}, {"Title": "Dog", "Loc": "https://google.com/my_dog.png"}],
                image_repo=image_repo,
                tensor_fields=['Title', 'Desc', 'Loc'],
                image_download_headers={}
            )
            assert list(image_repo.keys()) == ['https://google.com/my_dog.png']
            assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
            return True

        assert run()

    def test_image_download(self):
        image_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_images(
            allocated_docs=[test_doc],
            image_repo=image_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={}
        )
        assert len(image_repo) == 2
        assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(image_repo[good_url], types.ImageType)

    def test_threaded_download_images_non_tensor_field(self):
        """Tests add_docs.threaded_download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: types.ImageType
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: types.ImageType
             }),
        ]
        for docs, expected_repo_structure in examples:
            image_repo = dict()
            add_docs.threaded_download_images(
                allocated_docs=docs,
                image_repo=image_repo,
                tensor_fields=['field_1', 'field_2'],
                image_download_headers={}
            )
            assert len(expected_repo_structure) == len(image_repo)
            for k in expected_repo_structure:
                assert isinstance(image_repo[k], expected_repo_structure[k])

    def test_download_images_non_tensor_field(self):
        """tests add_docs.download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: types.ImageType
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: types.ImageType
             }),
        ]
        with mock.patch('PIL.Image.Image.close') as mock_close:
            for docs, expected_repo_structure in examples:
                with add_docs.download_images(
                        docs=docs,
                        thread_count=20,
                        tensor_fields=['field_1', 'field_2'],
                        image_download_headers={},
                ) as image_repo:
                    assert len(expected_repo_structure) == len(image_repo)
                    for k in expected_repo_structure:
                        assert isinstance(image_repo[k], expected_repo_structure[k])

            # Context manager must have closed all valid images
            assert mock_close.call_count == 2

    def test_bad_tensor_fields(self):
        test_cases = [
            ({"tensor_fields": None}, "tensor_fields must be explicitly provided", "None as tensor fields"),
            ({}, "tensor_fields must be explicitly provided", "No tensor fields"),
            ({"tensor_fields": ["_id", "some"]}, "`_id` field cannot be a tensor field", "_id can't be a tensor field")
        ]
        for tensor_fields, error_message, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(BadRequestError) as e:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(index_name=self.default_text_index,
                                                      docs=[{"some": "data"}], **tensor_fields))
                self.assertIn(error_message, e.exception.message)

    def test_download_images_thread_count(self):
        """
        Test that image download thread count is respected
        """
        docs = [
            {"_id": str(i),
             "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/"
                            "assets/ai_hippo_realistic.png"
             } for i in range(10)
        ]

        for thread_count in [2, 5]:
            with patch.object(
                    add_docs, 'threaded_download_images', wraps=add_docs.threaded_download_images
            ) as mock_download_images:
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_image_index, docs=docs, device="cpu",
                        image_download_thread_count=thread_count, tensor_fields=["image_field"]
                    )
                )

                self.assertEqual(thread_count, mock_download_images.call_count)
