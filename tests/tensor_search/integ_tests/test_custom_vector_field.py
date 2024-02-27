from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search import enums
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity, ScoreModifier
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.models.search import SearchContext
import numpy as np
import requests
import json
from unittest import mock
from unittest.mock import patch
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.api.exceptions import MarqoWebError, IndexNotFoundError, InvalidArgError, DocumentNotFoundError
from marqo.core.models.marqo_index import *
import os
import pprint
import unittest


class TestCustomVectorField(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Custom settings indexes
        unstructured_custom_index = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True,
            normalize_embeddings=False,
            distance_metric=DistanceMetric.Angular
        )

        structured_custom_index = cls.structured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            normalize_embeddings=False,
            distance_metric=DistanceMetric.Angular,
            fields=[
                FieldRequest(name="my_custom_vector", type="custom_vector"),
                FieldRequest(name="text_field", type="text")
            ],  # TODO: Fill in fields
            tensor_fields=["my_custom_vector", "text_field"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_custom_index,
            structured_custom_index
        ])

    def setUp(self):
        self.mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            }
        }

        # Using arbitrary values so they're easy to eyeball
        self.random_vector_1 = [1. for _ in range(512)]
        self.random_vector_2 = [i*2 for i in range(512)]
        self.random_vector_3 = [1 / (i + 1) for i in range(512)]

        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_add_documents_with_custom_vector_field(self):
        """
        TODO: revamp or remove
        Add a document with a custom vector field:
        mock HTTP call
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
        mock_post = mock.MagicMock()
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1',
                                                                                    '_id': '0', '_version': 1,
                                                                                    'result': 'created',
                                                                                    '_shards': {'total': 1,
                                                                                                'successful': 1,
                                                                                                'failed': 0},
                                                                                    '_seq_no': 0, '_primary_term': 1,
                                                                                    'status': 201}}]}

        #@mock.patch("marqo._httprequests.HttpRequests.post", mock_post)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.indexes[0].name,
                    docs=[{
                        "_id": "0",
                        "my_custom_vector": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector_1
                        }
                    }],
                    device="cpu",
                    mappings=self.mappings,
                    tensor_fields=["my_custom_vector"]
                )
            )
            print(f"DEBUG: index name is {self.indexes[0].name}")
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2  # 2nd call is the refresh

        post_args, post_kwargs = call_args[0]
        request_body_lines = [json.loads(line) for line in post_kwargs["body"].splitlines() if line]

        # Confirm content was used as custom field
        # First line [0] is index command, Second line [1] is the document itself
        assert request_body_lines[1]["my_custom_vector"] == "custom content is here!!"
        assert "vector" not in request_body_lines[1]

        assert len(request_body_lines[1]["__chunks"]) == 1

        for chunk in request_body_lines[1]["__chunks"]:
            # Confirm chunk metadata are all correct
            assert chunk["my_custom_vector"] == "custom content is here!!"

            # Confirm chunk data is correct
            if chunk["__field_name"] == "my_custom_vector":
                assert chunk["__field_content"] == "custom content is here!!"
                assert chunk["__vector_marqo_knn_field"] == self.random_vector_1
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")

    @unittest.skip
    def test_add_documents_with_custom_vector_field_no_content(self):
        # TODO - fix
        """
        Add a document with a custom vector field with no content:
        Content should be autofilled with ""
        mock HTTP call
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
        mock_post = mock.MagicMock()
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1',
                                                                                    '_id': '0', '_version': 1,
                                                                                    'result': 'created',
                                                                                    '_shards': {'total': 1,
                                                                                                'successful': 1,
                                                                                                'failed': 0},
                                                                                    '_seq_no': 0, '_primary_term': 1,
                                                                                    'status': 201}}]}

        @mock.patch("marqo._httprequests.HttpRequests.post", mock_post)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1,
                    docs=[{
                        "_id": "0",
                        "my_custom_vector": {
                            "vector": self.random_vector_1
                        }
                    }],
                    device="cpu", mappings=self.mappings
                )
            )
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2  # 2nd call is the refresh

        post_args, post_kwargs = call_args[0]
        request_body_lines = [json.loads(line) for line in post_kwargs["body"].splitlines() if line]

        # Confirm content is ""
        # First line [0] is index command, Second line [1] is the document itself
        assert request_body_lines[1]["my_custom_vector"] == ""
        assert len(request_body_lines[1]["__chunks"]) == 1
        for chunk in request_body_lines[1]["__chunks"]:
            # Confirm chunk metadata are all correct
            assert chunk["my_custom_vector"] == ""

            # Confirm chunk data is correct
            if chunk["__field_name"] == "my_custom_vector":
                assert chunk["__field_content"] == ""
                assert chunk["__vector_marqo_knn_field"] == self.random_vector_1
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")

    @unittest.skip
    def test_add_documents_with_custom_vector_field_backend_updated(self):
        """
        Add a document with a custom vector field:
        New field added to backend properties
        """
        # TODO - fix
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "my_custom_vector": {
                        "content": "custom content is here!!",
                        "vector": self.random_vector_1
                    }
                }],
                device="cpu", mappings=self.mappings
            )
        )

        # Confirm backend was updated
        index_info = tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1)
        assert index_info.properties['my_custom_vector']['type'] == 'text'  # It's text because content is stored here
        assert index_info.properties['__chunks']['properties']['my_custom_vector']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties'][enums.TensorField.marqo_knn_field]['type'] == 'knn_vector'

    @unittest.skip
    def test_add_documents_with_different_field_types(self):
        """
        Makes sure custom vector field doesn't mess up other kinds of fields
        Add a document with a custom vector field, multimodal, and standard:
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
        # TODO - fix
        # Mixed mapping to test both multimodal and custom vector
        mixed_mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            },
            "my_multimodal": {
                "type": "multimodal_combination",
                "weights": {
                    "text": 0.4,
                    "image": 0.6
                }
            }
        }
        mock_post = mock.MagicMock()
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1',
                                                                                    '_id': '0', '_version': 1,
                                                                                    'result': 'created',
                                                                                    '_shards': {'total': 1,
                                                                                                'successful': 1,
                                                                                                'failed': 0},
                                                                                    '_seq_no': 0, '_primary_term': 1,
                                                                                    'status': 201}}]}

        @mock.patch("marqo._httprequests.HttpRequests.post", mock_post)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1,
                    docs=[{
                        "_id": "0",
                        "text_field": "blah",
                        "my_custom_vector": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector_1
                        },
                        "my_multimodal": {
                            "text": "multimodal text",
                            "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                        }
                    }],
                    device="cpu", mappings=mixed_mappings
                )
            )
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2  # 2nd call is the refresh

        post_args, post_kwargs = call_args[0]
        request_body_lines = [json.loads(line) for line in post_kwargs["body"].splitlines() if line]

        # Confirm content was used as custom field
        # First line [0] is index command, Second line [1] is the document itself
        assert request_body_lines[1]["my_custom_vector"] == "custom content is here!!"
        assert "vector" not in request_body_lines[1]
        assert request_body_lines[1]["text_field"] == "blah"
        assert request_body_lines[1]["my_multimodal"] == {
            "text": "multimodal text",
            "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
        }

        for chunk in request_body_lines[1]["__chunks"]:
            # Confirm chunk metadata are all correct
            assert chunk["my_custom_vector"] == "custom content is here!!"
            assert chunk["text_field"] == "blah"
            assert chunk["my_multimodal"] == {
                "text": "multimodal text",
                "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
            }

            # Confirm chunk data is correct
            if chunk["__field_name"] == "my_custom_vector":
                assert chunk["__field_content"] == "custom content is here!!"
                assert chunk["__vector_marqo_knn_field"] == self.random_vector_1
            elif chunk["__field_name"] == "text_field":
                assert chunk["__field_content"] == "blah"
            elif chunk["__field_name"] == "my_multimodal":
                assert chunk["__field_content"] == json.dumps({
                    "text": "multimodal text",
                    "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                })
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")

    @unittest.skip
    def test_add_documents_with_different_field_types_backend_updated(self):
        """
        Makes sure custom vector field doesn't mess up other kinds of fields
        Add a document with a custom vector field, multimodal, and standard
        OpenSearch mapping is checked here.
        """
        # TODO - fix
        # Mixed mapping to test both multimodal and custom vector
        mixed_mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            },
            "my_multimodal": {
                "type": "multimodal_combination",
                "weights": {
                    "text": 0.4,
                    "image": 0.6
                }
            }
        }

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "text_field": "blah",
                    "my_custom_vector": {
                        "content": "custom content is here!!",
                        "vector": self.random_vector_1
                    },
                    "my_multimodal": {
                        "text": "multimodal text",
                        "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                    }
                }],
                device="cpu", mappings=mixed_mappings
            )
        )

        # Confirm backend was updated
        index_info = tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1)
        assert index_info.properties['my_custom_vector']['type'] == 'text'  # It's text because content is stored here
        assert index_info.properties['text_field']['type'] == 'text'
        assert index_info.properties['my_multimodal']['properties']['text']['type'] == 'text'
        assert index_info.properties['my_multimodal']['properties']['image']['type'] == 'text'

        assert index_info.properties['__chunks']['properties']['my_custom_vector']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties']['text_field']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties']['text_field']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties']['my_multimodal']['properties']['text'][
                   'type'] == 'keyword'
        assert index_info.properties['__chunks']['properties']['my_multimodal']['properties']['image'][
                   'type'] == 'keyword'

        assert index_info.properties['__chunks']['properties'][enums.TensorField.marqo_knn_field]['type'] == 'knn_vector'

    @unittest.skip
    def test_add_documents_use_existing_tensors_with_custom_vector_field(self):
        """
        Add a document with a custom vector field and use existing tensors:
        Will not actually use existing tensors, as custom vector pipeline
        doesn't chunk or vectorise anyway.
        """
        # TODO - fix
        # If we change the custom vector, doc should change
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "my_custom_vector": {
                        "content": "1 - custom content is here!!",
                        "vector": self.random_vector_1
                    }
                }],
                device="cpu", mappings=self.mappings
            )
        )

        get_doc_1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="0", show_vectors=True)

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "my_custom_vector": {
                        "content": "2 - custom content is here!!",
                        "vector": self.random_vector_2
                    }
                }],
                device="cpu", mappings=self.mappings,
                use_existing_tensors=True
            )
        )

        get_doc_2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="0", show_vectors=True)
        assert get_doc_1["my_custom_vector"] == "1 - custom content is here!!"
        assert get_doc_1[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_1
        assert get_doc_2["my_custom_vector"] == "2 - custom content is here!!"
        assert get_doc_2[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_2

        # If we do not, it should remain the same, no errors
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "my_custom_vector": {
                        "content": "2 - custom content is here!!",
                        "vector": self.random_vector_2
                    }
                }],
                device="cpu", mappings=self.mappings,
                use_existing_tensors=True
            )
        )

        get_doc_3 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="0", show_vectors=True)
        assert get_doc_2["my_custom_vector"] == "2 - custom content is here!!"
        assert get_doc_2[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_2

    def test_get_document_with_custom_vector_field(self):
        """
        Add a document with a custom vector field:
        Get the doc, both fetched content and embedding must be correct
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "custom content is here!!",
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Confirm get_document_by_id returns correct content
                res = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)

                # Check content is correct
                assert res["_id"] == "0"
                assert res["my_custom_vector"] == "custom content is here!!"

                # Check tensor facets and embedding are correct
                assert len(res[enums.TensorField.tensor_facets]) == 1
                assert res[enums.TensorField.tensor_facets][0]["my_custom_vector"] == "custom content is here!!"
                assert res[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_1

    def test_get_documents_with_custom_vector_field(self):
        """
        Get multiple docs with custom vectors,
        both fetched content and embedding must be correct
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "0",
                                "my_custom_vector": {
                                    "content": "custom content is here!!",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "1",
                                "my_custom_vector": {
                                    "content": "second custom vector",
                                    "vector": self.random_vector_2
                                }
                            },
                            {
                                "_id": "2",
                                "my_custom_vector": {
                                    "content": "third custom vector",
                                    "vector": self.random_vector_3
                                }
                            },
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Confirm get_document_by_id returns correct content
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name,
                    document_ids=["0", "1", "2"], show_vectors=True)

                assert len(res["results"]) == 3

                # Check content is correct
                assert res["results"][0]["_id"] == "0"
                assert res["results"][0]["my_custom_vector"] == "custom content is here!!"
                # Check tensor facets and embedding are correct
                assert len(res["results"][0][enums.TensorField.tensor_facets]) == 1
                assert res["results"][0][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "custom content is here!!"
                assert np.allclose(res["results"][0][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_1)

                # Check content is correct
                assert res["results"][1]["_id"] == "1"
                assert res["results"][1]["my_custom_vector"] == "second custom vector"
                # Check tensor facets and embedding are correct
                assert len(res["results"][1][enums.TensorField.tensor_facets]) == 1
                assert res["results"][1][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "second custom vector"
                assert np.allclose(res["results"][1][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_2)

                # Check content is correct
                assert res["results"][2]["_id"] == "2"
                assert res["results"][2]["my_custom_vector"] == "third custom vector"
                # Check tensor facets and embedding are correct
                assert len(res["results"][2][enums.TensorField.tensor_facets]) == 1
                assert res["results"][2][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "third custom vector"
                assert np.allclose(res["results"][2][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_3)

    def test_invalid_custom_vector_field_content(self):
        """
        Add a document with a custom vector field with invalid content/embedding/format
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                test_cases = [
                    # Wrong vector length
                    {"content": "custom content is here!!", "vector": [1.0, 1.0, 1.0]},
                    # Wrong content type
                    {"content": 12345, "vector": self.random_vector_1},
                    # Wrong vector type inside list (even if correct length)
                    {"content": "custom content is here!!", "vector": self.random_vector_1[:-1] + ["NOT A FLOAT"]},
                    # Field that shouldn't be there
                    {"content": "custom content is here!!", "vector": self.random_vector_1, "extra_field": "blah"},
                    # No vector
                    {"content": "custom content is here!!"},
                    # Nested dict inside custom vector content
                    {
                        "content": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector_1
                        },
                        "vector": self.random_vector_1
                    },
                ]

                for case in test_cases:
                    res = tensor_search.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=[{
                                "_id": "0",
                                "my_custom_vector": case
                            }],
                            device="cpu",
                            mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                            tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    assert res["errors"]
                    try:
                        tensor_search.get_document_by_id(config=self.config, index_name=index.name, document_id="0")
                        raise AssertionError
                    except DocumentNotFoundError:
                        pass

    def test_search_with_custom_vector_field(self):
        """
        Tensor search for the doc, with highlights
        """
        for index in self.indexes[0:1]:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom_vector_doc",
                                "my_custom_vector": {
                                    "content": "custom content is here!!",
                                    "vector": self.random_vector_1  # size is 512
                                }
                            },
                            {
                                "_id": "empty_content_custom_vector_doc",
                                "my_custom_vector": {
                                    "vector": self.random_vector_2  # size is 512
                                }
                            },
                            {
                                "_id": "normal_doc",
                                "text_field": "blah"
                            }
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "text_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name,
                    document_ids=["custom_vector_doc", "empty_content_custom_vector_doc", "normal_doc"], show_vectors=True)

                # Searching with context matching custom vector returns custom vector
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
                )

                assert res["hits"][0]["_id"] == "custom_vector_doc"
                assert res["hits"][0]["_score"] == 1.0
                assert res["hits"][0]["_highlights"][0]["my_custom_vector"] == "custom content is here!!"

                # Tensor search should work even if content is empty (highlight is empty string)
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_2, "weight": 1}], })
                )
                assert res["hits"][0]["_id"] == "empty_content_custom_vector_doc"
                assert res["hits"][0]["_score"] == 1.0
                assert res["hits"][0]["_highlights"][0]["my_custom_vector"] == ""

                # Searching with normal text returns text
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="blah",
                    search_method=enums.SearchMethod.TENSOR
                )
                assert res["hits"][0]["_id"] == "normal_doc"

    def test_lexical_search_with_custom_vector_field(self):
        """
        Lexical search for the doc
        """
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "custom_vector_doc",
                        "my_custom_vector": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector_1  # size is 512
                        }
                    },
                    {
                        "_id": "empty_content_custom_vector_doc",
                        "my_custom_vector": {
                            "vector": self.random_vector_2  # size is 512
                        }
                    },
                    {
                        "_id": "normal_doc",
                        "text_field": "blah"
                    }
                ],
                device="cpu", mappings=self.mappings
            )
        )

        # Searching matching custom vector content returns custom vector
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="custom content is here!!",
            search_method=enums.SearchMethod.LEXICAL
        )
        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "custom_vector_doc"
        # Empty content doc should not be in lexical results
        for hit in res["hits"]:
            assert hit["_id"] != "empty_content_custom_vector_doc"

        # Searching with normal text returns text
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="blah",
            search_method=enums.SearchMethod.LEXICAL,
        )
        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "normal_doc"
        # Empty content doc should not be in lexical results
        for hit in res["hits"]:
            assert hit["_id"] != "empty_content_custom_vector_doc"


    def test_search_with_custom_vector_field_score_modifiers(self):
        """
        Search for the doc, with score modifiers
        """
        # custom vector cannot be used as score modifier, as it cannot be numeric.
        # Using another field as score modifier on a custom vector:
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "doc0",
                        "my_custom_vector": {
                            "content": "vec 1",
                            "vector": self.random_vector_1  # size is 512
                        },
                        "multiply": 0.001  # Should make score tiny
                    },
                    {
                        "_id": "doc1",
                        "my_custom_vector": {
                            "content": "vec 2",
                            "vector": self.random_vector_2  # size is 512
                        },
                        "multiply": 1000  # Should make score huge
                    },
                ],
                device="cpu", mappings=self.mappings
            )
        )

        # Normal search should favor doc0
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
        )
        assert res["hits"][0]["_id"] == "doc0"

        # Search with score modifiers multiplyyshould favor doc1
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            score_modifiers=ScoreModifier(**{"multiply_score_by":
                                                 [{"field_name": "multiply",
                                                   "weight": 1}
                                                  ]
                                             })
        )
        assert res["hits"][0]["_id"] == "doc1"

    def test_search_with_custom_vector_field_boosting(self):
        """
        Search for the doc, with boosting
        """
        mappings = {
            "my_custom_vector_1": {
                "type": "custom_vector"
            },
            "my_custom_vector_2": {
                "type": "custom_vector"
            },
        }

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "doc0",
                        "my_custom_vector_1": {
                            "content": "vec 1",
                            "vector": self.random_vector_1  # size is 512
                        },
                    },
                    {
                        "_id": "doc1",
                        "my_custom_vector_2": {
                            "content": "vec 2",
                            "vector": self.random_vector_2  # size is 512
                        },
                    },
                ],
                device="cpu", mappings=mappings
            )
        )

        # Normal search should favor doc0
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
        )
        assert res["hits"][0]["_id"] == "doc0"

        # Search with boosting should favor doc1
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            boost={"my_custom_vector_2": [5, 1]}
        )
        assert res["hits"][0]["_id"] == "doc1"

    def test_search_with_custom_vector_field_filter_string(self):
        """
        Search for the doc, with filter string
        """
        new_mappings = {
            "my_custom_vector_1": {
                "type": "custom_vector"
            },
            "my_custom_vector_2": {
                "type": "custom_vector"
            },
            "my_custom_vector_3": {
                "type": "custom_vector"
            },
        }

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "custom vector doc 1",
                        "my_custom_vector_1": {
                            "content": "red blue yellow",
                            "vector": self.random_vector_1
                        }
                    },
                    {
                        "_id": "custom vector doc 2",
                        "my_custom_vector_2": {
                            "content": "red",
                            "vector": self.random_vector_1
                        }
                    },
                    {
                        "_id": "custom vector doc 3",
                        "my_custom_vector_2": {
                            "content": "blue",
                            "vector": self.random_vector_1
                        },
                        "my_custom_vector_3": {
                            "content": "chocolate",
                            "vector": self.random_vector_1
                        }
                    },
                    {
                        "_id": "custom vector doc 4",
                        "my_custom_vector_1": {
                            "content": "yellow",
                            "vector": self.random_vector_1
                        },
                        # Empty content field. Should not affect the document getting returned.
                        "my_custom_vector_2": {
                            "vector": self.random_vector_1
                        },
                        "my_custom_vector_3": {
                            "content": "chocolate",
                            "vector": self.random_vector_1
                        }
                    },
                ],
                device="cpu", mappings=new_mappings
            )
        )

        # Filter: all
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            filter="*:*", result_count=10
        )
        res_ids = set([hit["_id"] for hit in res["hits"]])
        assert res_ids == {"custom vector doc 1", "custom vector doc 2", "custom vector doc 3", "custom vector doc 4"}

        # Filter: custom vector 3 has chocolate
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            filter="my_custom_vector_3:chocolate", result_count=10
        )
        res_ids = set([hit["_id"] for hit in res["hits"]])
        assert res_ids == {"custom vector doc 3", "custom vector doc 4"}

        # Filter: AND statement
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            filter="my_custom_vector_3:chocolate AND my_custom_vector_2:blue", result_count=10
        )
        res_ids = set([hit["_id"] for hit in res["hits"]])
        assert res_ids == {"custom vector doc 3"}

        # Filter: OR statement
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            filter="my_custom_vector_1:red OR my_custom_vector_2:red", result_count=10
        )
        res_ids = set([hit["_id"] for hit in res["hits"]])
        assert res_ids == {"custom vector doc 2"}

        # Filter: parenthesis
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            filter="my_custom_vector_1:(red blue yellow)", result_count=10
        )
        res_ids = set([hit["_id"] for hit in res["hits"]])
        assert res_ids == {"custom vector doc 1"}

    def test_search_with_custom_vector_field_searchable_attributes(self):
        new_mappings = {
            "my_custom_vector_1": {
                "type": "custom_vector"
            },
            "my_custom_vector_2": {
                "type": "custom_vector"
            },
            "my_custom_vector_3": {
                "type": "custom_vector"
            },
        }

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "custom vector doc 1",
                        "my_custom_vector_1": {
                            "content": "doesn't matter",
                            "vector": self.random_vector_1
                        }
                    },
                    {
                        "_id": "custom vector doc 2",
                        "my_custom_vector_2": {
                            "content": "doesn't matter",
                            "vector": self.random_vector_2
                        }
                    },
                    {
                        "_id": "custom vector doc 3",
                        "my_custom_vector_3": {
                            "content": "doesn't matter",
                            "vector": self.random_vector_3
                        }
                    },
                ],
                device="cpu", mappings=new_mappings
            )
        )

        # All searchable attributes
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            searchable_attributes=["my_custom_vector_1", "my_custom_vector_2", "my_custom_vector_3"]
        )
        assert res["hits"][0]["_id"] == "custom vector doc 1"

        # Only 2 and 3
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            searchable_attributes=["my_custom_vector_2", "my_custom_vector_3"]
        )
        assert res["hits"][0]["_id"] == "custom vector doc 2"

        # Only 3
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            searchable_attributes=["my_custom_vector_3"]
        )
        assert res["hits"][0]["_id"] == "custom vector doc 3"

    def test_lexical_search_with_custom_vector_field_searchable_attributes(self):
        """
        Search for the doc, with searchable attributes
        """
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "custom vector doc",
                        "my_custom_vector": {
                            "content": "toxt to search",  # almost matching
                            "vector": self.random_vector_1  # size is 512
                        }
                    },
                    {
                        "_id": "barely matching doc",
                        "barely field": "random words search"
                    },
                    {
                        "_id": "exactly matching doc",
                        "exact field": "text to search"
                    }
                ],
                device="cpu", mappings=self.mappings
            )
        )

        # All searchable attributes
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="text to search",
            search_method=enums.SearchMethod.LEXICAL,
            searchable_attributes=["my_custom_vector", "barely field", "exact field"]
        )
        assert res["hits"][0]["_id"] == "exactly matching doc"

        # Only custom and barely matching
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="text to search",
            search_method=enums.SearchMethod.LEXICAL,
            searchable_attributes=["my_custom_vector", "barely field"]
        )
        assert res["hits"][0]["_id"] == "custom vector doc"

        # Only barely matching
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="text to search",
            search_method=enums.SearchMethod.LEXICAL,
            searchable_attributes=["barely field"]
        )
        assert res["hits"][0]["_id"] == "barely matching doc"

        # Only custom vector
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="text to search",
            search_method=enums.SearchMethod.LEXICAL,
            searchable_attributes=["my_custom_vector"]
        )
        assert res["hits"][0]["_id"] == "custom vector doc"