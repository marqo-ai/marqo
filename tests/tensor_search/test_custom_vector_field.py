from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.models.search import SearchContext
from marqo.errors import DocumentNotFoundError
import numpy as np
from marqo.tensor_search.validation import validate_dict
from marqo.s2_inference.s2_inference import vectorise
import requests
from marqo.s2_inference.clip_utils import load_image_from_path
import json
from unittest import mock
from unittest.mock import patch
from marqo.errors import MarqoWebError
import os
import pprint


class TestCustomVectorField(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            }
        }

        self.endpoint = self.authorized_url
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",     # dimension: 512
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })
        self.random_vector = [1. for _ in range(512)]
        
        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except:
            pass
        self.device_patcher.stop()

    def test_add_documents_with_custom_vector_field(self):
        """
        Add a document with a custom vector field:
        mock HTTP call
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
        mock_post = mock.MagicMock()
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1', '_id': '0', '_version': 1, 'result': 'created', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_seq_no': 0, '_primary_term': 1, 'status': 201}}]}
        
        @mock.patch("marqo._httprequests.HttpRequests.post", mock_post)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1,
                    docs=[{
                        "_id": "0",
                        "my_custom_vector": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector
                        }
                    }],
                    auto_refresh=True, device="cpu", mappings=self.mappings
                )
            )
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2      # 2nd call is the refresh

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
                assert chunk["__vector_marqo_knn_field"] == self.random_vector
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")
    
    def test_add_documents_with_custom_vector_field_no_content(self):
        """
        Add a document with a custom vector field with no content:
        Content should be autofilled with ""
        mock HTTP call
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
        mock_post = mock.MagicMock()
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1', '_id': '0', '_version': 1, 'result': 'created', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_seq_no': 0, '_primary_term': 1, 'status': 201}}]}
        
        @mock.patch("marqo._httprequests.HttpRequests.post", mock_post)
        def run():
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1,
                    docs=[{
                        "_id": "0",
                        "my_custom_vector": {
                            "vector": self.random_vector
                        }
                    }],
                    auto_refresh=True, device="cpu", mappings=self.mappings
                )
            )
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2      # 2nd call is the refresh

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
                assert chunk["__vector_marqo_knn_field"] == self.random_vector
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")
    
    def test_add_documents_with_custom_vector_field_backend_updated(self):
        """
        Add a document with a custom vector field:
        New field added to backend properties
        """
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[{
                    "_id": "0",
                    "my_custom_vector": {
                        "content": "custom content is here!!",
                        "vector": self.random_vector
                    }
                }],
                auto_refresh=True, device="cpu", mappings=self.mappings
            )
        )

        # Confirm backend was updated
        index_info = tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1)
        assert index_info.properties['my_custom_vector']['type'] == 'text'  # It's text because content is stored here
        assert index_info.properties['__chunks']['properties']['my_custom_vector']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties'][TensorField.marqo_knn_field]['type'] == 'knn_vector'

    def test_add_documents_with_different_field_types(self):
        """
        Makes sure custom vector field doesn't mess up other kinds of fields
        Add a document with a custom vector field, multimodal, and standard:
        In OpenSearch call, reformatted doc, chunks, and chunk metadata should be correct
        """
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
        mock_post.return_value = {'took': 15, 'errors': False, 'items': [{'index': {'_index': 'my-test-index-1', '_id': '0', '_version': 1, 'result': 'created', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_seq_no': 0, '_primary_term': 1, 'status': 201}}]}
        
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
                            "vector": self.random_vector
                        },
                        "my_multimodal": {
                            "text": "multimodal text",
                            "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                        }
                    }],
                    auto_refresh=True, device="cpu", mappings=mixed_mappings
                )
            )
            return True

        assert run()

        call_args = mock_post.call_args_list
        assert len(call_args) == 2      # 2nd call is the refresh

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
                assert chunk["__vector_marqo_knn_field"] == self.random_vector
            elif chunk["__field_name"] == "text_field":
                assert chunk["__field_content"] == "blah"
            elif chunk["__field_name"] == "my_multimodal":
                assert chunk["__field_content"] == json.dumps({
                    "text": "multimodal text",
                    "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                })
            else:
                raise AssertionError(f"Unexpected chunk field name: {chunk['__field_name']}")
    
    def test_add_documents_with_different_field_types_backend_updated(self):
        """
        Makes sure custom vector field doesn't mess up other kinds of fields
        Add a document with a custom vector field, multimodal, and standard
        OpenSearch mapping is checked here.
        """
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
                        "vector": self.random_vector
                    },
                    "my_multimodal": {
                        "text": "multimodal text",
                        "image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
                    }
                }],
                auto_refresh=True, device="cpu", mappings=mixed_mappings
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
        assert index_info.properties['__chunks']['properties']['my_multimodal']['properties']['text']['type'] == 'keyword'
        assert index_info.properties['__chunks']['properties']['my_multimodal']['properties']['image']['type'] == 'keyword'

        assert index_info.properties['__chunks']['properties'][TensorField.marqo_knn_field]['type'] == 'knn_vector'


    def test_add_documents_use_existing_tensors_with_custom_vector_field(self):
        """
        Add a document with a custom vector field and use existing tensors:
        """
        pass

    def test_get_documents_with_custom_vector_field(self):
        """
        Add a document with a custom vector field:
        Get the doc, both fetched content and embedding must be correct
        """
        pass

    def test_invalid_custom_vector_field_content(self):
        """
        Add a document with a custom vector field with invalid content/embedding/format
        """
        # Wrong vector length
        # Wrong content type
        # Wrong vector type inside list
        # Nested dict inside custom vector
        pass

    def test_search_with_custom_vector_field(self):
        """
        Tensor search for the doc, with highlights
        """
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "custom_vector_doc",
                        "my_custom_vector": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector    # size is 384
                        }
                    },
                    {
                        "_id": "normal_doc",
                        "text_field": "blah"
                    }
                ],
                auto_refresh=True, device="cpu", mappings=self.mappings
            )
        )

        # Searching with context matching custom vector returns custom vector
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector, "weight": 1}], })
        )
        assert res["hits"][0]["_id"] == "custom_vector_doc"
        assert res["hits"][0]["_score"] == 1.0
        assert res["hits"][0]["_highlights"]["my_custom_vector"] == "custom content is here!!"


        # Searching with normal text returns text
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="blah", 
            search_method=SearchMethod.TENSOR
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
                            "vector": self.random_vector    # size is 384
                        }
                    },
                    {
                        "_id": "normal_doc",
                        "text_field": "blah"
                    }
                ],
                auto_refresh=True, device="cpu", mappings=self.mappings
            )
        )

        # Searching matching custom vector content returns custom vector
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="custom content is here!!",
            search_method=SearchMethod.LEXICAL
        )
        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "custom_vector_doc"

        # Searching with normal text returns text
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="blah",
            search_method=SearchMethod.LEXICAL,
        )
        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "normal_doc"

    def test_bulk_search_with_custom_vector_field(self):
        """
        Bulk search for the doc
        """
        pass

    def test_search_with_custom_vector_field_score_modifiers(self):
        """
        Search for the doc, with score modifiers
        """
        pass

    def test_search_with_custom_vector_field_boosting(self):
        """
        Search for the doc, with boosting
        """
        pass

    def test_search_with_custom_vector_field_filter_string(self):
        """
        Search for the doc, with filter string
        """
        pass

    def test_search_with_custom_vector_field_searchable_attributes(self):
        """
        Search for the doc, with searchable attributes
        """
        pass




