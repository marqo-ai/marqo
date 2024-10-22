import uuid
from abc import abstractmethod

import pytest

from base_test_case import BaseTestCase
from marqo_test import MarqoTestCase
# from marqo.client import Client
# from marqo.errors import MarqoWebError
import marqo


@pytest.mark.marqo_version('2.13.0')
class TestVectorNormalisation(BaseTestCase):
    text_index_with_normalize_embeddings_true = "add_doc_api_test_structured_index_with_normalize_embeddings_true" + str(uuid.uuid4()).replace('-', '')

    DEFAULT_DIMENSIONS = 384
    custom_vector = [1.0 for _ in range(DEFAULT_DIMENSIONS)]
    expected_custom_vector_after_normalization = [0.05103103816509247 for _ in range(DEFAULT_DIMENSIONS)]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = marqo.Client(**cls.client_settings)
        print("Client has been initialized:", cls.client)

        cls.create_indexes([
            {
                "indexName": cls.text_index_with_normalize_embeddings_true,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "normalizeEmbeddings": True,
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "int_field_1", "type": "int"},
                    {"name": "float_field_1", "type": "float"},
                    {"name": "long_field_1", "type": "long"},
                    {"name": "double_field_1", "type": "double"},
                    {"name": "array_int_field_1", "type": "array<int>"},
                    {"name": "array_float_field_1", "type": "array<float>"},
                    {"name": "array_long_field_1", "type": "array<long>"},
                    {"name": "array_double_field_1", "type": "array<double>"},
                    {"name": "custom_vector_field_1", "type": "custom_vector",
                     "features": ["lexical_search", "filter"]},
                ],
                "tensorFields": ["title", "content", "custom_vector_field_1"],
            },
        ]
        )

        cls.indexes_to_delete = [cls.text_index_with_normalize_embeddings_true]

    def prepare(self):
        # Create structured and unstructured indexes and add some documents, set normalise embeddings to true
        # Add documents

        add_docs_res_normalized = self.client.index(index_name=self.text_index_with_normalize_embeddings_true).add_documents(
            documents=[
                {
                    "custom_vector_field_1": {
                        "content": "custom vector text",
                        "vector": self.custom_vector,
                    },
                    "content": "normal text",
                    "_id": "doc1",
                },
                {
                    "content": "second doc",
                    "_id": "doc2"
                }
            ])
        print(add_docs_res_normalized)

    def test_custom_vector_doc_in_normalized_embedding_true(self):
        # This runs on to_version
        doc_res_normalized = self.client.index(self.text_index_with_normalize_embeddings_true).get_document(
            document_id="doc1",
            expose_facets=True
        )
        get_indexes = self.client.get_indexes();
        print(get_indexes)
        # for index in get_indexes:

        assert doc_res_normalized["custom_vector_field_1"] == "custom vector text"
        assert doc_res_normalized['_tensor_facets'][0]["custom_vector_field_1"] == "custom vector text"
        assert doc_res_normalized['_tensor_facets'][0]['_embedding'] == self.expected_custom_vector_after_normalization
        pass
