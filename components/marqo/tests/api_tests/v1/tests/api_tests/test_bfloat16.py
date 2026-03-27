import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase


class TestBfloat16(MarqoTestCase):
    """End-to-end tests for creating indexes with bfloat16 vector numeric type,
    indexing documents, and searching."""

    unstructured_index_name = "bf16_unstructured_" + str(uuid.uuid4()).replace('-', '')
    structured_index_name = "bf16_structured_" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "vectorNumericType": "bfloat16",
            },
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "vectorNumericType": "bfloat16",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["lexical_search"]},
                ],
                "tensorFields": ["title"],
            },
        ])

        cls.indexes_to_delete = [cls.unstructured_index_name, cls.structured_index_name]

    def test_unstructured_bfloat16_settings(self):
        """Verify unstructured bf16 index has correct vectorNumericType setting."""
        settings = self.client.index(self.unstructured_index_name).get_settings()
        self.assertEqual("bfloat16", settings["vectorNumericType"])

    def test_structured_bfloat16_settings(self):
        """Verify structured bf16 index has correct vectorNumericType setting."""
        settings = self.client.index(self.structured_index_name).get_settings()
        self.assertEqual("bfloat16", settings["vectorNumericType"])

    def test_unstructured_bfloat16_add_and_search(self):
        """Add documents to unstructured bf16 index and verify all search methods work."""
        documents = [
            {"_id": "doc1", "title": "The quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "A fast auburn canine leaps above a sleepy hound"},
            {"_id": "doc3", "title": "Python is a popular programming language"},
        ]
        res = self.client.index(self.unstructured_index_name).add_documents(
            documents, tensor_fields=["title"]
        )
        self.assertFalse(res["errors"])

        # Tensor search - semantic match should rank doc1 first
        tensor_res = self.client.index(self.unstructured_index_name).search(
            q="fox jumping", search_method="TENSOR"
        )
        self.assertEqual("doc1", tensor_res["hits"][0]["_id"])

        # Lexical search - keyword match should rank doc3 first
        lexical_res = self.client.index(self.unstructured_index_name).search(
            q="programming language", search_method="LEXICAL"
        )
        self.assertEqual("doc3", lexical_res["hits"][0]["_id"])

        # Hybrid search
        hybrid_res = self.client.index(self.unstructured_index_name).search(
            q="fox jumping", search_method="HYBRID"
        )
        self.assertEqual(3, len(hybrid_res["hits"]))

    def test_structured_bfloat16_add_and_search(self):
        """Add documents to structured bf16 index and verify all search methods work."""
        documents = [
            {"_id": "doc1", "title": "The quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "A fast auburn canine leaps above a sleepy hound"},
            {"_id": "doc3", "title": "Python is a popular programming language"},
        ]
        res = self.client.index(self.structured_index_name).add_documents(documents)
        self.assertFalse(res["errors"])

        # Tensor search - semantic match should rank doc1 first
        tensor_res = self.client.index(self.structured_index_name).search(
            q="fox jumping", search_method="TENSOR"
        )
        self.assertEqual("doc1", tensor_res["hits"][0]["_id"])

        # Lexical search - keyword match should rank doc3 first
        lexical_res = self.client.index(self.structured_index_name).search(
            q="programming language", search_method="LEXICAL"
        )
        self.assertEqual("doc3", lexical_res["hits"][0]["_id"])

        # Hybrid search
        hybrid_res = self.client.index(self.structured_index_name).search(
            q="fox jumping", search_method="HYBRID"
        )
        self.assertEqual(3, len(hybrid_res["hits"]))

    def test_bfloat16_get_document_with_vectors(self):
        """Verify that documents can be retrieved with their bf16 vectors."""
        documents = [{"_id": "vec_doc1", "title": "test document for vector retrieval"}]
        self.client.index(self.unstructured_index_name).add_documents(
            documents, tensor_fields=["title"]
        )

        doc = self.client.index(self.unstructured_index_name).get_document(
            "vec_doc1", expose_facets=True
        )
        self.assertEqual("vec_doc1", doc["_id"])
        self.assertIn("_tensor_facets", doc)
