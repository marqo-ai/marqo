import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestBfloat16(MarqoTestCase):
    """End-to-end tests for creating indexes with bfloat16 vector numeric type,
    indexing documents, and searching."""

    def setUp(self) -> None:
        super().setUp()
        self.index_name = "test_bf16_" + str(uuid.uuid4()).replace('-', '')

    def tearDown(self):
        super().tearDown()
        try:
            self.client.delete_index(index_name=self.index_name)
        except MarqoWebError:
            pass

    def test_unstructured_bfloat16_index_add_and_search(self):
        """Create an unstructured bf16 index, add documents, and verify all search methods work."""
        self.client.create_index(
            index_name=self.index_name,
            type="unstructured",
            model="hf/all-MiniLM-L6-v2",
            vector_numeric_type="bfloat16",
        )

        documents = [
            {"_id": "doc1", "title": "The quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "A fast auburn canine leaps above a sleepy hound"},
            {"_id": "doc3", "title": "Python is a popular programming language"},
        ]
        res = self.client.index(self.index_name).add_documents(documents, tensor_fields=["title"])
        self.assertFalse(res["errors"])

        # Verify index settings
        settings = self.client.index(self.index_name).get_settings()
        self.assertEqual("bfloat16", settings["vectorNumericType"])

        # Tensor search
        tensor_res = self.client.index(self.index_name).search(q="fox jumping", search_method="TENSOR")
        self.assertGreater(len(tensor_res["hits"]), 0)
        self.assertEqual("doc1", tensor_res["hits"][0]["_id"])

        # Lexical search
        lexical_res = self.client.index(self.index_name).search(q="programming language", search_method="LEXICAL")
        self.assertGreater(len(lexical_res["hits"]), 0)
        self.assertEqual("doc3", lexical_res["hits"][0]["_id"])

        # Hybrid search
        hybrid_res = self.client.index(self.index_name).search(q="fox jumping", search_method="HYBRID")
        self.assertGreater(len(hybrid_res["hits"]), 0)

    def test_structured_bfloat16_index_add_and_search(self):
        """Create a structured bf16 index, add documents, and verify all search methods work."""
        self.client.create_index(
            index_name=self.index_name,
            type="structured",
            model="hf/all-MiniLM-L6-v2",
            vector_numeric_type="bfloat16",
            all_fields=[
                {"name": "title", "type": "text", "features": ["lexical_search"]},
            ],
            tensor_fields=["title"],
        )

        documents = [
            {"_id": "doc1", "title": "The quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "A fast auburn canine leaps above a sleepy hound"},
            {"_id": "doc3", "title": "Python is a popular programming language"},
        ]
        res = self.client.index(self.index_name).add_documents(documents)
        self.assertFalse(res["errors"])

        # Verify index settings
        settings = self.client.index(self.index_name).get_settings()
        self.assertEqual("bfloat16", settings["vectorNumericType"])

        # Tensor search
        tensor_res = self.client.index(self.index_name).search(q="fox jumping", search_method="TENSOR")
        self.assertGreater(len(tensor_res["hits"]), 0)
        self.assertEqual("doc1", tensor_res["hits"][0]["_id"])

        # Lexical search
        lexical_res = self.client.index(self.index_name).search(q="programming language", search_method="LEXICAL")
        self.assertGreater(len(lexical_res["hits"]), 0)
        self.assertEqual("doc3", lexical_res["hits"][0]["_id"])

        # Hybrid search
        hybrid_res = self.client.index(self.index_name).search(q="fox jumping", search_method="HYBRID")
        self.assertGreater(len(hybrid_res["hits"]), 0)

    def test_bfloat16_get_document_with_vectors(self):
        """Verify that documents can be retrieved with their bf16 vectors."""
        self.client.create_index(
            index_name=self.index_name,
            type="unstructured",
            model="hf/all-MiniLM-L6-v2",
            vector_numeric_type="bfloat16",
        )

        documents = [{"_id": "doc1", "title": "test document for vector retrieval"}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["title"])

        doc = self.client.index(self.index_name).get_document("doc1", expose_facets=True)
        self.assertEqual("doc1", doc["_id"])
        self.assertIn("_tensor_facets", doc)
        self.assertGreater(len(doc["_tensor_facets"]), 0)
