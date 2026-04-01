import struct
import uuid

from tests.marqo_test import MarqoTestCase


class TestBfloat16(MarqoTestCase):
    """End-to-end tests for creating an unstructured index with bfloat16 vector numeric type,
    indexing documents, and searching."""

    unstructured_index_name = "bf16_unstructured_" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.create_indexes([
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "vectorNumericType": "bfloat16",
            },
        ])

        cls.indexes_to_delete = [cls.unstructured_index_name]

    def test_bfloat16_settings(self):
        """Verify bf16 index has correct vectorNumericType setting."""
        settings = self.client.index(self.unstructured_index_name).get_settings()
        self.assertEqual("bfloat16", settings["vectorNumericType"])

    def test_bfloat16_add_and_search(self):
        """Add documents to bf16 index and verify all search methods work."""
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

        # Hybrid search - still works with bf16 vectors, should return all relevant docs
        hybrid_res = self.client.index(self.unstructured_index_name).search(
            q="fox jumping", search_method="HYBRID"
        )
        hits = [hit["_id"] for hit in hybrid_res["hits"]]
        self.assertEqual(["doc1", "doc2", "doc3"], hits)

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

        embedding = doc["_tensor_facets"][0]["_embedding"]
        for val in embedding:
            self.assertEqual(val, self._float_to_bfloat16(val),
                             f"Value {val} does not have bfloat16 precision")

    @staticmethod
    def _float_to_bfloat16(value: float) -> float:
        """Round-trip a float through bfloat16 precision by truncating the lower 16 bits."""
        float32_bytes = struct.pack('>f', value)
        bfloat16_bytes = float32_bytes[:2] + b'\x00\x00'
        return struct.unpack('>f', bfloat16_bytes)[0]
