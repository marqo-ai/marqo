import uuid
import math

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError
import requests


class TestDictScoreModifiers(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "vectorNumericType": "float",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence",
                },
                "imagePreprocessing": {"patchMethod": None},
                "allFields": [
                    {"name": "text_field", "type": "text", "features": ["lexical_search"]},
                    {"name": "int_field", "type": "int", "features": ["filter"]}
                    # test no whitespace
                ],
                "tensorFields": ["text_field"],
                "annParameters": {
                    "spaceType": "prenormalized-angular",
                    "parameters": {"efConstruction": 512, "m": 16},
                }
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def test_add_documents_headers(self):
        documents = [
            {"text_field": "hello", "int_field": 1, "_id": "1"},
            {"text_field": "world", "int_field": 2, "_id": "2"},
            {"text_field": "hello world", "int_field": 3, "_id": "3"},
            {"text_field": "hello world", "int_field": 3, "_id": 4}, # Error id
        ]
        for index_name in self.indexes_to_delete:
            with self.subTest(msg=str(index_name)):
                tensor_fields = ["text_field"] if index_name.startswith("unstructured") else None
                body = {
                    "documents": documents,
                    "tensorFields": tensor_fields
                }
                res = requests.post(f"{self._MARQO_URL}/indexes/{index_name}/documents", json=body)
                self.assertEqual(200, res.status_code)
                headers = res.headers
                self.assertEqual("3", headers["x-count-success"])
                self.assertEqual("1", headers["x-count-failure"])
                self.assertEqual("0", headers["x-count-error"])

    def test_update_documents_headers(self):
        documents = [
            {"text_field": "hello", "int_field": 1, "_id": "1"},
            {"text_field": "world", "int_field": 2, "_id": "2"},
            {"text_field": "hello world", "int_field": 3, "_id": "3"},
            {"text_field": "hello world", "int_field": 3, "_id": "4"},
        ]

        self.client.index(self.structured_index_name).add_documents(documents)

        update_documents = [
            {"int_field": 11, "_id": "1"},
            {"int_field": 22, "_id": "2"},
            {"int_field": "3", "_id": "3"}, # Error field type
            {"int_field": 3, "_id": 4}, # Error id
        ]
        index_name = self.structured_index_name
        body = {
            "documents": update_documents,
        }
        res = requests.patch(f"{self._MARQO_URL}/indexes/{index_name}/documents", json=body)
        self.assertEqual(200, res.status_code)
        headers = res.headers
        self.assertEqual("2", headers["x-count-success"])
        self.assertEqual("2", headers["x-count-failure"])
        self.assertEqual("0", headers["x-count-error"])

    def test_get_documents_headers(self):
        documents = [
            {"text_field": "hello", "int_field": 1, "_id": "1"},
            {"text_field": "world", "int_field": 2, "_id": "2"},
            {"text_field": "hello world", "int_field": 3, "_id": "3"},
            {"text_field": "hello world", "int_field": 4, "_id": "4"}
        ]
        for index_name in self.indexes_to_delete:
            with self.subTest(msg=str(index_name)):
                tensor_fields = ["text_field"] if index_name.startswith("unstructured") else None

                self.client.index(index_name).add_documents(documents, tensor_fields=tensor_fields)
                document_ids = ["1", "2", "3", "4", "5", "0"] # 0 and 5 are failures
                res = requests.get(f"{self._MARQO_URL}/indexes/{index_name}/documents", json=document_ids)
                self.assertEqual(200, res.status_code)
                headers = res.headers
                self.assertEqual("4", headers["x-count-success"])
                self.assertEqual("2", headers["x-count-failure"])
                self.assertEqual("0", headers["x-count-error"])