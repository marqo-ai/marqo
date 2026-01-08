import uuid
from unittest.mock import patch

from tests.marqo_test import MarqoTestCase


class TestHealth(MarqoTestCase):


    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text"},
                ],
                "tensorFields": ["title"]
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def test_check_index_health_response_format(self):
        test_cases = [
            (self.structured_index_name, "structured"),
            (self.unstructured_index_name, "unstructured")
        ]

        for index_name, msg in test_cases:
            with self.subTest(msg):
                res = self.client.index(index_name).health()
                self.assertIn("status", res)
                self.assertIn("inference", res)
                self.assertIn("backend", res)

                self.assertIn("status", res["inference"])

                self.assertIn("status", res["backend"])
                self.assertIn("memoryIsAvailable", res["backend"])
                self.assertIn("storageIsAvailable", res["backend"])

    def test_check_index_health_query(self):
        test_cases = [
            (self.structured_index_name, "structured"),
            (self.unstructured_index_name, "unstructured")
        ]
        for index_name, msg in test_cases:
            with self.subTest(msg):
                with patch("marqo._httprequests.HttpRequests.get") as mock_get:
                    res = self.client.index(index_name).health()
                    args, kwargs = mock_get.call_args
                    self.assertIn(f"/{index_name}/health", kwargs["path"])
