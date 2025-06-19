import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestSortByFeature(MarqoTestCase):

    unstructured_index_name = f"test_sort_by_feature_unstructured_{uuid.uuid4()}"
    structured_index_name = f"test_sort_by_feature_structured_{uuid.uuid4()}"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.create_indexes(
            [
                {
                    "indexName": cls.structured_index_name,
                    "type": "structured",
                    "model": "hf/all-MiniLM-L6-v2",
                    "allFields": [
                        {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                        {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                    ],
                    "tensorFields": ["title", "content"],
                },
                {
                    "indexName": cls.unstructured_index_name,
                    "type": "unstructured",
                    "model": "hf/all-MiniLM-L6-v2",
                }
            ]
        )

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]


    def test_sort_by_is_blocked_by_structured_index(self):
        """
        Tests that sort by feature is blocked for structured indexes.
        """

        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.structured_index_name).search(
                q="test",
                search_method="HYBRID",
                sort_by={
                    "fields": [
                        {
                            "fieldName": "title",
                            "order": "asc",
                            "missing": "last"
                        }
                    ]
                }
            )

        self.assertIn("Sort by feature is not supported for structured indexes", str(cm.exception))