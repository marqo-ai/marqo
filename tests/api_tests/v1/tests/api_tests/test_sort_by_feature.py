import uuid

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

    def setUp(self) -> None:
        self.clear_indexes([self.unstructured_index_name, self.structured_index_name])

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

        self.assertIn(
            "feature is only supported for unstructured indexes created with Marqo version",
            str(cm.exception)
        )

    def test_sort_by_and_global_modifiers_can_not_be_used_together(self):
        """
        Tests that sort by feature cannot be used with global modifiers.
        """
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
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
                },
                score_modifiers={
                    "multiply_score_by": [{"field_name": "itemPopularity", "weight": 2}],
                }
            )

        self.assertIn(
            "in hybrid search as they are working in the same rerank phase",
            str(cm.exception)
        )

    def test_sort_by_feature_on_unstructured_index(self):
        """
        Tests that sort by feature works on unstructured indexes.
        """
        # Add some documents to the unstructured index
        docs = [
            {"_id": "1", "title": "Apple", "content": "A fruit", "price": 1.0},
            {"_id": "2", "title": "Banana", "content": "Another fruit", "price": 0.5},
            {"_id": "3", "title": "Cherry", "content": "A small fruit", "price": 2.0},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["title", "content"]
        )

        # Perform a search with sort by feature
        response = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            sort_by={
                "fields": [
                    {
                        "fieldName": "price",
                        "order": "asc",
                        "missing": "last"
                    }
                ]
            }
        )
        ids = [doc["_id"] for doc in response["hits"]]
        self.assertEqual(["2", "1", "3"], ids)
        self.assertIn("_sortCandidates", response)
        self.assertEqual(3, response["_sortCandidates"])