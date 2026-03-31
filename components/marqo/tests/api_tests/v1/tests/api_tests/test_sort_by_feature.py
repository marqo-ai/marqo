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
            "is only supported for unstructured indexes created with Marqo version 2.22.0 or later",
            str(cm.exception)
        )

    def test_sort_by_feature_is_blocked_for_lexical_or_tensor_search(self):
        """
        Tests that sort by feature is blocked for lexical or tensor search.
        """
        for search_method in ["LEXICAL", "TENSOR"]:
            with self.subTest(f"Test sort by with search method {search_method}"):
                with self.assertRaises(MarqoWebError) as cm:
                    self.client.index(self.unstructured_index_name).search(
                        q="test",
                        search_method=search_method,
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
                f"sortBy can only be provided for",
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

    def test_sort_by_feature_still_works_with_lexical_or_tensor_score_modifiers(self):
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
            hybrid_parameters={
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.3,
                "rrfK": 10,
                "scoreModifiersTensor": {
                    "add_to_score": [{"field_name": "time_added_epoch", "weight": 0.001}]
                },
                "scoreModifiersLexical": {
                    "add_to_score": [{"field_name": "time_added_epoch", "weight": 0.001}]
                },
            },
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

    def test_sort_candidates_parameter(self):
        """
        Tests that sort candidate parameter works correctly with sort by feature.
        """
        # Add some documents to the unstructured index
        docs = [
            {"_id": "1", "title": "Cabbages", "content": "Vegetables", "price": 1.0},
            {"_id": "2", "title": "Broccoli", "content": "Vegetables", "price": 0.5},
            {"_id": "3", "title": "Cucumber", "content": "Vegetables", "price": 2.0},
            {"_id": "4", "title": "Apple", "content": "Fruits", "price": 1.0},
            {"_id": "5", "title": "Banana", "content": "Fruits", "price": 0.5},
            {"_id": "6", "title": "Cherry", "content": "Fruits", "price": 2.0},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["title", "content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            limit=3,
            sort_by={
                "fields": [
                    {
                        "fieldName": "price",
                        "order": "asc",
                        "missing": "last"
                    }
                ],
                "minSortCandidates": 3
            }
        )
        ids = [doc["_id"] for doc in response["hits"]]
        self.assertEqual(["5", "4", "6"], ids)
        self.assertIn("_sortCandidates", response)
        self.assertEqual(3, response["_sortCandidates"])

    def test_sort_depth_is_working_as_expected(self):
        """
        Tests that sort depth works correctly with sort by feature.
        """
        # Add some documents to the unstructured index
        docs = [
            {"_id": "1", "title": "Cabbages", "content": "Vegetables", "price": 1.0},
            {"_id": "2", "title": "Broccoli", "content": "Vegetables", "price": 0.5},
            {"_id": "3", "title": "Cucumber", "content": "Vegetables", "price": 2.0},
            {"_id": "4", "title": "Apple", "content": "Fruits", "price": 1.0},
            {"_id": "5", "title": "Banana", "content": "Fruits", "price": 0.5},
            {"_id": "6", "title": "Cherry", "content": "Fruits", "price": 2.0},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["title", "content"]
        )

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
                ],
                "sortDepth": 3 # Only sort the top 3 relevant documents
            }
        )

        ids = [doc["_id"] for doc in response["hits"]]
        self.assertEqual(["5", "4", "6"], ids[0:3])
        self.assertIn("_sortCandidates", response)
        self.assertEqual(6, response["_sortCandidates"])

        response_without_sort_depth = self.client.index(self.unstructured_index_name).search(
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

        ids_without_sort_depth = [doc["_id"] for doc in response_without_sort_depth["hits"]]
        self.assertEqual(["5", "2", "4", "1", "6", "3"], ids_without_sort_depth)
        self.assertIn("_sortCandidates", response_without_sort_depth)
        self.assertEqual(6, response_without_sort_depth["_sortCandidates"])

    def test_sort_candidate_must_larger_than_limit_plus_offset(self):
        res = self.client.index(self.unstructured_index_name).search(
            q="test",
            search_method="HYBRID",
            limit=1,
            offset=2,
            sort_by={
                "fields": [
                    {
                        "fieldName": "title",
                        "order": "asc",
                        "missing": "last"
                    },
                ],
                "minSortCandidates": 2
            }
        )
        # sort candidates should be returned even if it's less than limit + offset
        self.assertIn("_sortCandidates", res)

    def test_sort_by_can_not_sort_more_than_3_fields(self):
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
                        },
                        {
                            "fieldName": "content",
                            "order": "desc",
                            "missing": "first"
                        },
                        {
                            "fieldName": "price",
                            "order": "asc",
                            "missing": "last"
                        },
                        {
                            "fieldName": "another_field",
                            "order": "desc",
                            "missing": "first"
                        }
                    ]
                }
            )

        self.assertIn(
            "ensure this value has at most 3 items",
            str(cm.exception)
        )

    def test_sort_by_feature_sort_on_two_fields(self):
        """
        Tests that sort by feature can sort on two fields.
        """
        # Add some documents to the unstructured index
        docs = [
            {"_id": "1", "title": "Cabbages", "content": "Vegetables", "rating": 5.0, "price": 10.0},
            {"_id": "2", "title": "Broccoli", "content": "Vegetables", "rating": 5.0, "price": 20.0},
            {"_id": "3", "title": "Cucumber", "content": "Vegetables", "rating": 4.0, "price": 10.0},
            {"_id": "4", "title": "Apple", "content": "Fruits", "rating": 4.0, "price": 20.0},
            {"_id": "5", "title": "Banana", "content": "Fruits", "rating": 3.0, "price": 10.0},
            {"_id": "6", "title": "Cherry", "content": "Fruits", "rating": 3.0, "price": 20.0},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["title", "content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            sort_by={
                "fields": [
                    {
                        "fieldName": "rating",
                        "order": "desc",
                        "missing": "last"
                    },
                    {
                        "fieldName": "price",
                        "order": "asc",
                        "missing": "last"
                    }
                ]
            }
        )
        ids = [doc["_id"] for doc in response["hits"]]
        self.assertEqual(["1", "2", "3", "4", "5", "6"], ids)
        self.assertIn("_sortCandidates", response)
        self.assertEqual(6, response["_sortCandidates"])

        swapped_order_results = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            sort_by={
                "fields": [
                    {
                        "fieldName": "price",
                        "order": "asc",
                        "missing": "last"
                    },
                    {
                        "fieldName": "rating",
                        "order": "desc",
                        "missing": "last"
                    }
                ]
            }
        )

        swapped_ids = [doc["_id"] for doc in swapped_order_results["hits"]]
        self.assertEqual(["1", "3", "5", "2", "4", "6"], swapped_ids)

        self.assertNotEquals(ids, swapped_ids)

    def test_sort_by_feature_sort_on_three_fields(self):
        """
        Tests that sort by feature can sort on three fields.
        """
        # Add some documents to the unstructured index
        docs = [
            {"_id": "1", "title": "Item A", "content": "Category", "rating": 5.0, "price": 10.0, "discount": 2},
            {"_id": "2", "title": "Item B", "content": "Category", "rating": 5.0, "price": 10.0, "discount": 1},
            {"_id": "3", "title": "Item C", "content": "Category", "rating": 5.0, "price": 20.0, "discount": 1},
            {"_id": "4", "title": "Item D", "content": "Category", "rating": 5.0, "price": 20.0, "discount": 2},
            {"_id": "5", "title": "Item E", "content": "Category", "rating": 4.0, "price": 10.0, "discount": 1},
            {"_id": "6", "title": "Item F", "content": "Category", "rating": 4.0, "price": 20.0, "discount": 2},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["title", "content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            sort_by={
                "fields": [
                    {"fieldName": "rating", "order": "desc", "missing": "last"},
                    {"fieldName": "price", "order": "asc", "missing": "last"},
                    {"fieldName": "discount", "order": "asc", "missing": "last"},
                ]
            }
        )
        ids = [doc["_id"] for doc in response["hits"]]
        self.assertEqual(["2","1","3","4","5","6"], ids)
        self.assertIn("_sortCandidates", response)
        self.assertEqual(6, response["_sortCandidates"])

        swapped_order_results = self.client.index(self.unstructured_index_name).search(
            q="fruit",
            search_method="HYBRID",
            sort_by={
                "fields": [
                    {"fieldName": "rating", "order": "desc", "missing": "last"},
                    {"fieldName": "discount", "order": "asc", "missing": "last"},
                    {"fieldName": "price", "order": "asc", "missing": "last"},
                ]
            }
        )

        swapped_ids = [doc["_id"] for doc in swapped_order_results["hits"]]
        self.assertEqual(["2","3","1","4","5","6"], swapped_ids)

        self.assertNotEquals(ids, swapped_ids)