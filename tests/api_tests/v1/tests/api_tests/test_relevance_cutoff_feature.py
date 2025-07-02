import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestRelevanceCutoffFeature(MarqoTestCase):

    unstructured_index_name = f"test_relevance_cutoff_feature_unstructured_{uuid.uuid4()}"
    structured_index_name = f"test_relevance_cutoff_feature_structured_{uuid.uuid4()}"

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
                        {"name": "score", "type": "float"},
                        {"name": "sort_value", "type": "float"},
                        {"name": "rating", "type": "float"},
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

    def test_relevance_cutoff_basic_relative_max_score(self):
        """
        Tests basic relevance cutoff with relative_max_score method.
        """
        # Add documents with varying relevance
        test_indexes = [self.structured_index_name, self.unstructured_index_name]
        for index_name in test_indexes:
            if index_name == self.structured_index_name:
                tensor_fields = None
            else:
                tensor_fields = ["content"]
            with self.subTest(index_name):
                docs = [
                    {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "score": 1.0},
                    {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms", "score": 0.9},
                    {"_id": "m1", "content": "Machine learning processes data efficiently", "score": 0.5},
                    {"_id": "l1", "content": "Engineers use machine tools for cutting", "score": 0.2},
                    {"_id": "l2", "content": "Bright morning sunlight streams through the room", "score": 0.1},
                ]
                self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )

                response = self.client.index(index_name).search(
                    q="machine learning artificial intelligence",
                    search_method="HYBRID",
                    relevance_cutoff={
                        "method": "relative_max_score",
                        "parameters": {"relativeScoreFactor": 0.6}
                    },
                    limit=10
                )

                # Should filter out low relevance documents
                self.assertIn("_relevantCandidates", response)
                self.assertLess(response["_relevantCandidates"], len(docs))
                self.assertGreater(response["_relevantCandidates"], 0)

                # High relevance docs should be present
                result_ids = [hit["_id"] for hit in response["hits"]]
                self.assertIn("h1", result_ids)
                self.assertIn("h2", result_ids)

    def test_relevance_cutoff_gap_detection_method(self):
        """
        Tests relevance cutoff with gap_detection method.
        """
        test_indexes = [self.structured_index_name, self.unstructured_index_name]
        for index_name in test_indexes:
            if index_name == self.structured_index_name:
                tensor_fields = None
            else:
                tensor_fields = ["content"]
            with self.subTest(index_name):
                docs = [
                    {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence enable systems", "score": 1.0},
                    {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms", "score": 0.9},
                    {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning", "score": 0.8},
                    {"_id": "m1", "content": "Machine learning processes data efficiently", "score": 0.3},
                    {"_id": "l1", "content": "Engineers use machine tools", "score": 0.2},
                    {"_id": "l2", "content": "Bright morning sunlight", "score": 0.1},
                ]
                self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )

                response = self.client.index(index_name).search(
                    q="machine learning artificial intelligence",
                    search_method="HYBRID",
                    relevance_cutoff={
                        "method": "gap_detection"
                    },
                    limit=10
                )

                # Should detect gap and filter appropriately
                self.assertIn("_relevantCandidates", response)
                self.assertLess(response["_relevantCandidates"], 6)
                self.assertGreater(response["_relevantCandidates"], 0)

    def test_relevance_cutoff_mean_std_dev_method(self):
        """
        Tests relevance cutoff with mean_std_dev method.
        """
        test_indexes = [self.structured_index_name, self.unstructured_index_name]
        for index_name in test_indexes:
            if index_name == self.structured_index_name:
                tensor_fields = None
            else:
                tensor_fields = ["content"]

            with self.subTest(index_name):
                docs = [
                    {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence systems", "rating": 5.0},
                    {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms", "rating": 4.8},
                    {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning", "rating": 4.5},
                    {"_id": "m1", "content": "Machine learning processes financial data", "rating": 3.0},
                    {"_id": "l1", "content": "Engineers use machine tools", "rating": 2.0},
                    {"_id": "l2", "content": "Bright morning sunlight streams", "rating": 1.0},
                ]
                self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )
        
                response = self.client.index(index_name).search(
                    q="machine learning artificial intelligence",
                    search_method="HYBRID",
                    relevance_cutoff={
                        "method": "mean_std_dev",
                        "parameters": {"stdDevFactor": 0.3}
                    },
                    limit=10
                )
        
                # Should filter based on mean + std deviation
                self.assertIn("_relevantCandidates", response)
                self.assertLess(response["_relevantCandidates"], 6)
                self.assertGreater(response["_relevantCandidates"], 0)

    def test_relevance_cutoff_with_sorting_integration(self):
        """
        Tests that relevance cutoff works correctly with sorting.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "sort_value": 8.1},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms", "sort_value": 9.2},
            {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning", "sort_value": 7.4},
            {"_id": "l1", "content": "Engineers use machine tools for cutting", "sort_value": 10.0},  # High sort, low relevance
            {"_id": "l2", "content": "Bright morning sunlight streams through", "sort_value": 9.5},    # High sort, low relevance
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        # Without relevance cutoff - low relevance docs with high sort values appear first
        response_no_cutoff = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}]
            },
            limit=5
        )
        ids_no_cutoff = [hit["_id"] for hit in response_no_cutoff["hits"]]
        # Should include low relevance docs with high sort values
        self.assertIn("l1", ids_no_cutoff[:3])  # Should be in top 3 due to high sort value

        # With relevance cutoff - should filter out low relevance docs despite high sort values
        response_with_cutoff = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.7}
            },
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}]
            },
            limit=5
        )
        
        # Should have cutoff metadata
        self.assertIn("_relevantCandidates", response_with_cutoff)
        self.assertIn("_sortCandidates", response_with_cutoff)
        
        # Should filter out low relevance docs
        ids_with_cutoff = [hit["_id"] for hit in response_with_cutoff["hits"]]
        self.assertNotIn("l1", ids_with_cutoff)  # Should be filtered out
        self.assertNotIn("l2", ids_with_cutoff)  # Should be filtered out
        
        # Should still include high relevance docs
        self.assertIn("h1", ids_with_cutoff)
        self.assertIn("h2", ids_with_cutoff)
        self.assertIn("h3", ids_with_cutoff)

    def test_structured_index_blocks_relevance_cutoff_with_sorting(self):
        """
        Tests that the structured index blocks relevance cutoff when sorting is applied.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "sort_value": 8.1},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms",
             "sort_value": 9.2},
            {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning", "sort_value": 7.4},
            {"_id": "l1", "content": "Engineers use machine tools for cutting", "sort_value": 10.0},
            # High sort, low relevance
            {"_id": "l2", "content": "Bright morning sunlight streams through", "sort_value": 9.5},
            # High sort, low relevance
        ]
        self.client.index(self.structured_index_name).add_documents(
            docs
        )

        with self.assertRaises(MarqoWebError) as cm:
            # Attempt to use relevance cutoff with sorting on structured index
            self.client.index(self.structured_index_name).search(
                q="machine learning artificial intelligence",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": 0.7}
                },
                sort_by={
                    "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}]
                },
                limit=5
            )
        self.assertIn(
            "is only supported for unstructured indexes created with Marqo version 2.22.0",
            str(cm.exception)
        )

    def test_relevance_cutoff_with_min_sort_candidates(self):
        """
        Tests interaction between relevance cutoff and minSortCandidates.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "sort_value": 8.1},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning", "sort_value": 9.2},
            {"_id": "l1", "content": "Engineers use machine tools", "sort_value": 10.0},
            {"_id": "l2", "content": "Bright morning sunlight", "sort_value": 9.5},
            {"_id": "l3", "content": "Weather patterns emerge", "sort_value": 8.8},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        # High minSortCandidates should override relevance cutoff
        response = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.9}  # Very restrictive
            },
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}],
                "minSortCandidates": 5  # Override the cutoff
            },
            limit=5
        )

        # Should have metadata
        self.assertIn("_relevantCandidates", response)
        self.assertIn("_sortCandidates", response)
        
        # minSortCandidates should override relevance filtering
        self.assertEqual(response["_sortCandidates"], 5)
        
        # Should include low relevance docs due to override
        ids = [hit["_id"] for hit in response["hits"]]
        self.assertIn("l1", ids)  # Should be included due to minSortCandidates override

    def test_relevance_cutoff_extreme_parameter_values(self):
        """
        Tests relevance cutoff with extreme parameter values.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms artificial intelligence", "score": 1.0},
            {"_id": "h2", "content": "Artificial intelligence machine learning", "score": 0.9},
            {"_id": "m1", "content": "Machine learning processes", "score": 0.5},
            {"_id": "l1", "content": "Engineers use tools", "score": 0.1},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        # Test with factor = 1.0 (most restrictive)
        response_max = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 1.0}
            },
            limit=10
        )
        self.assertLessEqual(response_max["_relevantCandidates"], 2)

        # Test with factor = 0.0 (least restrictive)
        response_min = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.0}
            },
            limit=10
        )
        self.assertGreaterEqual(response_min["_relevantCandidates"], 0)

    def test_relevance_cutoff_with_pagination(self):
        """
        Tests relevance cutoff works correctly with pagination.
        """
        docs = [
            {
                "_id": f"h{i}", "content": f"Machine learning artificial intelligence algorithms doc {i}",
                "sort_value": 10 - i
            }
            for i in range(1, 11)
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        # Test pagination with relevance cutoff
        page1 = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}]
            },
            limit=3,
            offset=0
        )

        page2 = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc", "missing": "last"}]
            },
            limit=3,
            offset=3
        )

        # Both pages should have consistent metadata
        self.assertEqual(page1["_relevantCandidates"], page2["_relevantCandidates"])
        self.assertEqual(page1["_sortCandidates"], page2["_sortCandidates"])

        # Should respect limit
        self.assertLessEqual(len(page1["hits"]), 3)
        self.assertLessEqual(len(page2["hits"]), 3)

        # Combined results should maintain sort order
        all_sort_values = []
        all_sort_values.extend([hit["sort_value"] for hit in page1["hits"]])
        all_sort_values.extend([hit["sort_value"] for hit in page2["hits"]])
        self.assertEqual(all_sort_values, sorted(all_sort_values, reverse=True))

    def test_relevance_cutoff_invalid_method(self):
        """
        Tests that invalid relevance cutoff methods are rejected.
        """
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "invalid_method",
                    "parameters": {"threshold": 0.5}
                }
            )
        self.assertIn("value is not a valid enumeration member; permitted", str(cm.exception).lower())

    def test_relevance_cutoff_missing_required_parameters(self):
        """
        Tests that missing required parameters are rejected.
        """
        # Missing relativeScoreFactor for relative_max_score
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {}
                }
            )

        self.assertIn("[{'loc': ['__root__', 'relevancecutoff', 'parameters', 'relativescorefactor']",
                      str(cm.exception).lower())

        # Missing stdDevFactor for mean_std_dev
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "mean_std_dev",
                    "parameters": {}
                }
            )

        self.assertIn("['__root__', 'relevancecutoff', 'parameters', 'stddevfactor']",
                      str(cm.exception).lower())

    def test_relevance_cutoff_invalid_parameter_values(self):
        """
        Tests that invalid parameter values are rejected.
        """
        # Negative relativeScoreFactor
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": -0.5}
                }
            )

        self.assertIn("ensure this value is greater than or equal to 0", str(cm.exception).lower())

        # relativeScoreFactor > 1.0
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.unstructured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": 1.5}
                }
            )

        self.assertIn("ensure this value is less than or equal to 1", str(cm.exception).lower())

    def test_relevance_cutoff_consistency_across_calls(self):
        """
        Tests that identical relevance cutoff calls return consistent results.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "score": 1.0},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning", "score": 0.9},
            {"_id": "m1", "content": "Machine learning processes data", "score": 0.5},
            {"_id": "l1", "content": "Engineers use tools", "score": 0.1},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        cutoff_params = {
            "method": "relative_max_score",
            "parameters": {"relativeScoreFactor": 0.6}
        }

        # Make multiple identical calls
        response1 = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff=cutoff_params,
            limit=10
        )

        response2 = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff=cutoff_params,
            limit=10
        )

        # Results should be consistent
        self.assertEqual(response1["_relevantCandidates"], response2["_relevantCandidates"])
        self.assertEqual(len(response1["hits"]), len(response2["hits"]))

        # Order should be consistent
        ids1 = [hit["_id"] for hit in response1["hits"]]
        ids2 = [hit["_id"] for hit in response2["hits"]]
        self.assertEqual(ids1, ids2)

    def test_relevance_cutoff_with_different_search_methods(self):
        """
        Tests relevance cutoff with different hybrid search configurations.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence", "score": 1.0},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning", "score": 0.9},
            {"_id": "m1", "content": "Machine learning processes data", "score": 0.5},
            {"_id": "l1", "content": "Engineers use tools", "score": 0.1},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        cutoff_config = {
            "method": "relative_max_score",
            "parameters": {"relativeScoreFactor": 0.7}
        }

        # Test with different hybrid configurations
        hybrid_configs = [
            {"retrievalMethod": "disjunction", "rankingMethod": "rrf"},
            {"retrievalMethod": "lexical", "rankingMethod": "tensor"},
            {"retrievalMethod": "tensor", "rankingMethod": "lexical"},
        ]

        for hybrid_params in hybrid_configs:
            response = self.client.index(self.unstructured_index_name).search(
                q="machine learning artificial intelligence",
                search_method="HYBRID",
                hybrid_parameters=hybrid_params,
                relevance_cutoff=cutoff_config,
                limit=10
            )

            # All should have cutoff metadata
            self.assertIn("_relevantCandidates", response)
            self.assertIn("_probeCandidates", response)
            self.assertLess(response["_relevantCandidates"], len(docs))

    def test_relevance_cutoff_preserves_document_structure(self):
        """
        Tests that relevance cutoff preserves document structure and metadata.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms", "metadata": {"category": "AI"}, "score": 1.0},
            {"_id": "h2", "content": "Artificial intelligence systems", "metadata": {"category": "AI"}, "score": 0.9},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "gap_detection"
            },
            limit=10
        )

        # Verify basic structure
        self.assertIn("hits", response)
        self.assertIn("_relevantCandidates", response)
        self.assertIn("_probeCandidates", response)

        # Verify each hit has required fields
        for hit in response["hits"]:
            self.assertIn("_id", hit)
            self.assertIn("_score", hit)
            self.assertIn("content", hit)
            self.assertIn("metadata", hit)
            self.assertIsInstance(hit["_score"], (int, float))
            self.assertGreater(hit["_score"], 0)

    def test_relevance_cutoff_blocks_irrelevant_documents(self):
        """
        Tests that relevance cutoff effectively blocks completely irrelevant documents.
        """
        docs = [
            {"_id": "relevant", "content": "Machine learning algorithms in artificial intelligence", "score": 1.0},
            {"_id": "irrelevant1", "content": "Bright morning sunlight streams through windows", "score": 0.1},
            {"_id": "irrelevant2", "content": "Weather patterns emerge across the landscape", "score": 0.1},
            {"_id": "irrelevant3", "content": "Ancient manuscripts reveal historical secrets", "score": 0.1},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        # Search with completely unrelated query
        response = self.client.index(self.unstructured_index_name).search(
            q="completely unrelated search query that matches nothing",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            limit=10
        )

        # Should have very few or no results due to low relevance
        self.assertLessEqual(len(response["hits"]), 2)
        self.assertIn("_relevantCandidates", response)
        self.assertLessEqual(response["_relevantCandidates"], 2)