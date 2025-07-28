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

    def test_structured_index_relevance_cutoff_not_supported(self):
        """
        Tests that relevance cutoff is not supported on structured indexes.
        """
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.structured_index_name).search(
                q="test",
                search_method="HYBRID",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": 0.5}
                }
            )
        self.assertIn(
            "is only supported for unstructured indexes created with Marqo version 2.13.0 or later",
            str(cm.exception)
        )

    def test_relevance_cutoff_basic_relative_max_score(self):
        """
        Tests basic relevance cutoff with relative_max_score method.
        """
        # Add documents with varying relevance
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence"},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms"},
            {"_id": "m1", "content": "Machine learning processes data efficiently"},
            {"_id": "l1", "content": "Engineers use machine tools for cutting"},
            {"_id": "l2", "content": "Bright morning sunlight streams through the room"},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.6}
            },
            limit=10
        )

        # Assert on exact metadata values
        self.assertEqual(2, response["_relevantCandidates"])
        self.assertEqual(4, response["_probeCandidates"])
        
        # Assert on exact document IDs returned
        ids = [hit["_id"] for hit in response["hits"]]
        self.assertEqual({'h1', 'h2'}, set(ids))

    def test_relevance_cutoff_feature_is_blocked_for_lexical_or_tensor_search(self):
        """
        Tests that relevance cutoff feature is blocked for lexical or tensor search.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence enable systems"},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        for search_method in ["LEXICAL", "TENSOR"]:
            with self.subTest(f"Test sort by with search method {search_method}"):
                with self.assertRaises(MarqoWebError) as cm:
                    self.client.index(self.unstructured_index_name).search(
                        q="test",
                        search_method=search_method,
                        relevance_cutoff={
                            "method": "gap_detection"
                        },
                    )

            self.assertIn(
                f"relevanceCutoff can only be provided for",
                str(cm.exception)
            )

    def test_relevance_cutoff_gap_detection_method(self):
        """
        Tests relevance cutoff with gap_detection method.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence enable systems"},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms"},
            {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning"},
            {"_id": "m1", "content": "Machine learning processes data efficiently"},
            {"_id": "l1", "content": "Engineers use machine tools"},
            {"_id": "l2", "content": "Bright morning sunlight"},
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
        ids = [hit["_id"] for hit in response["hits"]]

        self.assertEqual(5, response["_probeCandidates"])
        self.assertEqual(3, response["_relevantCandidates"])
        self.assertEqual({'h1', 'h3', 'h2'}, set(ids))

    def test_relevance_cutoff_mean_std_dev_method(self):
        """
        Tests relevance cutoff with mean_std_dev method.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence systems"},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning algorithms"},
            {"_id": "h3", "content": "Researchers develop artificial intelligence machine learning"},
            {"_id": "m1", "content": "Machine learning processes financial data"},
            {"_id": "l1", "content": "Engineers use machine tools"},
            {"_id": "l2", "content": "Bright morning sunlight streams"},
        ]
        self.client.index(self.unstructured_index_name).add_documents(
            docs, tensor_fields=["content"]
        )

        response = self.client.index(self.unstructured_index_name).search(
            q="machine learning artificial intelligence",
            search_method="HYBRID",
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.3}
            },
            limit=10
        )

        ids = [hit["_id"] for hit in response["hits"]]
        self.assertEqual(5, response["_probeCandidates"])
        self.assertEqual(3, response["_relevantCandidates"])
        self.assertEqual({'h1', 'h3', 'h2'}, set(ids))

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
        self.assertEqual(["l1", "l2", "h2", "h1", "h3"], ids_no_cutoff)

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

        self.assertEqual(4, response_with_cutoff["_probeCandidates"])
        self.assertEqual(3, response_with_cutoff["_relevantCandidates"])
        self.assertEqual(3, response_with_cutoff["_sortCandidates"])
        ids_with_cutoff = [hit["_id"] for hit in response_with_cutoff["hits"]]
        self.assertEqual(["h2", "h1", "h3"], ids_with_cutoff)

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

        self.assertEqual(3, response["_probeCandidates"])
        self.assertEqual(2, response["_relevantCandidates"])
        self.assertEqual(5, response["_sortCandidates"])
        ids = [hit["_id"] for hit in response["hits"]]
        self.assertEqual(['l1', 'l2', 'h2', 'l3', 'h1'], ids)

    def test_relevance_cutoff_extreme_parameter_values(self):
        """
        Tests relevance cutoff with extreme parameter values.
        """
        docs = [
            {"_id": "h1", "content": "Machine learning algorithms artificial intelligence"},
            {"_id": "h2", "content": "Artificial intelligence machine learning"},
            {"_id": "m1", "content": "Machine learning processes"},
            {"_id": "l1", "content": "Engineers use tools"}
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
        self.assertEqual(3, response_max["_probeCandidates"])
        self.assertEqual(1, response_max["_relevantCandidates"])
        ids_max = [hit["_id"] for hit in response_max["hits"]]
        self.assertEqual(['h2'], ids_max)

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
        self.assertEqual(3, response_min["_probeCandidates"])
        self.assertEqual(3, response_min["_relevantCandidates"])
        ids_min = [hit["_id"] for hit in response_min["hits"]]
        self.assertEqual(['h2', 'h1', 'm1'], ids_min)

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
        self.assertEqual(3, len(page1["hits"]), 3)
        self.assertEqual(3, len(page2["hits"]), 3)

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
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence"},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning"},
            {"_id": "m1", "content": "Machine learning processes data"},
            {"_id": "l1", "content": "Engineers use tools"},
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
            {"_id": "h1", "content": "Machine learning algorithms in artificial intelligence"},
            {"_id": "h2", "content": "Artificial intelligence relies on machine learning"},
            {"_id": "m1", "content": "Machine learning processes data"},
            {"_id": "l1", "content": "Engineers use tools"},
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

            # Assert exact metadata values
            self.assertEqual(2, response["_relevantCandidates"])
            self.assertEqual(3, response["_probeCandidates"])
            
            # Assert exact document IDs
            ids = [hit["_id"] for hit in response["hits"]]
            self.assertEqual({"h1", "h2"}, set(ids))