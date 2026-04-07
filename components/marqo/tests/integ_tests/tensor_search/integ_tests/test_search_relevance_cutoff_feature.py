import json
import pytest

from fastapi.exceptions import RequestValidationError

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search.api import search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase, TestImageUrls


@pytest.mark.skip_for_multinode(
    "Multi-nodes will return different lexical results so we can not assert on the results.")
class TestSearchRelevanceCutoffFeature(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        unstructured_index_request = cls.unstructured_marqo_index_request(
            name="relevance_cutoff_unstructured_index",
            model=Model(name="hf/all-MiniLM-L6-v2")
        )

        structured_index_request = cls.structured_marqo_index_request(
            name="relevance_cutoff_structured_index",
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(
                    name="content",
                    type="text"
                )
            ],
            tensor_fields=["content"]
        )

        cls.create_indexes([unstructured_index_request, structured_index_request])

        cls.unstructured_index_name = unstructured_index_request.name
        cls.structured_marqo_index_name = structured_index_request.name

        # 30 documents designed for "machine learning artificial intelligence algorithms" query
        test_docs = [
            # === HIGH RELEVANCE (10 docs) - Contains ALL 5 query words ===
            {"_id": "h1",
             "content": "Machine learning algorithms in artificial intelligence enable systems to adapt by processing data efficiently.",
             "sort_value": 8.1},
            {"_id": "h2",
             "content": "Artificial intelligence relies on machine learning algorithms to build predictive models from large datasets.",
             "sort_value": 9.2},
            {"_id": "h3",
             "content": "Researchers develop artificial intelligence machine learning algorithms to improve decision-making processes.",
             "sort_value": 7.4},
            {"_id": "h4",
             "content": "Scalable artificial intelligence frameworks integrate machine learning algorithms for real-time data analysis.",
             "sort_value": 9.8},
            {"_id": "h5",
             "content": "Modern artificial intelligence and machine learning algorithms optimize operational workflows across industries.",
             "sort_value": 6.5},
            {"_id": "h6",
             "content": "Sophisticated artificial intelligence machine learning algorithms optimize data mining operations effectively.",
             "sort_value": 8.9},
            {"_id": "h7",
             "content": "Cutting-edge artificial intelligence machine learning algorithms accelerate data processing in cloud platforms.",
             "sort_value": 5.3},
            {"_id": "h8",
             "content": "Enterprise artificial intelligence solutions embed machine learning algorithms to enhance user experiences.",
             "sort_value": 9.0},
            {"_id": "h9",
             "content": "Robust artificial intelligence machine learning algorithms improve data quality assessment procedures.",
             "sort_value": 7.8},
            {"_id": "h10",
             "content": "Innovative artificial intelligence and machine learning algorithms revolutionize data analytics workflows.",
             "sort_value": 8.4},

            # === MEDIUM RELEVANCE (10 docs) - Contains EXACTLY 3 of the 5 query words ===
            # (e.g., {machine, learning, algorithms} or {artificial, intelligence, learning}, etc.)
            {"_id": "m1",
             "content": "Machine learning algorithms process financial time series for forecasting market trends.",
             "sort_value": 64},
            {"_id": "m2",
             "content": "Artificial intelligence algorithms underpin recommendation engines in e-commerce platforms.",
             "sort_value": 6.7},
            {"_id": "m3",
             "content": "Artificial intelligence learning models adapt to new user behaviors in real time.",
             "sort_value": 4.3},
            {"_id": "m4",
             "content": "Machine and artificial intelligence technologies converge to create autonomous robotic systems.",
             "sort_value": 7.1},
            {"_id": "m5",
             "content": "Machine learning artificial neural networks mimic animal brain structures.",
             "sort_value": 6.2},
            {"_id": "m6",
             "content": "Advanced machine learning algorithms accelerate computational biology research.",
             "sort_value": 5.9},
            {"_id": "m7",
             "content": "Distributed artificial intelligence systems leverage algorithms for parallel decision making.",
             "sort_value": 4.8},
            {"_id": "m8",
             "content": "Deep learning frameworks support neural architectures and optimization algorithms.",
             "sort_value": 7.5},
            {"_id": "m9",
             "content": "Evolutionary algorithms integrate with machine frameworks for adaptive problem solving.",
             "sort_value": 6.0},
            {"_id": "m10",
             "content": "Artificial learning simulations test intelligence benchmarks under controlled conditions.",
             "sort_value": 4.1},

            # === LOW RELEVANCE ===
            # 5 docs with exactly 1 query word, matching the word counts of l1–l5
            {"_id": "l1",
             "content": "Engineers use machine tools for precise cutting.",
             "sort_value": 65},

            {"_id": "l2",
             "content": "Innovators encourage collaborative learning environments to foster team growth.",
             "sort_value": 2.7},  # 9 words, contains "learning"

            {"_id": "l3",
             "content": "Manufacturers produce artificial components designed precisely for specialized industrial applications.",
             "sort_value": 1.4},  # 10 words, contains "artificial"

            {"_id": "l4",
             "content": "Local units value human intelligence during critical decision making.",
             "sort_value": 100},  # 9 words, contains "intelligence"

            {"_id": "l5",
             "content": "Researchers propose algorithms optimized specifically to accelerate image processing tasks.",
             "sort_value": 60},  # 10 words, contains "algorithms"

            # === Irrelevant ===
            # 5 docs with 0 words from the query
            {"_id": "l6",
             "content": "Bright morning sunlight streamed through the quiet study room.",
             "sort_value": 2.1},

            {"_id": "l7",
             "content": "Surprising weather patterns emerged across the town.",
             "sort_value": 70},

            {"_id": "l8",
             "content": "Vibrant wildflowers adorned the rolling hills during summer.",
             "sort_value": 1.9},

            {"_id": "l9",
             "content": "Chilly autumn breeze painted golden leaves across streets.",
             "sort_value": 24},

            {"_id": "l10",
             "content": "The ancient manuscript revealed hidden stories from forgotten civilizations.",
             "sort_value": 5.6}

            # We should see 25 probe candidates in the relevance cutoff tests 5 documents are irrelevant.
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=test_docs,
                index_name=cls.unstructured_index_name,
                tensor_fields=['content']
            )
        )

        # Test results without relevance cutoff should return top 10 documents
        regular_search_results = cls._search_helper(limit=10)
        regular_search_results_ids = set(hit["_id"] for hit in regular_search_results["hits"])
        # All high relevance IDs should be present in the results
        expected_high_relevance_ids = set([f"h{i}" for i in range(1, 11)])
        if not expected_high_relevance_ids == regular_search_results_ids:
            raise RuntimeError(
                f"Expected high relevance IDs {expected_high_relevance_ids} but got {regular_search_results_ids}."
            )

        cls.PROBE_CANDIDATES = 25  # Expected number of probe candidates for relevance cutoff tests

    def setUp(self):
        pass  # To override the parent class setup method that deletes the documents after each test.

    @classmethod
    def _search_helper(
            cls,
            index_name: Optional[str] = None,
            query: str = "machine learning artificial intelligence algorithms",
            relevance_cutoff: Optional[dict] = None,
            sort_by: Optional[dict] = None,
            limit: int = 10, offset: int = 0,
            hybrid_parameters: Optional[dict] = None,
            rerank_depth_lexical: Optional[int] = None,
    ) -> dict:
        """Helper method to perform search with consistent parameters."""

        if hybrid_parameters is None:
            hybrid_parameters = {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5,
                "rerankDepthLexical": rerank_depth_lexical
            }

        if index_name is None:
            index_name = cls.unstructured_index_name

        result = json.loads(search(
            index_name=index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": hybrid_parameters,
                "relevanceCutoff": relevance_cutoff,
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset
            }
        ).body.decode('utf-8'))

        if relevance_cutoff and query == "machine learning artificial intelligence algorithms":
            if result["_probeCandidates"] != cls.PROBE_CANDIDATES:
                raise RuntimeError(
                    f"Expected 25 probe candidates, but got {result['_probeCandidates']}."
                )
        return result

    def test_relevance_cutoff_is_blocked_by_structured_index(self):
        """Test that relevance cutoff is blocked for structured index."""
        with self.assertRaises(UnsupportedFeatureError) as context:
            self._search_helper(
                index_name=self.structured_marqo_index_name,
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": 0.5},
                },
            )
        self.assertIn("The 'relevanceCutoff' feature is only supported for unstructured indexes created",
                      str(context.exception))

    def test_relevance_cutoff_is_blocked_by_tensor_search(self):
        """Test that relevance cutoff is blocked for tensor or lexical search."""
        with self.assertRaises(RequestValidationError) as context:
            _ = search(
                index_name=self.unstructured_index_name,
                marqo_config=self.config,
                device="cpu",
                search_query_dict={
                    "q": "machine learning artificial intelligence algorithms",
                    "searchMethod": SearchMethod.TENSOR,
                    "relevanceCutoff": {
                        "method": "relative_max_score",
                        "parameters": {"relativeScoreFactor": 0.5},
                    },
                }
            )
        self.assertIn("relevanceCutoff can only be provided for", str(context.exception.errors()))

    def test_relevance_cutoff_is_blocked_by_lexical_search(self):
        """Test that relevance cutoff is blocked for lexical search."""
        with self.assertRaises(RequestValidationError) as context:
            _ = search(
                index_name=self.unstructured_index_name,
                marqo_config=self.config,
                device="cpu",
                search_query_dict={
                    "q": "machine learning artificial intelligence algorithms",
                    "searchMethod": SearchMethod.LEXICAL,
                    "relevanceCutoff": {
                        "method": "relative_max_score",
                        "parameters": {"relativeScoreFactor": 0.5},
                    },
                }
            )
        self.assertIn("relevanceCutoff can only be provided for", str(context.exception.errors()))

    def test_relevance_cutoff_relative_max_score_low_threshold(self):
        """Test that relative_max_score cutoff with low threshold."""
        result = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.01},
            },
        )
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(25, result["_relevantCandidates"])

    def test_relevance_cutoff_relative_max_score_high_threshold(self):
        """Test that relative_max_score cutoff with high threshold."""
        result = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.9},
            },
        )

        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(8, result["_relevantCandidates"])

    def test_relevance_cutoff_relative_max_score_with_changing_threshold(self):
        """Test that relative_max_score cutoff with changing threshold.
        We vary the threshold from 0.1 to 0.9 and check that the number of relevance candidates
        """
        previous_relevance_candidates = 100  # Start with a high number to ensure the first check passes
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            current_relevance_candidates = self._search_helper(
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": threshold},
                },
            )["_relevantCandidates"]

            if current_relevance_candidates > previous_relevance_candidates:
                raise RuntimeError(
                    f"Expected relevance candidates to decrease with increasing threshold, "
                    f"but got {current_relevance_candidates} <= {previous_relevance_candidates}."
                )
            previous_relevance_candidates = current_relevance_candidates

    def test_relevance_cutoff_gap_detection(self):
        """Test that gap_detection cutoff works as expected."""
        result = self._search_helper(
            relevance_cutoff={
                "method": "gap_detection",
            },
        )
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(10, result["_relevantCandidates"])

    def test_relevance_cutoff_mean_std_dev(self):
        """Test that mean_std_dev cutoff works as expected."""
        result = self._search_helper(
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.1},
            },
        )

        self.assertEqual(10, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])

    def test_relevance_cutoff_changing_mean_std_dev_threshold(self):
        """Test that mean_std_dev cutoff with changing stdDevFactor works as expected."""
        previous_relevance_candidates = 100  # Start with a high number to ensure the first check passes
        for std_dev_factor in [-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6]:
            current_relevance_candidates = self._search_helper(
                relevance_cutoff={
                    "method": "mean_std_dev",
                    "parameters": {"stdDevFactor": std_dev_factor},
                },
            )["_relevantCandidates"]

            if current_relevance_candidates > previous_relevance_candidates:
                raise RuntimeError(
                    f"Expected relevance candidates to decrease with increasing stdDevFactor, "
                    f"but got {current_relevance_candidates} <= {previous_relevance_candidates}."
                )
            previous_relevance_candidates = current_relevance_candidates

    def test_simple_sort_results(self):
        """Test that relevance cutoff works correctly with sorting.

        This is an example to show that without relevance cutoff, the results are sorted by sort_value in descending order
        which leads low relevance documents to be at the top of the results, e.g., l4, l7, m1, l1, l9. These results
        are possibly retrieved by the tensor search part of the hybrid search and further sorted by sort_value.

        In production, we wouldn't want to see these documents at the top of the results,
        so we would use relevance cutoff to filter them out.
        """
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            limit=10
        )
        ids = [hit["_id"] for hit in result["hits"]]
        expected_ids = ['l4', 'l7', 'l1', 'm1', 'l5', 'l9', 'h4', 'h2', 'h8', 'h6']
        self.assertEqual(expected_ids, ids)
        # Check that the sort candidates are correct.
        self.assertEqual(30, result["_sortCandidates"])

    def test_sort_results_with_relevance_cut_off_with_low_threshold(self):
        """A test to show that low relevance threshold does not chang the results."""
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.001},
            },
            limit=10
        )
        ids = [hit["_id"] for hit in result["hits"]]
        expected_ids = [
            "l4", "l1", "m1", "l5", "h4",
            "h2", "h8", "h6", "h10", "h1"
        ]
        self.assertEqual(expected_ids, ids)
        # Check that the sort candidates are correct.
        self.assertEqual(25, result["_sortCandidates"])

    def test_sort_results_with_relevance_cut_off_with_higher_threshold(self):
        """A test to show that relative_max_score helps to cut off low relevance documents."""
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.85},
            },
            limit=10
        )
        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(10, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(['m1', 'h4', 'h2', 'h8', 'h6', 'h10', 'h1', 'h9', 'h3', 'h5'], ids)

    def test_sort_results_with_relevance_cut_off_with_higher_threshold_but_min_sort_candidates_can_be_set(self):
        """A test to show that minSortCandidates can be set to a higher value to make relevance cutoff useless.

        This is an example to show that if we set minSortCandidates to a high value, the relevance cutoff will not
        filter out any documents, even if the relativeScoreFactor is high. However, users can detect this by
        observing the `_relevantCandidates` and `_sortCandidates` metadata in the response.
        """
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}],
                "minSortCandidates": 25  # Set a high minSortCandidates
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.85},
            },
            limit=10
        )
        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(10, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(['l4', 'l1', 'm1', 'l5', 'h4', 'h2', 'h8', 'h6', 'h10', 'h1'], ids)

    def test_sort_asc_with_relevance_cutoff(self):
        """Test ascending sort order with relevance cutoff to ensure filtering works both ways."""
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "asc"}]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.8},
            },
            limit=10
        )

        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(10, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(['h7', 'h5', 'h3', 'h9', 'h1', 'h10', 'h6', 'h8', 'h2', 'h4'], ids)

    def test_sort_with_varying_min_sort_candidates(self):
        """Test how different minSortCandidates values interact with relevance cutoff."""
        base_cutoff = {
            "method": "relative_max_score",
            "parameters": {"relativeScoreFactor": 0.7}
        }

        # Test with different minSortCandidates values
        for min_sort in [5, 15, 25]:
            result = self._search_helper(
                sort_by={
                    "fields": [{"field_name": "sort_value", "order": "desc"}],
                    "minSortCandidates": min_sort
                },
                relevance_cutoff=base_cutoff,
                limit=10
            )

            self.assertGreaterEqual(
                result["_sortCandidates"], min_sort,
                f"Sort candidates should at least greater than or equal "
                f"to minSortCandidates: {min_sort}"
            )

            # When minSortCandidates is high, it should override relevance filtering
            if min_sort >= 25:
                # Should include low-relevance docs due to high minSortCandidates
                ids = [hit["_id"] for hit in result["hits"]]
                self.assertIn("l4", ids, "High minSortCandidates should include low-relevance docs")

    def test_sort_with_relevance_cutoff_different_thresholds_effectiveness(self):
        """Test how different relevance cutoff thresholds affect sort results."""
        sort_params = {"fields": [{"field_name": "sort_value", "order": "desc"}]}

        results = {}
        thresholds = [0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            result = self._search_helper(
                sort_by=sort_params,
                relevance_cutoff={
                    "method": "relative_max_score",
                    "parameters": {"relativeScoreFactor": threshold}
                },
                limit=10
            )
            results[threshold] = result

        # Higher thresholds should result in fewer relevance candidates
        prev_candidates = 30
        for threshold in thresholds:
            current_candidates = results[threshold]["_relevantCandidates"]
            self.assertLessEqual(current_candidates, prev_candidates,
                                 f"Threshold {threshold} should have <= candidates than previous")
            prev_candidates = current_candidates

        # Most restrictive threshold should exclude low-relevance docs
        restrictive_ids = [hit["_id"] for hit in results[0.9]["hits"]]
        self.assertTrue(
            {"l1", "l4", "l5"}.isdisjoint(restrictive_ids),
            "Most restrictive threshold should exclude low-relevance docs"
        )

    def test_sort_with_relevance_cutoff_preserves_high_relevance_docs_with_rerankDepthLexical(self):
        """Test that relevance cutoff with sort preserves high-relevance docs regardless of sort values when using
        weakAnd in the lexical search, with varying rerankDepthLexical values.

        Note that the tagetHits for rerankDepthLexical is not a strict constraint on the matched or returned documents
        from Vespa. So no matter what value we set for rerankDepthLexical, the results remain the same in this test.
        """
        test_cases = [
            (1, "small rerankDepthLexical 1",),
            (10, "medium rerankDepthLexical 10"),
            (100, "large rerankDepthLexical 100"),
        ]

        for rerank_depth_lexical, description in test_cases:
            with self.subTest(description):
                result = self._search_helper(
                    sort_by={
                        "fields": [{"field_name": "sort_value", "order": "asc"}]  # Ascending favors low sort values
                    },
                    relevance_cutoff={
                        "method": "relative_max_score",
                        "parameters": {"relativeScoreFactor": 0.6}
                    },
                    limit=15,
                    rerank_depth_lexical=rerank_depth_lexical,
                )

                ids = [hit["_id"] for hit in result["hits"]]

                self.assertEqual(16, result["_relevantCandidates"])
                self.assertEqual(25, result["_probeCandidates"])
                self.assertEqual(
                    ['m10', 'm7', 'h7', 'm6', 'm5', 'h5', 'm2', 'm4', 'h3', 'h9', 'h1', 'h10', 'h6', 'h8', 'h2'],
                    ids
                )

    def test_sort_with_relevance_cutoff_preserves_high_relevance_docs(self):
        """Test that relevance cutoff with sort preserves high-relevance docs regardless of sort values."""
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "asc"}]  # Ascending favors low sort values
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.6}
            },
            limit=15
        )

        ids = [hit["_id"] for hit in result["hits"]]

        self.assertEqual(16, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual(['m10', 'm7', 'h7', 'm6', 'm5', 'h5', 'm2', 'm4', 'h3', 'h9', 'h1', 'h10', 'h6', 'h8', 'h2'],
                         ids)

    def test_sort_candidates_vs_relevance_candidates_relationship(self):
        """Test the relationship between _sortCandidates and _relevantCandidates."""
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.6}
            },
            limit=10
        )

        # Sort candidates should be >= relevance candidates because sorting candidates includes the results from
        # tensor search.
        self.assertGreaterEqual(result["_sortCandidates"], result["_relevantCandidates"],
                                "Sort candidates should not exceed relevance candidates")

        # Both should be <= total available documents (25 probe candidates)
        self.assertLessEqual(result["_relevantCandidates"], 25)
        self.assertLessEqual(result["_sortCandidates"], 25)

    def test_sort_with_relevance_cutoff_edge_case_no_sort_field(self):
        """Test relevance cutoff when some documents don't have the sort field."""
        # This test ensures robustness when sort field is missing
        result = self._search_helper(
            sort_by={
                "fields": [{"field_name": "nonexistent_field", "order": "desc", "missing": "last"}]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            limit=10
        )

        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(18, result["_relevantCandidates"])
        self.assertEqual(25, result["_probeCandidates"])
        self.assertEqual({'h9', 'h3', 'h6', 'h1', 'h10', 'h4', 'h2', 'h8', 'h7', 'h5'}, set(ids))

    def test_relevance_cutoff_with_pagination(self):
        """Test relevance cutoff with different limit and offset values."""
        # Test with small limit
        result_small = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.4},
            },
            limit=3
        )
        self.assertEqual(len(result_small["hits"]), 3, "Should respect limit")
        self.assertIn("_relevantCandidates", result_small)

        # Test with offset
        result_offset = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.4},
            },
            limit=5,
            offset=2
        )
        self.assertEqual(len(result_offset["hits"]), 5, "Should respect limit with offset")

    def test_relevance_cutoff_edge_case_extreme_values(self):
        """Test edge cases with extreme parameter values."""
        # Test with factor = 1.0 (most restrictive)
        result_max = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 1.0},
            }
        )
        self.assertEqual(3, result_max["_relevantCandidates"], "Factor 1.0 should be very restrictive")

        # Test with factor = 0.0 (least restrictive)
        result_min = self._search_helper(
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.0},
            }
        )
        self.assertEqual(25, result_min["_relevantCandidates"], "Factor 0.0 should not crash")

    def test_relevance_cutoff_preserves_document_structure(self):
        """Test that relevance cutoff preserves document structure and metadata."""
        result = self._search_helper(
            relevance_cutoff={
                "method": "gap_detection",
            }
        )

        # Verify basic structure
        self.assertIn("hits", result)
        self.assertIn("_relevantCandidates", result)
        self.assertIn("_probeCandidates", result)

        # Verify each hit has required fields
        for hit in result["hits"]:
            self.assertIn("_id", hit, "Each hit should have _id")
            self.assertIn("_score", hit, "Each hit should have _score")
            self.assertIn("content", hit, "Each hit should have content")
            self.assertIsInstance(hit["_score"], (int, float), "Score should be numeric")
            self.assertGreater(hit["_score"], 0, "Score should be positive")

    def test_relevance_cutoff_baseline_without_cutoff(self):
        """Test baseline search without relevance cutoff returns expected results."""
        result = self._search_helper()

        # Should not have cutoff metadata
        self.assertNotIn("_relevantCandidates", result, "Should not have cutoff metadata")
        self.assertNotIn("_probeCandidates", result, "Should not have probe metadata")

        # Should return all high relevance documents in top 10
        result_ids = set(hit["_id"] for hit in result["hits"])
        high_relevance_ids = set([f"h{i}" for i in range(1, 11)])
        self.assertEqual(result_ids, high_relevance_ids,
                         "Should return all high relevance documents without cutoff")

    def test_relevance_cutoff_consistency_across_calls(self):
        """Test that identical relevance cutoff calls return consistent results."""
        cutoff_params = {
            "method": "relative_max_score",
            "parameters": {"relativeScoreFactor": 0.6},
        }

        result1 = self._search_helper(relevance_cutoff=cutoff_params)
        result2 = self._search_helper(relevance_cutoff=cutoff_params)

        # Results should be consistent
        self.assertEqual(result1["_relevantCandidates"], result2["_relevantCandidates"],
                         "Relevance candidates should be consistent across calls")
        self.assertEqual(len(result1["hits"]), len(result2["hits"]),
                         "Number of hits should be consistent across calls")

        # Order should be consistent
        ids1 = [hit["_id"] for hit in result1["hits"]]
        ids2 = [hit["_id"] for hit in result2["hits"]]
        self.assertEqual(ids1, ids2, "Result order should be consistent across calls")

    def test_mean_std_dev_comprehensive_std_dev_factor_range(self):
        """Test mean_std_dev method with comprehensive range of stdDevFactor values."""
        # Test negative, zero, and positive factors
        factors = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

        previous_candidates = 30
        for factor in factors:
            result = self._search_helper(
                relevance_cutoff={
                    "method": "mean_std_dev",
                    "parameters": {"stdDevFactor": factor}
                }
            )

            current_candidates = result["_relevantCandidates"]

            # Higher factors should generally result in fewer or equal candidates
            self.assertLessEqual(current_candidates, previous_candidates,
                                 f"Factor {factor} should have <= candidates than previous factor")

            # All results should be valid
            self.assertGreaterEqual(current_candidates, 0, "Should have non-negative candidates")
            self.assertLessEqual(current_candidates, 25, "Should not exceed probe candidates")

            previous_candidates = current_candidates

    def test_mean_std_dev_with_different_sort_orders(self):
        """Test mean_std_dev behavior with ascending vs descending sort."""

        base_cutoff = {
            "method": "mean_std_dev",
            "parameters": {"stdDevFactor": 1.0}
        }

        # Test with descending sort
        desc_result = self._search_helper(
            sort_by={"fields": [{"field_name": "sort_value", "order": "desc"}]},
            relevance_cutoff=base_cutoff,
            limit=8
        )

        # Test with ascending sort
        asc_result = self._search_helper(
            sort_by={"fields": [{"field_name": "sort_value", "order": "asc"}]},
            relevance_cutoff=base_cutoff,
            limit=8
        )

        # Both should apply same relevance filtering
        self.assertEqual(desc_result["_relevantCandidates"], asc_result["_relevantCandidates"],
                         "Sort order should not affect relevance filtering")

        # Should maintain appropriate sort orders
        desc_values = [hit["sort_value"] for hit in desc_result["hits"]]
        asc_values = [hit["sort_value"] for hit in asc_result["hits"]]

        self.assertEqual(desc_values, sorted(desc_values, reverse=True), "Desc should be descending")
        self.assertEqual(asc_values, sorted(asc_values), "Asc should be ascending")

    def test_mean_std_dev_with_min_sort_candidates_interaction(self):
        """Test how mean_std_dev interacts with minSortCandidates in production."""

        # Test different combinations of filtering vs minSortCandidates
        test_cases = [
            {"stdDevFactor": 0.5, "minSortCandidates": 10},
            {"stdDevFactor": 1.0, "minSortCandidates": 15},
            {"stdDevFactor": 1.5, "minSortCandidates": 20},
            {"stdDevFactor": 2.0, "minSortCandidates": 25}  # Override scenario
        ]

        for case in test_cases:
            result = self._search_helper(
                sort_by={
                    "fields": [{"field_name": "sort_value", "order": "desc"}],
                    "minSortCandidates": case["minSortCandidates"]
                },
                relevance_cutoff={
                    "method": "mean_std_dev",
                    "parameters": {"stdDevFactor": case["stdDevFactor"]}
                },
                limit=10
            )

            # Sort candidates should meet minimum requirement
            self.assertGreaterEqual(result["_sortCandidates"], case["minSortCandidates"],
                                    f"Case {case}: Sort candidates should meet minimum")

            # When minSortCandidates is very high, it overrides filtering
            if case["minSortCandidates"] >= 25:
                ids = [hit["_id"] for hit in result["hits"]]
                # Should include low-relevance docs due to override
                self.assertIn("l4", ids, f"Case {case}: High minSortCandidates should override filtering")

    def test_mean_std_dev_consistency_across_multiple_calls(self):
        """Test that mean_std_dev produces consistent results across calls."""

        cutoff_params = {
            "method": "mean_std_dev",
            "parameters": {"stdDevFactor": 1.0}
        }

        # Multiple calls with same parameters
        results = []
        for _ in range(3):
            result = self._search_helper(
                relevance_cutoff=cutoff_params,
                sort_by={"fields": [{"field_name": "sort_value", "order": "desc"}]},
                limit=8
            )
            results.append(result)

        # All calls should produce identical results
        for i in range(1, len(results)):
            self.assertEqual(results[0]["_relevantCandidates"], results[i]["_relevantCandidates"],
                             f"Call {i} should have same relevance candidates as call 0")

            ids_0 = [hit["_id"] for hit in results[0]["hits"]]
            ids_i = [hit["_id"] for hit in results[i]["hits"]]
            self.assertEqual(ids_0, ids_i, f"Call {i} should have same result order as call 0")

    def test_sort_and_relevance_cutoff_pagination(self):
        """Test that sorting and relevance cutoff work correctly with pagination.

        This test is carefully crafted with _relevantCandidates=10, and _sortCandidates=11. The _sortCandidates is
        higher because the _sortCandidates is a fusion of both lexical and tensor search results using retrieval
        size 10(_relevantCandidates).

        In this case, the first two pages (with limit=4) should return 4 results each, and the third page only returns
        3 results because there are only 11 sort candidates in total. The fourth page should return no results.
        """
        # Test with limit and offset
        page_1_results = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.5}
            },
            limit=4,
            offset=0
        )

        page_1_sort_candidates = page_1_results["_sortCandidates"]
        self.assertEqual(11, page_1_sort_candidates)

        page_2_results = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.5}
            },
            limit=4,
            offset=4
        )

        page_2_sort_candidates = page_2_results["_sortCandidates"]
        self.assertEqual(11, page_2_sort_candidates)

        page_3_results = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.5}
            },
            limit=4,
            offset=8
        )

        self.assertEqual(11, page_3_results["_sortCandidates"])
        self.assertEqual(10, page_3_results["_relevantCandidates"])
        self.assertEqual(3, len(page_3_results["hits"]))

        # We should see a consistent sort value order in both pages
        page_1_sort_values = [hit["sort_value"] for hit in page_1_results["hits"]]
        page_2_sort_values = [hit["sort_value"] for hit in page_2_results["hits"]]
        page_3_sort_values = [hit["sort_value"] for hit in page_3_results["hits"]]

        self.assertEqual(page_1_sort_values, sorted(page_1_sort_values, reverse=True),
                         "Page 1 results should be sorted descending by sort_value")
        self.assertEqual(page_2_sort_values, sorted(page_2_sort_values, reverse=True))
        self.assertEqual(page_3_sort_values, sorted(page_3_sort_values, reverse=True))
        self.assertEqual(
            page_1_sort_values + page_2_sort_values + page_3_sort_values,
            sorted(page_1_sort_values + page_2_sort_values + page_3_sort_values, reverse=True),
            "Combined pages should maintain overall descending sort order"
        )

        page_4_results = self._search_helper(
            sort_by={
                "fields": [{"field_name": "sort_value", "order": "desc"}]
            },
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.5}
            },
            limit=4,
            offset=12
        )

        self.assertEqual(0, len(page_4_results["hits"]))

    def test_relevance_cutoff_feature_works_for_lexical_tensor_search(self):
        """A test to ensure relevance cutoff works for lexical tensor search."""
        hybrid_search_parameters = {
            "retrievalMethod": "lexical",
            "rankingMethod": "tensor"
        }
        regular_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            limit=10
        )
        self.assertEqual(10, len(regular_result["hits"]))
        relevance_cutoff_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.98},
            },
            limit=10
        )
        self.assertEqual(3, len(relevance_cutoff_result["hits"]))
        self.assertEqual(3, relevance_cutoff_result["_relevantCandidates"])
        relevance_cutoff_result_ids = [hit["_id"] for hit in relevance_cutoff_result["hits"]]
        regular_result_ids = [hit["_id"] for hit in regular_result["hits"]]
        self.assertEqual(
            set(relevance_cutoff_result_ids),
            {"h9", "h10", "h6"},
            "Relevance cutoff should return the most relevant documents."
        )

    def test_relevance_cutoff_feature_works_for_lexical_lexical_search(self):
        """A test to ensure relevance cutoff works for lexical tensor search."""
        hybrid_search_parameters = {
            "retrievalMethod": "lexical",
            "rankingMethod": "lexical"
        }
        regular_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            limit=10
        )
        self.assertEqual(10, len(regular_result["hits"]))
        relevance_cutoff_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.98},
            },
            limit=10
        )
        self.assertEqual(3, len(relevance_cutoff_result["hits"]))
        self.assertEqual(3, relevance_cutoff_result["_relevantCandidates"])
        relevance_cutoff_result_ids = [hit["_id"] for hit in relevance_cutoff_result["hits"]]
        regular_result_ids = [hit["_id"] for hit in regular_result["hits"]]
        self.assertEqual(
            set(relevance_cutoff_result_ids),
            set(regular_result_ids[:3]),
            "Relevance cutoff should return the top 3 most relevant documents."
        )

    def test_relevance_cutoff_feature_works_for_tensor_lexical_search(self):
        """A test to ensure relevance cutoff works for lexical tensor search."""
        hybrid_search_parameters = {
            "retrievalMethod": "tensor",
            "rankingMethod": "lexical",
        }
        regular_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            limit=10
        )
        self.assertEqual(10, len(regular_result["hits"]))
        relevance_cutoff_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.98},
            },
            limit=10
        )
        self.assertEqual(3, len(relevance_cutoff_result["hits"]))
        self.assertEqual(3, relevance_cutoff_result["_relevantCandidates"])
        relevance_cutoff_result_ids = [hit["_id"] for hit in relevance_cutoff_result["hits"]]
        regular_result_ids = [hit["_id"] for hit in regular_result["hits"]]
        # We can't guarantee the tensor retrieval will return the same documents after the relevance cutoff as
        # the targetHit is changed and the results from ANN search are not deterministic.
        self.assertTrue(
            set(relevance_cutoff_result_ids).issubset(set(regular_result_ids)),
            "Relevance cutoff should still return relevant documents"
        )

    def test_relevance_cutoff_feature_works_for_tensor_tensor_search(self):
        """A test to ensure relevance cutoff works for tensor tensor search."""
        hybrid_search_parameters = {
            "retrievalMethod": "tensor",
            "rankingMethod": "tensor",
        }
        regular_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            limit=10
        )
        self.assertEqual(10, len(regular_result["hits"]))
        relevance_cutoff_result = self._search_helper(
            hybrid_parameters=hybrid_search_parameters,
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.98},
            },
            limit=10
        )
        self.assertEqual(3, len(relevance_cutoff_result["hits"]))
        self.assertEqual(3, relevance_cutoff_result["_relevantCandidates"])
        relevance_cutoff_result_ids = [hit["_id"] for hit in relevance_cutoff_result["hits"]]
        regular_result_ids = [hit["_id"] for hit in regular_result["hits"]]
        # We can't guarantee the tensor retrieval will return the same documents after the relevance cutoff as
        # the targetHit is changed and the results from ANN search are not deterministic.
        self.assertTrue(
            set(relevance_cutoff_result_ids).issubset(set(regular_result_ids)),
            "Relevance cutoff should still return relevant documents"
        )


class TestRelevanceCutoffAndSortByWithMoreComplicatedDocumentsAndQueries(MarqoTestCase):
    """
    This is a test class that tests the interaction between relevance cutoff and sort in a real-world like index.
    We will have a more complicated index with documents that have multiple fields and a more complex query.
    Things that are included here:
        - Documents with multiple fields
        - Documents with multimodal fields
        - Complex queries that include searchableAttributes,
        - Complex queries with score modifiers
        - Complex queries with filters
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        index_request = cls.unstructured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )
        cls.create_indexes([index_request])
        cls.index_name = index_request.name

        image_url = TestImageUrls.IMAGE1.value

        # 30 fashion documents with various fields, tags, and multimodal content
        # We categorize them into 3 categories: Glasses, Hats, and Shoes.
        documents = [
            {
                "_id": "0",
                "filter_field_1": ["us", "ca"],
                "tags": ["category:Glasses", "type:Aviator", "color:Black"],
                "price": 19.99,
                "title": "Midnight Aviator Sunglasses",
                "aux_value": 0.45,
                "image_url": image_url
            },
            {
                "_id": "1",
                "filter_field_1": ["eu", "us"],
                "tags": ["category:Glasses", "type:Round", "color:Gold"],
                "price": 22.50,
                "title": "Golden Round Spectacles",
                "aux_value": 1.12,
                "image_url": image_url
            },
            {
                "_id": "2",
                "filter_field_1": ["us", "gb", "au"],
                "tags": ["category:Glasses", "type:Wayfarer", "color:Tortoise"],
                "price": 24.00,
                "title": "Classic Tortoise Wayfarers",
                "aux_value": 0.98,
                "image_url": image_url
            },
            {
                "_id": "3",
                "filter_field_1": ["jp", "us"],
                "tags": ["category:Glasses", "type:Rectangle", "color:Silver"],
                "price": 18.75,
                "title": "Silver Frame Rectangles",
                "aux_value": 2.34,
                "image_url": image_url
            },
            {
                "_id": "4",
                "filter_field_1": ["us", "ca", "mx"],
                "tags": ["category:Glasses", "type:Cat Eye", "color:Pink"],
                "price": 20.00,
                "title": "Blush Cat Eye Glasses",
                "aux_value": 3.10,
                "image_url": image_url
            },
            {
                "_id": "5",
                "filter_field_1": ["eu", "us", "gb"],
                "tags": ["category:Glasses", "type:Sport", "color:Blue"],
                "price": 21.99,
                "title": "Azure Sport Frames",
                "aux_value": 0.67,
                "image_url": image_url
            },
            {
                "_id": "6",
                "filter_field_1": ["us", "au"],
                "tags": ["category:Glasses", "type:Shield", "color:Black"],
                "price": 25.50,
                "title": "Stealth Shield Visors",
                "aux_value": 1.75,
                "image_url": image_url
            },
            {
                "_id": "7",
                "filter_field_1": ["ca", "us"],
                "tags": ["category:Glasses", "type:Clip-On", "color:Green"],
                "price": 16.49,
                "title": "Emerald Clip-On Shades",
                "aux_value": 2.88,
                "image_url": image_url
            },
            {
                "_id": "8",
                "filter_field_1": ["us", "gb", "eu"],
                "tags": ["category:Glasses", "type:Mirrored", "color:Gold"],
                "price": 23.99,
                "title": "Mirrored Gold Lenses",
                "aux_value": 0.53,
                "image_url": image_url
            },
            {
                "_id": "9",
                "filter_field_1": ["us", "ca"],
                "tags": ["category:Glasses", "type:Polarized", "color:Brown"],
                "price": 27.00,
                "title": "Polarized Brown Sunnies",
                "aux_value": 1.40,
                "image_url": image_url
            },
            {
                "_id": "10",
                "filter_field_1": ["us", "gb"],
                "tags": ["category:Hat", "type:Fedora", "color:Beige"],
                "price": 34.99,
                "title": "Classic Beige Fedora",
                "aux_value": 0.29,
                "image_url": image_url
            },
            {
                "_id": "11",
                "filter_field_1": ["au", "us"],
                "tags": ["category:Hat", "type:Baseball", "color:Red"],
                "price": 18.00,
                "title": "Scarlet Baseball Cap",
                "aux_value": 2.05,
                "image_url": image_url
            },
            {
                "_id": "12",
                "filter_field_1": ["eu", "us", "ca"],
                "tags": ["category:Hat", "type:Beanie", "color:Gray"],
                "price": 15.75,
                "title": "Heather Gray Beanie",
                "aux_value": 1.67,
                "image_url": image_url
            },
            {
                "_id": "13",
                "filter_field_1": ["us", "mx"],
                "tags": ["category:Hat", "type:Panama", "color:Natural"],
                "price": 45.00,
                "title": "Tropical Panama Hat",
                "aux_value": 0.82,
                "image_url": image_url
            },
            {
                "_id": "14",
                "filter_field_1": ["us", "gb"],
                "tags": ["category:Hat", "type:Bucket", "color:Olive"],
                "price": 20.49,
                "title": "Olive Bucket Hat",
                "aux_value": 2.13,
                "image_url": image_url
            },
            {
                "_id": "15",
                "filter_field_1": ["ca", "us", "au"],
                "tags": ["category:Hat", "type:Snapback", "color:Black"],
                "price": 22.00,
                "title": "Midnight Snapback",
                "aux_value": 1.99,
                "image_url": image_url
            },
            {
                "_id": "16",
                "filter_field_1": ["us", "eu"],
                "tags": ["category:Hat", "type:Trucker", "color:Navy"],
                "price": 19.25,
                "title": "Navy Trucker Hat",
                "aux_value": 0.76,
                "image_url": image_url
            },
            {
                "_id": "17",
                "filter_field_1": ["us", "ca"],
                "tags": ["category:Hat", "type:Sun", "color:Yellow"],
                "price": 28.00,
                "title": "Sunny Wide-Brim Hat",
                "aux_value": 2.44,
                "image_url": image_url
            },
            {
                "_id": "18",
                "filter_field_1": ["gb", "us"],
                "tags": ["category:Hat", "type:Visor", "color:White"],
                "price": 17.50,
                "title": "White Sport Visor",
                "aux_value": 1.11,
                "image_url": image_url
            },
            {
                "_id": "19",
                "filter_field_1": ["us", "au"],
                "tags": ["category:Hat", "type:Cloche", "color:Black"],
                "price": 32.99,
                "title": "Elegant Black Cloche",
                "aux_value": 0.58,
                "image_url": image_url
            },
            {
                "_id": "20",
                "filter_field_1": ["us", "ca", "mx"],
                "tags": ["category:Hat", "type:Bowler", "color:Charcoal"],
                "price": 38.00,
                "title": "Charcoal Bowler Hat",
                "aux_value": 1.90,
                "image_url": image_url
            },
            {
                "_id": "21",
                "filter_field_1": ["us", "ca"],
                "tags": ["category:Shoes", "type:Sneakers", "color:White"],
                "price": 49.99,
                "title": "Urban Runner Sneakers",
                "aux_value": 0.34,
                "image_url": image_url
            },
            {
                "_id": "22",
                "filter_field_1": ["eu", "us"],
                "tags": ["category:Shoes", "type:Loafers", "color:Brown"],
                "price": 65.00,
                "title": "Mahogany Leather Loafers",
                "aux_value": 2.22,
                "image_url": image_url
            },
            {
                "_id": "23",
                "filter_field_1": ["us", "ca", "au"],
                "tags": ["category:Shoes", "type:Boots", "color:Tan"],
                "price": 79.50,
                "title": "Desert Tan Boots",
                "aux_value": 3.01,
                "image_url": image_url
            },
            {
                "_id": "24",
                "filter_field_1": ["us", "gb"],
                "tags": ["category:Shoes", "type:Sandals", "color:Black"],
                "price": 29.25,
                "title": "Black Slide Sandals",
                "aux_value": 1.47,
                "image_url": image_url
            },
            {
                "_id": "25",
                "filter_field_1": ["us", "eu"],
                "tags": ["category:Shoes", "type:Heels", "color:Red"],
                "price": 54.99,
                "title": "Crimson Stiletto Heels",
                "aux_value": 0.89,
                "image_url": image_url
            },
            {
                "_id": "26",
                "filter_field_1": ["ca", "us"],
                "tags": ["category:Shoes", "type:Flats", "color:Blush"],
                "price": 39.00,
                "title": "Blush Ballet Flats",
                "aux_value": 2.73,
                "image_url": image_url
            },
            {
                "_id": "27",
                "filter_field_1": ["us", "mx"],
                "tags": ["category:Shoes", "type:Slip-On", "color:Navy"],
                "price": 44.50,
                "title": "Navy Slip-On Loafers",
                "aux_value": 1.05,
                "image_url": image_url
            },
            {
                "_id": "28",
                "filter_field_1": ["us", "gb", "au"],
                "tags": ["category:Shoes", "type:Oxfords", "color:Black"],
                "price": 70.00,
                "title": "Classic Black Oxfords",
                "aux_value": 2.68,
                "image_url": image_url
            },
            {
                "_id": "29",
                "filter_field_1": ["eu", "us", "ca"],
                "tags": ["category:Shoes", "type:Running", "color:Green"],
                "price": 55.99,
                "title": "Forest Trail Runners",
                "aux_value": 3.14,
                "image_url": image_url
            }
        ]

        res = cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=documents,
                index_name=cls.index_name,
                mappings={
                    "multimodal_combination": {
                        "type": "multimodal_combination",
                        "weights": {
                            "title": 0.99,
                            "image_url": 0.1  # Give it a low weight as it just a dummy image URL
                        }
                    },

                },
                tensor_fields=['title, multimodal_combination'],
            )
        )

    def setUp(self):
        pass  # Override to avoid running the parent class setup that clears the index

    @classmethod
    def _search_helper(
            cls, query: str,
            filter: Optional[str] = None,
            relevance_cutoff: Optional[dict] = None,
            sort_by: Optional[dict] = None,
            limit: int = 10, offset: int = 0,
            hybrid_parameters: Optional[dict] = None,
            attributes_to_retrieve: Optional[list] = None
    ) -> dict:
        """Helper method to perform search with consistent parameters."""

        if hybrid_parameters is None:
            hybrid_parameters = {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5
            }

        result = json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": hybrid_parameters,
                "relevanceCutoff": relevance_cutoff,
                "filter": filter,
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset,
                "attributesToRetrieve": attributes_to_retrieve
            }
        ).body.decode('utf-8'))
        return result

    def test_relevance_cut_off_with_filters(self):
        """Test relevance cutoff with filters applied."""
        # Test with a filter that should exclude some documents
        result = self._search_helper(
            query="glasses and sunglasses",
            hybrid_parameters={
                "rrfK": 60,
                "searchableAttributesLexical": [
                    "title",
                ],
                "alpha": 0.7
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            limit=10,
            filter="filter_field_1:us"
        )

        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(2, result["_relevantCandidates"])
        self.assertEqual(2, result["_probeCandidates"])
        self.assertEqual(3, result["_postProcessCandidates"])
        self.assertEqual({'0', '1', "4"}, set(ids))

    def test_relevance_cut_off_with_incorrect_lexical_searchable_fields(self):
        """It is expected that the relevance cutoff will return 0 results
        if the searchableAttributesLexical is not set correctly."""

        result = self._search_helper(
            query="glasses and sunglasses",
            hybrid_parameters={
                "rrfK": 60,
                "searchableAttributesLexical": [
                    "image_url",
                ],
                "alpha": 0.7
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            limit=10,
        )
        ids = [hit["_id"] for hit in result["hits"]]

        self.assertEqual(0, result["_relevantCandidates"])
        self.assertEqual(0, result["_probeCandidates"])
        self.assertEqual([], ids)

    def test_sort_works_and_returns_all_the_results(self):
        result = self._search_helper(
            query="fashion things",
            hybrid_parameters={
                "rrfK": 60,
                "searchableAttributesLexical": [
                    "image_url",
                ],
                "alpha": 0.7
            },
            sort_by={
                "fields": [{"field_name": "price", "order": "asc"}],
            },
            limit=10,
        )
        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(['12', '7', '18', '11', '3', '16', '0', '4', '14', '5'], ids)

    @pytest.mark.skip_for_multinode("Multi-node will not return the same results as single-node for relevance cutoff")
    def test_lexical_score_modifiers_should_work_with_relevance_cutoff(self):
        """Test that lexical score modifiers work with relevance cutoff."""
        result = self._search_helper(
            query="glasses or shoes or hats",
            hybrid_parameters={
                "rrfK": 60,
                "alpha": 0.7,
                "searchableAttributesLexical": [
                    "title",
                ],
                "scoreModifiersLexical": {
                    "add_to_score": [{
                        "field_name": "aux_value",
                        "weight": 0.001
                    }]
                }
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.7}
            },
            sort_by={
                "fields": [{"field_name": "price", "order": "asc"}],
            },
            limit=10,
        )

        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(1, result["_relevantCandidates"])
        self.assertEqual(6, result["_probeCandidates"])
        self.assertEqual(['4', '17'], ids)

    def test_attributes_to_retrieve_works_as_expected(self):
        """Test that attributes_to_retrieve works as expected with relevance cutoff."""
        result = self._search_helper(
            query="fashion things",
            hybrid_parameters={
                "rrfK": 60,
                "searchableAttributesLexical": [
                    "image_url",
                ],
                "alpha": 0.7
            },
            sort_by={
                "fields": [{"field_name": "price", "order": "asc"}],
            },
            limit=10,
            attributes_to_retrieve=["_id", "title", "price"]
        )

        hits = result["hits"]
        self.assertEqual(
            10, len(hits),
            "Relevance cutoff should return 10 results"
        )
        for hit in hits:
            self.assertIn("_id", hit)
            self.assertIn("title", hit)
            self.assertIn("price", hit)
            self.assertNotIn("tags", hit, "tags should not be retrieved")
            self.assertNotIn("image_url", hit, "image_url should not be retrieved")
            self.assertNotIn("aux_value", hit, "aux_value should not be retrieved")

    @pytest.mark.skip_for_multinode("Multi-node will not return the same results as single-node for relevance cutoff")
    def test_relevance_cutoff_with_sort_and_filter(self):
        """Test relevance cutoff with sort and filter applied."""
        # Test with a filter that should exclude some documents
        result = self._search_helper(
            query="glasses, shoes, and hats",
            hybrid_parameters={
                "rrfK": 60,
                "searchableAttributesLexical": [
                    "title",
                ],
                "alpha": 0.7
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "parameters": {"relativeScoreFactor": 0.5}
            },
            sort_by={
                "fields": [{"field_name": "price", "order": "desc"}],
            },
            limit=10,
            filter="filter_field_1:us"
        )

        ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(6, result["_relevantCandidates"])
        self.assertEqual(6, result["_probeCandidates"])
        self.assertEqual(9, result["_postProcessCandidates"])
        self.assertEqual(['13', '20', '10', '17', '1', '14', '4', '0', '16'], ids)


@pytest.mark.skip_for_multinode(
    "Multi-nodes will return different lexical results so we can not assert on the results.")
class TestRelevanceCutoffWithFacetsAndTotalHits(MarqoTestCase):
    """Tests that facets, totalHits, and sortCandidates correctly reflect relevance cutoff
    when used together with sortBy.

    When affectFacets is enabled, facets and totalHits should only count documents that
    pass the relevance cutoff. When overrideSortCandidatesWithRelevantCandidates is enabled,
    _sortCandidates should equal _relevantCandidates. When affectFacets is disabled,
    facets and totalHits should count all matching documents as usual.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.index_request = cls.unstructured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
            collapse_fields=[CollapseField(name="parentId")],
        )
        cls.create_indexes([cls.index_request])
        cls.index_name = cls.index_request.name

        cls.test_docs = [
            {"_id": "doc1", "text": "The quick brown fox jumps over the lazy dog.",
             "color": "red", "price": 9.99, "tags": ["animal", "nature"], "brand": "Marqo", "parentId": "p1"},
            {"_id": "doc2", "text": "Artificial intelligence is transforming the modern world.",
             "color": "red", "price": 24.50, "tags": ["technology", "science"], "brand": "External", "parentId": "p1"},
            {"_id": "doc3", "text": "The sun sets beautifully over the mountain horizon.",
             "color": "red", "price": 4.75, "tags": ["nature", "travel"], "brand": "Marqo", "parentId": "p1"},
            {"_id": "doc4", "text": "Learning a new language opens many doors in life.",
             "color": "red", "price": 49.99, "tags": ["education", "lifestyle"], "brand": "External", "parentId": "p1"},
            {"_id": "doc5", "text": "Fresh coffee in the morning is the best way to start the day.",
             "color": "red", "price": 12.00, "tags": ["food", "lifestyle"], "brand": "Marqo", "parentId": "p2"},
            {"_id": "doc6", "text": "The ocean is home to millions of undiscovered species.",
             "color": "blue", "price": 7.30, "tags": ["nature", "science"], "brand": "External", "parentId": "p2"},
            {"_id": "doc7", "text": "Reading books regularly improves focus and vocabulary.",
             "color": "blue", "price": 33.80, "tags": ["education", "lifestyle"], "brand": "Marqo", "parentId": "p2"},
            {"_id": "doc8", "text": "Space exploration has uncovered fascinating mysteries of the universe.",
             "color": "blue", "price": 18.45, "tags": ["technology", "science"], "brand": "External", "parentId": "p2"},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=cls.test_docs,
                index_name=cls.index_name,
                tensor_fields=['text']
            )
        )

    def setUp(self):
        pass  # Override parent to preserve documents between tests

    @classmethod
    def _search(cls, query="universe ocean intelligence world vocabulary millions day", relevance_cutoff=None,
                facets=None, track_total_hits=None, limit=10, hybrid_parameters=None, sort_by=None, offset=0, alpha=0.5,
                collapse_fields=None, lexical_operand=None):
        if hybrid_parameters is None:
            hybrid_parameters = {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": alpha,
            }
        if lexical_operand is not None:
            hybrid_parameters["lexicalOperand"] = lexical_operand
        search_query_dict = {
            "q": query,
            "searchMethod": SearchMethod.HYBRID,
            "hybridParameters": hybrid_parameters,
            "limit": limit,
            "offset": offset,
            "collapseFields": collapse_fields,
        }
        if relevance_cutoff is not None:
            search_query_dict["relevanceCutoff"] = relevance_cutoff
        if facets is not None:
            search_query_dict["facets"] = facets
        if track_total_hits is not None:
            search_query_dict["trackTotalHits"] = track_total_hits
        if sort_by is not None:
            search_query_dict["sortBy"] = sort_by
        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict=search_query_dict
        ).body.decode('utf-8'))

    def test_baseline_total_hits_counts_all_matches(self):
        """Without relevance cutoff, totalHits should count all matching documents."""
        result = self._search(track_total_hits=True)
        self.assertEqual(8, result["totalHits"])

    def test_baseline_facets_count_all_matches(self):
        """Without relevance cutoff, facets should count all matching documents."""
        result = self._search(
            facets={"fields": {"color": {"type": "string"}}},
        )
        self.assertEqual(5, result["facets"]["color"]["red"]["count"])
        self.assertEqual(3, result["facets"]["color"]["blue"]["count"])

    def test_affect_facets_false_facets_and_total_hits_count_all_matches(self):
        """With affectFacets=False, facets and totalHits count all matches, not just relevant ones."""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.1},
                "affectFacets": False
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
        )
        self.assertEqual(8, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        expected_facets = {
            "color": {"blue": {"count": 3}, "red": {"count": 5}},
        }
        self.assertEqual(expected_facets, result["facets"])

    def test_affect_facets_true_facets_and_total_hits_reflect_relevant_candidates(self):
        """With sortBy + affectFacets=True, facets and totalHits only count relevant documents.
        However, we return more results as _sortCandidates is larger.
        """
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
        )
        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(6, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc5", "doc6"]
        expected_facets = {
            "color": {"blue": {"count": 3}, "red": {"count": 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_override_sort_candidates_aligns_sort_candidates_with_relevant_candidates(self):
        """With overrideSortCandidatesWithRelevantCandidates, _sortCandidates equals _relevantCandidates."""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "overrideSortCandidatesWithRelevantCandidates": True,
                "affectFacets": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
        )
        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            "color": {"blue": {"count": 3}, "red": {"count": 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_override_sort_candidates_aligns_sort_candidates_with_relevant_candidates_with_two_facets(self):
        """With overrideSortCandidatesWithRelevantCandidates, _sortCandidates equals _relevantCandidates."""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "overrideSortCandidatesWithRelevantCandidates": True,
                "affectFacets": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={
                "fields": {
                    "color": {"type": "string"},
                    "brand": {"type": "string"},
                },
            },
            track_total_hits=True,
        )
        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
            'brand': {'External': {'count': 3}, 'Marqo': {'count': 2}}
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_stricter_cutoff_reduces_all_counts_consistently(self):
        """With a stricter relativeScoreFactor (0.5), fewer documents pass the cutoff.
        All counts (totalHits, relevantCandidates, sortCandidates, facets) reflect
        the reduced set consistently."""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.5},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
        )
        self.assertEqual(3, result["totalHits"])
        self.assertEqual(3, result["_relevantCandidates"])
        self.assertEqual(3, result["_sortCandidates"])

        expected_hits = ["doc2", "doc8", "doc6"]
        expected_facets = {
            "color": {"blue": {"count": 2}, "red": {"count": 1}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_cutoff_and_sort_by_and_facets_pagination(self):
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "overrideSortCandidatesWithRelevantCandidates": True,
                "affectFacets": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={
                "fields": {
                    "color": {"type": "string"},
                    "brand": {"type": "string"},
                },
            },
            limit=3,
            offset=3,
            track_total_hits=True,
        )
        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc8", "doc6"] # You should only see 2 results here
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
            'brand': {'External': {'count': 3}, 'Marqo': {'count': 2}}
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_cutoff_and_sort_by_and_facets_collapse_field(self):
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "overrideSortCandidatesWithRelevantCandidates": True,
                "affectFacets": True,
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={
                "fields": {
                    "color": {"type": "string"},
                    "brand": {"type": "string"},
                },
            },
            limit=10,
            track_total_hits=True,
            collapse_fields=[{"name": "parentId"}],
        )
        self.assertEqual(2, result["totalHits"])
        self.assertEqual(2, result["_relevantCandidates"])
        self.assertEqual(2, result["_sortCandidates"])

        expected_hits = ["doc2", "doc6"]  # You should only see 2 results here
        expected_facets = {
            'color': {'blue': {'count':1}, 'red': {'count': 1}},
            'brand': {'External': {'count': 2}}
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_search_operand_and_main_query_or_relevance_cutoff(self):
        """and for main query, or in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "or"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            alpha=0.2,
            lexical_operand="and"
        )

        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

        for hit in result["hits"]:
            self.assertIn("_tensor_score", hit)
            self.assertNotIn("_lexical_score", hit) # You shouldn't see any retrievals from lexical

    def test_search_operand_weakAnd_main_or_relevance_cutoff(self):
        """weakAnd for main query, or in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "or"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand="or"
        )

        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

        # "doc4" is purely retrieved by tensor, others have tensor and lexical scores
        self.assertNotIn("_lexical_score", result["hits"][0])
        for hit in result["hits"][1:]:
            self.assertIn("_lexical_score", hit)

    def test_search_operand_or_main_and_relevance_cutoff(self):
        """or for main query, and in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "and"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand="or"
        )

        self.assertEqual(0, result["totalHits"])
        self.assertEqual(0, result["_relevantCandidates"])
        self.assertEqual(0, result["_sortCandidates"])

        self.assertEqual(0, len(result["hits"]))

    def test_search_operand_none_main_and_relevance_cutoff(self):
        """None for main query, and in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "and"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand=None # use default
        )

        self.assertEqual(0, result["totalHits"])
        self.assertEqual(0, result["_relevantCandidates"])
        self.assertEqual(0, result["_sortCandidates"])

        self.assertEqual(0, len(result["hits"]))

    def test_search_operand_none_main_or_relevance_cutoff(self):
        """None for main query, or in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "or"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand=None # use default
        )

        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_search_operand_none_main_weakAnd_relevance_cutoff(self):
        """None for main query, weakAnd in relevance cutoff query"""
        result = self._search(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
                "lexicalOperand": "weakAnd"
            },
            sort_by={"fields": [{"fieldName": "price"}]},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand=None # use default
        )

        self.assertEqual(5, result["totalHits"])
        self.assertEqual(5, result["_relevantCandidates"])
        self.assertEqual(5, result["_sortCandidates"])

        expected_hits = ["doc4", "doc7", "doc2", "doc8", "doc6"]
        expected_facets = {
            'color': {'blue': {'count': 3}, 'red': {'count': 2}},
        }
        returned_hits = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(expected_hits, returned_hits)

    def test_search_operand_works_with_quoted_queries(self):
        """Ensure quoted queries still work"""
        result = self._search(
            query = '\"void\" \"test\"',
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
            lexical_operand=None # use default
        )
        for hit in result["hits"]:
            self.assertNotIn("_lexical_score", hit)
            self.assertIn("_tensor_score", hit)

    def test_relevance_cutoff_when_min_sort_candidates_and_override_sort_candidates_with_relevant_candidates(self):
        """
        Test a special case that when override_sort_candidates_with_relevant_candidates=True, and relevantCandidates is
        small, and min_sort_candidates is set, we use min_sort_candidates to do facets, retrieval, and sort
        """
        result = self._search(
            query = 'sea solar',
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.2},
                "affectFacets": True,
                "overrideSortCandidatesWithRelevantCandidates": True,
            },
            sort_by={"fields": [{"fieldName": "price"}], "min_sort_candidates": 2},
            facets={"fields": {"color": {"type": "string"}}},
            track_total_hits=True,
        )

        hits = [hit["_id"] for hit in result["hits"]]
        expected_hits = ["doc6", "doc3"]
        self.assertEqual(expected_hits, hits)

        # Facets results is not stable due to different match set, so we only
        # assert the existence
        self.assertIn("facets", result)

        self.assertEqual(2, result["totalHits"])
        self.assertEqual(0, result["_relevantCandidates"])
        self.assertEqual(2, result["_sortCandidates"])

        for hit in result["hits"]:
            self.assertNotIn("_lexical_score", hit)
            self.assertIn("_tensor_score", hit)


@pytest.mark.skip_for_multinode(
    "Multi-nodes will return different lexical results so we can not assert on the results.")
class TestRelevanceCutoffApplyInRetrievalWithLexicalOperand(MarqoTestCase):
    """Tests that applyInRetrieval='tensor' preserves all lexical AND-matched results
    even under a strict cutoff, while using OR for the relevance cutoff probe.

    Documents are crafted so that:
    - 1 doc has ONLY "ocean species" (short text, highest lexical score — the anchor)
    - 8 docs contain "ocean species" plus additional text (longer, lower lexical scores)

    All 9 docs match the AND lexical query for "ocean" AND "species".
    With a strict relativeScoreFactor (0.9), only the short anchor doc (and possibly 1-2
    others) pass the cutoff threshold. With applyInRetrieval='tensor', the lexical leg is
    unrestricted (uses probeDepth), so all 9 AND-matched docs are returned regardless of
    the cutoff count.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.index_request = cls.unstructured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
        )
        cls.create_indexes([cls.index_request])
        cls.index_name = cls.index_request.name

        # 1 short doc (highest lexical score — the cutoff anchor)
        # 8 longer docs (all contain "ocean" and "species" but diluted by extra text)
        # category: "deep" for 5 docs, "surface" for 4 docs
        cls.test_docs = [
            {"_id": "anchor", "text": "ocean species", "price": 50.00, "category": "deep"},
            {"_id": "long_1", "text": "The deep ocean is teeming with diverse species of fish and coral in tropical waters.", "price": 10.00, "category": "deep"},
            {"_id": "long_2", "text": "Ocean currents transport species across vast marine distances around the globe.", "price": 20.00, "category": "surface"},
            {"_id": "long_3", "text": "Protecting ocean habitats is essential for preserving endangered species worldwide.", "price": 30.00, "category": "deep"},
            {"_id": "long_4", "text": "Scientists discovered new ocean species living near hydrothermal vents on the seabed.", "price": 40.00, "category": "deep"},
            {"_id": "long_5", "text": "The Arctic ocean supports species adapted to extreme cold temperatures and ice.", "price": 15.00, "category": "surface"},
            {"_id": "long_6", "text": "Pollution threatens ocean species that depend on clean water for survival and growth.", "price": 25.00, "category": "surface"},
            {"_id": "long_7", "text": "Mapping the ocean floor revealed species previously unknown to marine biology researchers.", "price": 35.00, "category": "deep"},
            {"_id": "long_8", "text": "Climate change alters ocean temperatures affecting species migration patterns and habitats.", "price": 45.00, "category": "surface"},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=cls.test_docs,
                index_name=cls.index_name,
                tensor_fields=['text']
            )
        )

    def setUp(self):
        pass

    @classmethod
    def _search(cls, query, relevance_cutoff=None, lexical_operand=None,
                limit=10, offset=0, alpha=0.5, sort_by=None, facets=None,
                track_total_hits=None):
        hybrid_parameters = {
            "retrievalMethod": "disjunction",
            "rankingMethod": "rrf",
            "alpha": alpha,
        }
        if lexical_operand is not None:
            hybrid_parameters["lexicalOperand"] = lexical_operand
        search_query_dict = {
            "q": query,
            "searchMethod": SearchMethod.HYBRID,
            "hybridParameters": hybrid_parameters,
            "limit": limit,
            "offset": offset,
        }
        if relevance_cutoff is not None:
            search_query_dict["relevanceCutoff"] = relevance_cutoff
        if sort_by is not None:
            search_query_dict["sortBy"] = sort_by
        if facets is not None:
            search_query_dict["facets"] = facets
        if track_total_hits is not None:
            search_query_dict["trackTotalHits"] = track_total_hits
        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict=search_query_dict
        ).body.decode('utf-8'))

    def test_apply_in_tensor_preserves_all_lexical_and_matches(self):
        """With applyInRetrieval='tensor', a strict cutoff limits tensor retrieval but
        leaves lexical unrestricted. All 9 documents matching the AND lexical query
        ("ocean" AND "species") must appear in results with _lexical_score.

        The anchor doc ("ocean species") scores highest in lexical, so with
        relativeScoreFactor=0.9 the threshold is very high and most longer docs
        fall below it — but that only affects the tensor leg.
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "tensor",
            },
        )

        expected_ids = {'long_3', 'long_2', 'long_5', 'long_4', 'long_7', 'anchor', 'long_8', 'long_1', 'long_6'}
        returned_ids = {r['_id'] for r in result['hits']}

        # We make expected ids a set as the lexical score can vary in different runs
        self.assertEqual(1, result["_relevantCandidates"])
        self.assertEqual(9, result["_postProcessCandidates"])
        self.assertEqual(expected_ids, returned_ids)

        self.assertIn(
            "_lexical_score", result["hits"][0]
        )
        self.assertIn("_tensor_score", result["hits"][0])

        # Since _relevantCandidates is 1, all following results should not have tensor score
        for hit in result['hits'][1:]:
            self.assertIn("_lexical_score", hit)
            self.assertNotIn("_tensor_score", hit)

    def test_apply_in_tensor_preserves_update_to_probe_depth_lexical_matches(self):
        """A test to ensure probeDepth can be used to cap the lexical retrievals even if it is not cut.
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 5,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "tensor",
            },
        )

        # Can't assert on the actual returned hits as the returned documents vary accross different runs
        self.assertEqual(5, len(result['hits']))
        # We make expected ids a set as the lexical score can vary in different runs
        self.assertEqual(1, result["_relevantCandidates"])
        # We only get 5 results because lexical only return 5 results
        self.assertEqual(5, result["_postProcessCandidates"])

        self.assertIn(
            "_lexical_score", result["hits"][0]
        )
        self.assertIn("_tensor_score", result["hits"][0])

        # Since _relevantCandidates is 1, all following results should not have tensor score
        for hit in result['hits'][1:]:
            self.assertIn("_lexical_score", hit)
            self.assertNotIn("_tensor_score", hit)


    def test_apply_in_tensor_with_sort_by_preserves_all_lexical_and_matches(self):
        """With applyInRetrieval='tensor' + sortBy, all 9 AND-matched docs should
        still appear and be sorted by price. The cutoff only limits tensor retrieval,
        so lexical returns all matches unrestricted. Sort ordering uses all candidates.

        Expected order by price ascending:
        long_1(10) < long_5(15) < long_2(20) < long_6(25) < long_3(30)
        < long_7(35) < long_4(40) < long_8(45) < anchor(50)
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "tensor",
            },
            sort_by={"fields": [{"fieldName": "price", "order": "asc"}]},
        )

        returned_ids = [hit["_id"] for hit in result["hits"]]
        expected_order = [
            "long_1", "long_5", "long_2", "long_6", "long_3",
            "long_7", "long_4", "long_8", "anchor"
        ]
        # Sorted by price ascending
        self.assertEqual(expected_order, returned_ids)

        # relevantCandidates still reflects the strict cutoff
        self.assertEqual(1, result["_relevantCandidates"])
        self.assertEqual(9, result["_sortCandidates"])

    def test_apply_in_tensor_with_affect_facets_false(self):
        """With affectFacets=False (default), facets count all matching documents
        regardless of cutoff. All 9 docs match, so facets should reflect all 9.
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "tensor",
                "affectFacets": False,
            },
            facets={"fields": {"category": {"type": "string"}}},
            track_total_hits=True,
        )

        # All 9 docs returned
        self.assertEqual(9, len(result["hits"]))
        self.assertEqual(1, result["_relevantCandidates"])

        # Facets count all matching docs (not limited by cutoff)
        # While this is correct in this example, it is not generally correct in the real case as facets
        # results might match more result. A reimplementation is needed to solve this problem
        expected_facets = {
            "category": {"deep": {"count": 5}, "surface": {"count": 4}},
        }
        self.assertEqual(expected_facets, result["facets"])
        self.assertEqual(9, result["totalHits"])

    def test_apply_in_tensor_with_affect_facets_true(self):
        """With affectFacets=True, facets and totalHits only count docs that pass
        the cutoff. Since relevantCandidates=1 (only the anchor passes the strict
        cutoff), facets reflect just that 1 doc — even though 9 docs are returned.

        This is the expected "conservative" behavior: facets are smaller than the
        actual result set because cutoff controls only one retrieval leg.
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "tensor",
                "affectFacets": True,
            },
            facets={"fields": {"category": {"type": "string"}}},
            track_total_hits=True,
        )

        # All 9 docs still returned (lexical leg unrestricted)
        self.assertEqual(9, len(result["hits"]))
        self.assertEqual(1, result["_relevantCandidates"])

        # Facets is only applied to _relevantCandidates
        expected_facets = {
            "category": {"deep": {"count": 1}},
        }
        self.assertEqual(expected_facets, result["facets"])

    def test_apply_in_both_blocks_irrelevant_documents_in_both_retrievals(self):
        """With applyInRetrieval='both', the relevance cutoff is applied to both retrievals
        """
        result = self._search(
            query='ocean species',
            lexical_operand="and",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 1000,
                "parameters": {"relativeScoreFactor": 0.9},
                "lexicalOperand": "or",
                "applyInRetrieval": "both",
            },
        )

        expected_ids = {'anchor',}
        returned_ids = {r['_id'] for r in result['hits']}

        # We make expected ids a set as the lexical score can vary in different runs
        self.assertEqual(1, result["_relevantCandidates"])
        self.assertEqual(1, result["_postProcessCandidates"])
        self.assertEqual(expected_ids, returned_ids)

        self.assertIn(
            "_lexical_score", result["hits"][0]
        )
        self.assertIn("_tensor_score", result["hits"][0])

    def test_apply_in_tensor_with_sort_by_preserves_all_lexical_and_matches_pagination(self):
        """Test the pagination behaviour of 'applyInRetrieval'
        """

        limit = 2

        expected_order = [
            "long_1", "long_5", "long_2", "long_6", "long_3",
            "long_7", "long_4", "long_8", "anchor"
        ]

        for offset in range(0, len(expected_order), limit):
            result = self._search(
                query='ocean species',
                lexical_operand="and",
                relevance_cutoff={
                    "method": "relative_max_score",
                    "probeDepth": 1000,
                    "parameters": {"relativeScoreFactor": 0.9},
                    "lexicalOperand": "or",
                    "applyInRetrieval": "tensor",
                    "overrideTotalHitsWithPostProcessCandidates": True,
                },
                sort_by={"fields": [{"fieldName": "price", "order": "asc"}]},
                limit = limit,
                offset=offset,
            )
            returned_ids = [r["_id"] for r in result['hits']]
            expected_returned_ids = expected_order[offset: offset + limit]

            self.assertEqual(expected_returned_ids, returned_ids)
            # relevantCandidates still reflects the strict cutoff
            self.assertEqual(1, result["_relevantCandidates"])
            self.assertEqual(9, result["_postProcessCandidates"])
            self.assertEqual(9, result["_sortCandidates"])
            self.assertEqual(9, result["totalHits"])