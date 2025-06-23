import json

from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.tensor_search.api import search
from marqo.tensor_search.enums import SearchMethod


class TestSearchRelevanceCutoffFeature(MarqoTestCase):
    """
    Integration tests for the relevance cutoff feature in Marqo search.
    
    This test suite validates the end-to-end functionality of relevance cutoff methods
    that filter search results based on relevance scores to improve result quality.
    
    Test Data Setup:
    ================
    The test uses 10 carefully crafted documents with varying semantic relevance:
    
    High Relevance (to ML/AI queries):
    - ml_ai_guide: "Comprehensive guide to machine learning and artificial intelligence..."
    - ai_research: "Latest research in artificial intelligence, machine learning models..."
    - ml_tutorial: "Machine learning tutorial covering supervised learning..."
    
    Medium Relevance:
    - programming: "Software development best practices, programming languages..."
    - data_science: "Data science fundamentals including statistics, data analysis..."
    - web_dev: "Web development frameworks, frontend technologies..."
    
    Low Relevance (to ML/AI queries):
    - cooking: "Traditional cooking techniques, recipe development..."
    - gardening: "Organic gardening tips, plant care, soil management..."
    - sports: "Basketball training drills, team strategies..."
    - fashion: "Fashion trends, clothing design principles..."
    
    Expected Behavior:
    ==================
    For ML/AI queries ("machine learning artificial intelligence"):
    - Without cutoff: All 10 documents returned, ranked by semantic similarity
    - With strict cutoff (0.8+ relative factor): Only top 2-4 highly relevant docs
    - With lenient cutoff (0.1-0.3 relative factor): 6-9 documents, filtering lowest relevance
    - Gap detection: Filters documents with significant score gaps from top results
    - Mean+StdDev: Filters documents more than N standard deviations below mean score
    
    For Programming queries ("programming software development"):
    - Programming doc should rank highest, ML/AI docs still relevant but lower
    - Cooking/gardening/sports should rank lowest and be filtered by cutoffs
    
    For Cooking queries ("cooking recipes food"):
    - Cooking doc should rank highest, other domains much lower
    - Tech domains (ML/AI/programming) may still have some relevance
    
    Metadata Validation:
    ====================
    All search results should include:
    - _relevanceCandidates: Number of documents considered for relevance scoring
    - _probeCandidates: Number of documents examined during probe phase for cutoff calculation
    
    These metadata fields provide insight into the cutoff algorithm's operation.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name="hf/e5-base-v2")
        )

        cls.create_indexes([semi_structured_default_text_index])

        cls.index_name = semi_structured_default_text_index.name

        # Documents with varying relevance to test relevance cutoff functionality
        test_relevance_cutoff_docs = [
            # High relevance: Direct match for "machine learning artificial intelligence"
            {"_id": "ml_ai_guide", "content": "Comprehensive guide to machine learning and artificial intelligence algorithms, neural networks, and deep learning techniques", "sort_field_1": 9.5},
            
            # High relevance: Related ML/AI content  
            {"_id": "ai_research", "content": "Latest research in artificial intelligence, machine learning models, and AI applications in various industries", "sort_field_1": 8.7},
            
            # Medium-High relevance: ML focused
            {"_id": "ml_tutorial", "content": "Machine learning tutorial covering supervised learning, unsupervised learning, and reinforcement learning concepts", "sort_field_1": 7.2},
            
            # Medium relevance: Tech but not ML/AI specific
            {"_id": "programming", "content": "Software development best practices, programming languages, and coding methodologies for modern applications", "sort_field_1": 5.5},
            
            # Medium relevance: Data science (related field)
            {"_id": "data_science", "content": "Data science fundamentals including statistics, data analysis, visualization, and predictive modeling techniques", "sort_field_1": 6.1},
            
            # Lower relevance: Technology but different domain
            {"_id": "web_dev", "content": "Web development frameworks, frontend technologies, backend systems, and modern web application architecture", "sort_field_1": 4.0},
            
            # Lower relevance: Different domain
            {"_id": "cooking", "content": "Traditional cooking techniques, recipe development, culinary arts, and international cuisine preparation methods", "sort_field_1": 2.3},
            
            # Very low relevance: Completely unrelated
            {"_id": "gardening", "content": "Organic gardening tips, plant care, soil management, and sustainable growing practices for home gardens", "sort_field_1": 1.5},
            
            # Very low relevance: Sports content
            {"_id": "sports", "content": "Basketball training drills, team strategies, player fitness programs, and competitive sports psychology", "sort_field_1": 1.0},
            
            # Very low relevance: Fashion content
            {"_id": "fashion", "content": "Fashion trends, clothing design principles, textile materials, and seasonal style recommendations", "sort_field_1": 0.5}
        ]

        _ = cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=test_relevance_cutoff_docs,
                index_name=semi_structured_default_text_index.name,
                documents=test_relevance_cutoff_docs,
                tensor_fields=['content'],
            )
        )

        # Verify documents are indexed correctly for ML/AI query
        normal_search_res = cls._help_sort_function(query="machine learning artificial intelligence")["hits"]
        if len(normal_search_res) != 10:
            raise RuntimeError(
                f"Expected 10 documents in index, but got {len(normal_search_res)}"
            )
        
        # Verify the most relevant documents are at the top
        top_results = [r["_id"] for r in normal_search_res[:3]]
        expected_top_results = ["ml_ai_guide", "ai_research", "ml_tutorial"]
        for expected_id in expected_top_results:
            if expected_id not in top_results:
                raise RuntimeError(
                    f"Expected high relevance documents {expected_top_results} to be in top 3 results, "
                    f"but got {top_results}"
                )

    def setUp(self):
        """Ensure documents are not changed before each test."""
        if 10 != self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents:
            raise RuntimeError(
                f"Expected 10 documents in index {self.index_name} for sorting tests"
            )

    def tearDown(self):
        """Ensure documents are not changed after each test."""
        if 10 != self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents:
            raise RuntimeError(
                f"Expected 10 documents in index {self.index_name} for sorting tests"
            )

    @classmethod
    def _help_sort_function(cls, query: Optional[str] = "machine learning artificial intelligence",
                            sort_by: Optional[dict] = None,
                            relevance_cutoff: Optional[dict] = None,
                            limit=10, offset=0) -> dict:
        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.5,
                },
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset,
                "relevanceCutoff": relevance_cutoff
            }
        ).body.decode('utf-8'))

    def test_relevance_cutoff_relative_max_score(self):
        """Test relevance cutoff with relative_max_score method using ML/AI query"""
        # First get baseline results without cutoff to understand the data
        baseline = self._help_sort_function(query="machine learning artificial intelligence")
        
        # Test with lenient cutoff (0.1) - should return most documents
        result_lenient = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.1}
            }
        )
        self.assertGreater(len(result_lenient["hits"]), 8, "Should return some documents with lenient cutoff")
        
        # Test with strict cutoff (0.8) - should return fewer documents
        result_strict = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "relative_max_score", 
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.8}
            }
        )
        
        # Strict cutoff should return fewer or equal documents than lenient
        self.assertLessEqual(len(result_strict["hits"]), len(result_lenient["hits"]),
                           "Strict cutoff should not return more documents than lenient cutoff")
        
        # The highest scoring documents should appear in both results
        if len(result_strict["hits"]) > 0 and len(baseline["hits"]) > 0:
            # Get the top document from baseline (highest score)
            top_doc_id = baseline["hits"][0]["_id"]
            strict_ids = [hit["_id"] for hit in result_strict["hits"]]
            self.assertIn(top_doc_id, strict_ids, 
                         "Highest scoring document should survive strict cutoff")
        
        # Verify probe candidates are reasonable (should be <= total documents in index)
        self.assertLessEqual(result_lenient["_probeCandidates"], 10, 
                           "Probe candidates should not exceed total documents in index")
        self.assertLessEqual(result_strict["_probeCandidates"], 10,
                           "Probe candidates should not exceed total documents in index")

    def test_relevance_cutoff_gap_detection(self):
        """Test relevance cutoff with gap_detection method"""
        # Get baseline without cutoff
        baseline = self._help_sort_function(query="machine learning artificial intelligence")
        
        # Apply gap detection cutoff
        result = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "gap_detection",
                "probeDepth": 50
            }
        )
        
        self.assertGreater(len(result["hits"]), 0, "Should return some documents")
        self.assertLessEqual(len(result["hits"]), len(baseline["hits"]), 
                           "Gap detection should not increase document count")
        self.assertIn("_relevanceCandidates", result, "Result should contain _relevanceCandidates metadata")
        self.assertIn("_probeCandidates", result, "Result should contain _probeCandidates metadata")
        
        # Verify all returned documents have reasonable scores
        for hit in result["hits"]:
            self.assertIn("_score", hit)
            self.assertIsInstance(hit["_score"], (int, float))
            self.assertGreater(hit["_score"], 0, "All returned documents should have positive scores")

    def test_relevance_cutoff_mean_std_dev(self):
        """Test relevance cutoff with mean_std_dev method on ML/AI query"""
        # Get baseline without cutoff
        baseline = self._help_sort_function(query="machine learning artificial intelligence")
        
        # Test with conservative std dev factor (1.0) - should be more selective
        result_conservative = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "mean_std_dev",
                "probeDepth": 50,
                "parameters": {"stdDevFactor": 1.0}
            }
        )
        self.assertGreater(len(result_conservative["hits"]), 0, "Should return some documents")
        self.assertLessEqual(len(result_conservative["hits"]), len(baseline["hits"]),
                           "Conservative filter should not increase document count")

        
        # Test with aggressive std dev factor (2.5) - should be less selective  
        result_aggressive = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "mean_std_dev",
                "probeDepth": 50, 
                "parameters": {"stdDevFactor": 2.5}
            }
        )
        self.assertGreater(len(result_aggressive["hits"]), 0, "Should return documents with high std dev factor")
        self.assertGreaterEqual(len(result_aggressive["hits"]), len(result_conservative["hits"]), 
                               "Higher std dev factor should return same or more documents")
        
        # The top scoring document should survive both filters
        if len(baseline["hits"]) > 0:
            top_doc_id = baseline["hits"][0]["_id"]
            conservative_ids = [hit["_id"] for hit in result_conservative["hits"]]
            aggressive_ids = [hit["_id"] for hit in result_aggressive["hits"]]
            
            if len(result_conservative["hits"]) > 0:
                self.assertIn(top_doc_id, conservative_ids, "Top document should survive conservative filter")
            if len(result_aggressive["hits"]) > 0:
                self.assertIn(top_doc_id, aggressive_ids, "Top document should survive aggressive filter")

    def test_relevance_cutoff_comparison_with_without(self):
        """Test that relevance cutoff actually filters documents appropriately"""
        # Search without cutoff
        result_no_cutoff = self._help_sort_function(
            query="machine learning artificial intelligence"
        )
        
        # Search with moderate cutoff
        result_with_cutoff = self._help_sort_function(
            query="machine learning artificial intelligence",
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.6}
            }
        )
        
        # Cutoff should return fewer or equal documents
        self.assertLessEqual(len(result_with_cutoff["hits"]), len(result_no_cutoff["hits"]),
                           "Relevance cutoff should not increase document count")
        
        # All documents in cutoff result should also be in no-cutoff result
        no_cutoff_ids = set(hit["_id"] for hit in result_no_cutoff["hits"])
        with_cutoff_ids = set(hit["_id"] for hit in result_with_cutoff["hits"])
        self.assertTrue(with_cutoff_ids.issubset(no_cutoff_ids),
                      "All cutoff results should be subset of no-cutoff results")
        
        # If cutoff filtered documents, the top scoring docs should be retained
        if len(result_with_cutoff["hits"]) < len(result_no_cutoff["hits"]) and len(result_no_cutoff["hits"]) > 0:
            # Check that highest scoring documents are more likely to survive
            top_3_no_cutoff = [hit["_id"] for hit in result_no_cutoff["hits"][:3]]
            surviving_from_top_3 = sum(1 for doc_id in top_3_no_cutoff if doc_id in with_cutoff_ids)
            
            # At least some of the top 3 should survive the cutoff
            if len(result_with_cutoff["hits"]) > 0:
                self.assertGreater(surviving_from_top_3, 0, 
                                 "Some of the highest scoring documents should survive cutoff")

    def test_relevance_cutoff_different_query_types(self):
        """Test that relevance cutoff works consistently across different query types"""
        queries_to_test = [
            "machine learning artificial intelligence",
            "programming software development", 
            "cooking recipes food preparation"
        ]
        
        for query in queries_to_test:
            with self.subTest(query=query):
                # Get baseline without cutoff
                baseline = self._help_sort_function(query=query)
                
                # Apply cutoff
                result_with_cutoff = self._help_sort_function(
                    query=query,
                    relevance_cutoff={
                        "method": "relative_max_score",
                        "probeDepth": 50,
                        "parameters": {"relativeScoreFactor": 0.5}
                    }
                )
                
                # Basic sanity checks that apply to any query
                self.assertGreater(len(baseline["hits"]), 0, f"Baseline should return results for query: {query}")
                self.assertLessEqual(len(result_with_cutoff["hits"]), len(baseline["hits"]),
                                   f"Cutoff should not increase results for query: {query}")
                
                # If there are results after cutoff, they should be a subset of baseline
                if len(result_with_cutoff["hits"]) > 0:
                    baseline_ids = set(hit["_id"] for hit in baseline["hits"])
                    cutoff_ids = set(hit["_id"] for hit in result_with_cutoff["hits"])
                    self.assertTrue(cutoff_ids.issubset(baseline_ids),
                                  f"Cutoff results should be subset of baseline for query: {query}")
                    
                    # Top scoring document should be likely to survive cutoff
                    if len(baseline["hits"]) > 0:
                        top_doc_id = baseline["hits"][0]["_id"]
                        self.assertIn(top_doc_id, cutoff_ids,
                                    f"Top document should survive cutoff for query: {query}")

    def test_relevance_cutoff_with_sort_by(self):
        """Test relevance cutoff works correctly with sort_by"""
        result = self._help_sort_function(
            sort_by={
                "fields": [
                    {
                        "field_name": "sort_field_1",
                        "order": "desc",
                        "missing": "last"
                    }
                ]
            },
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.5}
            }
        )
        
        self.assertGreater(len(result["hits"]), 0, "Should return documents with both sort and cutoff")
        
        # Verify sorting is maintained after cutoff
        sort_values = [hit.get("sort_field_1") for hit in result["hits"] if "sort_field_1" in hit]
        if len(sort_values) > 1:
            for i in range(len(sort_values) - 1):
                if sort_values[i] is not None and sort_values[i+1] is not None:
                    self.assertGreaterEqual(sort_values[i], sort_values[i+1], 
                                          "Results should maintain descending sort order")

    def test_relevance_cutoff_with_limit_and_offset(self):
        """Test relevance cutoff works with pagination"""
        # First get results without cutoff for comparison
        no_cutoff_result = self._help_sort_function(limit=5, offset=0)
        
        # Apply cutoff with pagination
        cutoff_result = self._help_sort_function(
            limit=5,
            offset=0,
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.3}
            }
        )
        
        self.assertLessEqual(len(cutoff_result["hits"]), 5, "Should respect limit")
        self.assertLessEqual(len(cutoff_result["hits"]), len(no_cutoff_result["hits"]), 
                           "Cutoff should not increase result count")

    def test_relevance_cutoff_preserves_document_structure(self):
        """Test that cutoff preserves the structure of returned documents"""
        result = self._help_sort_function(
            relevance_cutoff={
                "method": "relative_max_score",
                "probeDepth": 50,
                "parameters": {"relativeScoreFactor": 0.4}
            }
        )
        
        self.assertIn("hits", result)
        for hit in result["hits"]:
            self.assertIn("_id", hit)
            self.assertIn("_score", hit)
            self.assertIn("content", hit)
            # Verify score is a valid number
            self.assertIsInstance(hit["_score"], (int, float))
            self.assertGreaterEqual(hit["_score"], 0)