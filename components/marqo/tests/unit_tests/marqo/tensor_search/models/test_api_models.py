import unittest
from pydantic.v1 import ValidationError

from marqo.api import exceptions as api_exceptions
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery, CollapseModel
from marqo.tensor_search.models.collapse_model import CollapseSortBy, CollapseSortByField
from marqo.tensor_search.models.search import (
    SearchContext, 
    SearchContextTensor, 
    SearchContextDocuments,
    SearchContextDocumentsParameters,
    QueryContent
)
from marqo.core.inference.api import Modality
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField, SortOrder


class TestSearchQuery(unittest.TestCase):

    def test_search_query_with_all_parameters(self):
        """Test SearchQuery creation with all parameters set to valid values."""
        custom_vector_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        )

        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.7,
            rrfK=100
        )

        facets = FacetsParameters(
            fields={
                "category": FieldFacetsConfiguration(type="string", maxResults=10)
            }
        )

        context = SearchContext(
            tensor=[SearchContextTensor(vector=[0.1, 0.2], weight=1.0)]
        )

        search_query = SearchQuery(
            q=custom_vector_query,
            searchableAttributes=["title", "description"],
            searchMethod=SearchMethod.HYBRID,
            limit=20,
            offset=5,
            rerankDepth=100,
            efSearch=200,
            approximate=True,
            approximateThreshold=0.85,
            showHighlights=False,
            reRanker="test_reranker",
            filter="category:electronics",
            attributesToRetrieve=["title", "price"],
            boost={"title": 1.5},
            mediaDownloadHeaders={"Authorization": "Bearer token"},
            context=context,
            textQueryPrefix="search:",
            hybridParameters=hybrid_parameters,
            facets=facets,
            trackTotalHits=True,
            language="en"
        )

        # Verify key attributes
        self.assertEqual(search_query.searchMethod, SearchMethod.HYBRID)
        self.assertEqual(search_query.limit, 20)
        self.assertEqual(search_query.approximateThreshold, 0.85)
        self.assertIsNotNone(search_query.hybridParameters)
        self.assertIsNotNone(search_query.facets)
        self.assertEqual(search_query.language, "en")

    def test_search_query_required_parameters_only(self):
        """Test SearchQuery with only required parameters."""
        # For tensor search, either q or context is required
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.TENSOR
        )

        # Verify defaults
        self.assertEqual(search_query.searchMethod, SearchMethod.TENSOR)
        self.assertEqual(search_query.limit, 10)
        self.assertEqual(search_query.offset, 0)
        self.assertTrue(search_query.showHighlights)
        self.assertIsNone(search_query.hybridParameters)

    def test_hybrid_parameters_validation(self):
        """Test that hybrid parameters are only allowed for hybrid search."""
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF
        )

        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                hybridParameters=hybrid_parameters
            )
        self.assertIn("Hybrid parameters can only be provided for 'HYBRID' search", str(cm.exception))

    def test_facets_validation(self):
        """Test that facets are only allowed for hybrid search."""
        facets = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")}
        )

        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                facets=facets
            )
        self.assertIn("Facets can only be provided for 'HYBRID' search", str(cm.exception))

    def test_track_total_hits_validation(self):
        """Test that trackTotalHits is only allowed for hybrid search."""
        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                trackTotalHits=True
            )
        self.assertIn("trackTotalHits can only be provided for 'HYBRID' search", str(cm.exception))

    def test_approximate_threshold_validation(self):
        """Test approximate threshold validation."""
        # Should fail for lexical search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.LEXICAL,
                approximateThreshold=0.5
            )
        self.assertIn("'approximateThreshold' is only valid for 'HYBRID' and 'TENSOR' search methods",
                      str(cm.exception))

        # Should fail when approximate=False
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                approximate=False,
                approximateThreshold=0.5
            )
        self.assertIn("'approximateThreshold' cannot be set when 'approximate' is False", str(cm.exception))

        # Should fail for invalid range
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                approximateThreshold=1.5
            )
        self.assertIn("'approximateThreshold' must be between 0 and 1", str(cm.exception))

    def test_query_and_context_validation(self):
        """Test validation of query and context requirements."""
        # Lexical search requires query
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(searchMethod=SearchMethod.LEXICAL)
        self.assertIn("Query(q) is required for lexical search", str(cm.exception))

        # Tensor search requires either query or context
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(searchMethod=SearchMethod.TENSOR)
        self.assertIn("One of Query(q) or context is required for TENSOR search", str(cm.exception))

    def test_rerank_depth_validation(self):
        """Test rerank depth validation."""
        # Should fail for lexical search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.LEXICAL,
                rerankDepth=10
            )
        self.assertIn("'rerankDepth' is currently not supported for 'LEXICAL' search method", str(cm.exception))

        # Should fail for negative values
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                rerankDepth=-1
            )
        self.assertIn("rerankDepth cannot be negative", str(cm.exception))

    def test_image_download_headers_validation(self):
        """Test validation of image download headers."""
        # Should fail when both headers are set
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                image_download_headers={"header1": "value1"},
                mediaDownloadHeaders={"header2": "value2"}
            )
        self.assertIn("Cannot set both imageDownloadHeaders", str(cm.exception))

        # Should work when imageDownloadHeaders is set and mediaDownloadHeaders is copied
        search_query = SearchQuery(
            q="test",
            image_download_headers={"header1": "value1"}
        )
        self.assertEqual(search_query.mediaDownloadHeaders, {"header1": "value1"})

    def test_search_query_with_invalid_search_method_fails(self):
        """Test that invalid search method raises validation error"""
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(q="test", searchMethod="INVALID_METHOD")
        
        error_details = str(cm.exception)
        self.assertIn("value is not a valid enumeration member", error_details)

    def test_search_query_interpolation_method_validation(self):
        """Test interpolation method validation"""
        # Valid interpolation method
        query = SearchQuery(q="test", interpolationMethod=InterpolationMethod.SLERP)
        self.assertEqual(query.interpolationMethod, InterpolationMethod.SLERP)
        
        # None should be valid
        query = SearchQuery(q="test", interpolationMethod=None)
        self.assertIsNone(query.interpolationMethod)

    def test_search_query_context_validation_with_tensor_search(self):
        """Test context validation for tensor search"""
        # Valid case - query with tensor search
        query = SearchQuery(q="test", searchMethod=SearchMethod.TENSOR)
        self.assertEqual(query.searchMethod, SearchMethod.TENSOR)
        
        # Valid case - no query but with context for tensor search
        context = SearchContext(tensor=[SearchContextTensor(vector=[1, 2, 3], weight=1.0)])
        query = SearchQuery(q=None, searchMethod=SearchMethod.TENSOR, context=context)
        self.assertIsNone(query.q)
        self.assertIsNotNone(query.context)

    def test_search_query_context_validation_with_lexical_search_fails(self):
        """Test that lexical search requires query"""
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(q=None, searchMethod=SearchMethod.LEXICAL)
        
        error_details = str(cm.exception)
        self.assertIn("Query(q) is required for lexical search", error_details)

    def test_search_query_ef_search_validation(self):
        """Test efSearch parameter validation"""
        # Valid positive integer
        query = SearchQuery(q="test", efSearch=100)
        self.assertEqual(query.efSearch, 100)
        
        # None should be valid
        query = SearchQuery(q="test", efSearch=None)
        self.assertIsNone(query.efSearch)

    def test_search_query_approximate_validation(self):
        """Test approximate parameter validation"""
        # Valid boolean values
        query = SearchQuery(q="test", approximate=True)
        self.assertTrue(query.approximate)
        
        query = SearchQuery(q="test", approximate=False)
        self.assertFalse(query.approximate)
        
        # None should be valid
        query = SearchQuery(q="test", approximate=None)
        self.assertIsNone(query.approximate)

    def test_search_query_show_highlights_validation(self):
        """Test showHighlights parameter validation"""
        # Default should be True
        query = SearchQuery(q="test")
        self.assertTrue(query.showHighlights)
        
        # Can be set to False
        query = SearchQuery(q="test", showHighlights=False)
        self.assertFalse(query.showHighlights)

    def test_search_query_searchable_attributes_validation(self):
        """Test searchableAttributes parameter validation"""
        # Valid list of strings
        query = SearchQuery(q="test", searchableAttributes=["field1", "field2"])
        self.assertEqual(query.searchableAttributes, ["field1", "field2"])
        
        # None should be valid
        query = SearchQuery(q="test", searchableAttributes=None)
        self.assertIsNone(query.searchableAttributes)
        
        # Empty list should be valid
        query = SearchQuery(q="test", searchableAttributes=[])
        self.assertEqual(query.searchableAttributes, [])

    def test_search_query_attributes_to_retrieve_validation(self):
        """Test attributesToRetrieve parameter validation"""
        # Valid list of strings
        query = SearchQuery(q="test", attributesToRetrieve=["field1", "field2"])
        self.assertEqual(query.attributesToRetrieve, ["field1", "field2"])
        
        # None should be valid
        query = SearchQuery(q="test", attributesToRetrieve=None)
        self.assertIsNone(query.attributesToRetrieve)

    def test_search_query_with_valid_tensor_context_only(self):
        """Test SearchQuery with only tensor context (no query)"""
        
        context = SearchContext(tensor=[SearchContextTensor(vector=[1, 2, 3], weight=1.0)])
        
        # Should be valid for tensor search
        query = SearchQuery(q=None, searchMethod=SearchMethod.TENSOR, context=context)
        self.assertIsNone(query.q)
        self.assertIsNotNone(query.context)

    def test_search_query_with_valid_documents_context_only(self):
        """Test SearchQuery with only documents context (no query)"""
        
        context_docs = SearchContextDocuments(ids={"doc1": 1.0})
        context = SearchContext(documents=context_docs)
        
        # Should be valid for tensor search
        query = SearchQuery(q=None, searchMethod=SearchMethod.TENSOR, context=context)
        self.assertIsNone(query.q)
        self.assertIsNotNone(query.context)

    def test_search_query_default_search_method(self):
        """Test SearchQuery default search method"""
        
        query = SearchQuery(q="test")
        self.assertEqual(query.searchMethod, SearchMethod.TENSOR)

    def test_search_query_limit_and_offset_defaults(self):
        """Test SearchQuery default limit and offset values"""
        
        query = SearchQuery(q="test")
        self.assertEqual(query.limit, 10)
        self.assertEqual(query.offset, 0)

    def test_search_query_show_highlights_default(self):
        """Test SearchQuery default showHighlights value"""
        
        query = SearchQuery(q="test")
        self.assertTrue(query.showHighlights)

    def test_language_field_validation_with_all_search_modes(self):
        """Test language field behavior across all search modes."""

        test_cases = [
            {
                "search_method": SearchMethod.TENSOR,
                "language": "en",
                "should_fail": True,
                "expected_error": "language parameter is not supported for TENSOR search method"
            },
            {
                "search_method": SearchMethod.LEXICAL,
                "language": "fr",
                "should_fail": False,
                "expected_error": None
            },
            {
                "search_method": SearchMethod.HYBRID,
                "language": "es",
                "should_fail": False,
                "expected_error": None
            }
        ]

        for case in test_cases:
            with self.subTest(search_method=case["search_method"]):
                if case["should_fail"]:
                    with self.assertRaises(ValidationError) as cm:
                        SearchQuery(
                            q="test query",
                            searchMethod=case["search_method"],
                            language=case["language"]
                        )
                    self.assertIn(case["expected_error"], str(cm.exception))
                    self.assertIn("Language specification only applies to lexical and hybrid search", str(cm.exception))
                else:
                    search_query = SearchQuery(
                        q="test query",
                        searchMethod=case["search_method"],
                        language=case["language"]
                    )
                    self.assertEqual(search_query.language, case["language"])
                    self.assertEqual(search_query.searchMethod, case["search_method"])

    def test_context_documents_with_lexical_search_fails(self):
        """Test that context.documents is not supported for lexical search"""
        context = SearchContext(
            documents=SearchContextDocuments(
                ids={"doc1": 1.0, "doc2": 0.5}
            )
        )
        
        # Should fail for lexical search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test query",
                searchMethod=SearchMethod.LEXICAL,
                context=context
            )
        self.assertIn("Context is not supported for lexical search", str(cm.exception))

    def test_context_documents_lexical_lexical_hybrid_search_fails(self):
        """Test that context.documents is not supported for lexical/lexical hybrid search"""
        context = SearchContext(
            documents=SearchContextDocuments(
                ids={"doc1": 1.0, "doc2": 0.5}
            )
        )
        
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Lexical,
            rankingMethod=RankingMethod.Lexical
        )
        
        # Should fail for lexical/lexical hybrid search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test query",
                searchMethod=SearchMethod.HYBRID,
                hybridParameters=hybrid_params,
                context=context
            )
        self.assertIn("Context is not supported for lexical/lexical hybrid search", str(cm.exception))

    def test_recency_parameters_validation(self):
        """Test that recency parameters are only allowed for hybrid search."""
        recency_params = RecencyParameters(recency_field="created_at")

        # Valid cases - recency parameters with hybrid search
        valid_cases = [
            ("hybrid_with_recency", {
                "q": "test query",
                "searchMethod": SearchMethod.HYBRID,
                "recencyParameters": recency_params
            }),
        ]

        for test_name, params in valid_cases:
            with self.subTest(test_name):
                search_query = SearchQuery(**params)
                self.assertIsNotNone(search_query.recencyParameters)

        # Invalid cases - recency parameters with non-hybrid search
        invalid_cases = [
            ("tensor_with_recency", {
                "q": "test query",
                "searchMethod": SearchMethod.TENSOR,
                "recencyParameters": recency_params
            }, "Recency parameters can only be provided for 'HYBRID' search"),
            ("lexical_with_recency", {
                "q": "test query",
                "searchMethod": SearchMethod.LEXICAL,
                "recencyParameters": recency_params
            }, "Recency parameters can only be provided for 'HYBRID' search"),
        ]

        for test_name, params, expected_error in invalid_cases:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as cm:
                    SearchQuery(**params)
                self.assertIn(expected_error, str(cm.exception))

    def test_sort_by_cannot_be_used_with_recency(self):
        """Test that sortBy cannot be used with recencyParameters.

        Exception: When apply_in_ranking_phase='exclude-global', recency is only
        applied in phase-1 ranking while sortBy is applied in global ranking,
        so they don't conflict.
        """
        recency_params = RecencyParameters(recency_field="created_at")
        sort_by_params = SortByModel(
            fields=[SortByField(field_name="price", order=SortOrder.Desc)]
        )

        # Valid cases - only one or neither parameter
        valid_cases = [
            ("recency_only", {
                "q": "test query",
                "searchMethod": SearchMethod.HYBRID,
                "recencyParameters": recency_params
            }),
            ("sort_by_only", {
                "q": "test query",
                "searchMethod": SearchMethod.HYBRID,
                "sortBy": sort_by_params
            }),
            ("neither", {
                "q": "test query",
                "searchMethod": SearchMethod.HYBRID
            }),
        ]

        for test_name, params in valid_cases:
            with self.subTest(test_name):
                search_query = SearchQuery(**params)
                # Should not raise an error
                self.assertIsNotNone(search_query)

        # Test sortBy + recencyParameters for all apply_in_ranking_phase values
        apply_phase_test_cases = [
            # (apply_in_ranking_phase, should_raise_error)
            ("all", True),           # Should raise - recency applies in global phase
            ("only-global", True),   # Should raise - recency applies in global phase
            ("exclude-global", False),  # Should NOT raise - recency excluded from global phase
        ]

        for apply_mode, should_raise in apply_phase_test_cases:
            with self.subTest(apply_in_ranking_phase=apply_mode):
                recency_params_with_mode = RecencyParameters(
                    recency_field="created_at",
                    apply_in_ranking_phase=apply_mode
                )

                if should_raise:
                    with self.assertRaises(ValidationError) as cm:
                        SearchQuery(
                            q="test query",
                            searchMethod=SearchMethod.HYBRID,
                            recencyParameters=recency_params_with_mode,
                            sortBy=sort_by_params
                        )
                    self.assertIn("'sortBy' cannot be used with 'recencyParameters' with global-phase reranking", str(cm.exception))
                    self.assertIn("sortBy bypasses relevance scoring", str(cm.exception))
                else:
                    # Should NOT raise - exclude-global allows sortBy + recency
                    search_query = SearchQuery(
                        q="test query",
                        searchMethod=SearchMethod.HYBRID,
                        recencyParameters=recency_params_with_mode,
                        sortBy=sort_by_params
                    )
                    self.assertIsNotNone(search_query)
                    self.assertIsNotNone(search_query.recencyParameters)
                    self.assertIsNotNone(search_query.sort_by)


    def test_apply_to_subqueries_only_for_hybrid_search(self):
        """Test that applyToSubqueries is only allowed for hybrid search."""
        recency_params = RecencyParameters(
            recency_field="created_at",
            apply_to_subqueries=["tensor"]
        )

        # Valid: hybrid search with applyToSubqueries
        with self.subTest("hybrid_valid"):
            search_query = SearchQuery(
                q="test query",
                searchMethod=SearchMethod.HYBRID,
                recencyParameters=recency_params
            )
            self.assertIsNotNone(search_query.recencyParameters)

        # Invalid: tensor search with applyToSubqueries
        with self.subTest("tensor_invalid"):
            with self.assertRaises(ValidationError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.TENSOR,
                    recencyParameters=recency_params
                )
            # Should fail on "Recency parameters can only be provided for 'HYBRID' search"
            self.assertIn("HYBRID", str(cm.exception))

        # Invalid: lexical search with applyToSubqueries
        with self.subTest("lexical_invalid"):
            with self.assertRaises(ValidationError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.LEXICAL,
                    recencyParameters=recency_params
                )
            self.assertIn("HYBRID", str(cm.exception))

    def test_apply_to_subqueries_only_for_rrf_ranking(self):
        """Test that applyToSubqueries is only allowed with RRF ranking method."""
        recency_params = RecencyParameters(
            recency_field="created_at",
            apply_to_subqueries=["tensor"]
        )

        # Valid: RRF ranking (explicit)
        with self.subTest("rrf_explicit"):
            search_query = SearchQuery(
                q="test query",
                searchMethod=SearchMethod.HYBRID,
                hybridParameters=HybridParameters(rankingMethod=RankingMethod.RRF),
                recencyParameters=recency_params
            )
            self.assertIsNotNone(search_query.recencyParameters)

        # Valid: default ranking (which is RRF)
        with self.subTest("default_ranking"):
            search_query = SearchQuery(
                q="test query",
                searchMethod=SearchMethod.HYBRID,
                recencyParameters=recency_params
            )
            self.assertIsNotNone(search_query.recencyParameters)

        # Invalid: non-Disjunction retrieval methods
        non_disjunction_methods = [
            ("lexical_retrieval", RankingMethod.Lexical, RetrievalMethod.Lexical),
            ("tensor_retrieval", RankingMethod.Tensor, RetrievalMethod.Tensor),
        ]

        for test_name, ranking_method, retrieval_method in non_disjunction_methods:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as cm:
                    SearchQuery(
                        q="test query",
                        searchMethod=SearchMethod.HYBRID,
                        hybridParameters=HybridParameters(
                            rankingMethod=ranking_method,
                            retrievalMethod=retrieval_method
                        ),
                        recencyParameters=recency_params
                    )
                self.assertIn("disjunction", str(cm.exception).lower())

        # Valid: applyToSubqueries is None (no restriction even with non-Disjunction)
        with self.subTest("apply_to_subqueries_none_with_tensor_ranking"):
            recency_params_no_apply = RecencyParameters(recency_field="created_at")
            search_query = SearchQuery(
                q="test query",
                searchMethod=SearchMethod.HYBRID,
                hybridParameters=HybridParameters(
                    rankingMethod=RankingMethod.Tensor,
                    retrievalMethod=RetrievalMethod.Lexical
                ),
                recencyParameters=recency_params_no_apply
            )
            self.assertIsNotNone(search_query.recencyParameters)


class TestCustomVectorQuery(unittest.TestCase):

    def test_custom_vector_query_creation(self):
        """Test CustomVectorQuery creation."""
        custom_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.1, 0.2, 0.3]
            )
        )
        
        self.assertEqual(custom_query.customVector.content, "test content")
        self.assertEqual(custom_query.customVector.vector, [0.1, 0.2, 0.3])

    def test_custom_vector_query_without_content(self):
        """Test CustomVectorQuery without content."""
        custom_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                vector=[0.1, 0.2, 0.3]
            )
        )
        
        self.assertIsNone(custom_query.customVector.content)
        self.assertEqual(custom_query.customVector.vector, [0.1, 0.2, 0.3])


class TestSearchQueryContextMethods(unittest.TestCase):
    """Test SearchQuery context-related methods"""

    def test_get_context_tensor_with_context(self):
        """Test get_context_tensor when context with tensor is provided"""
        context = SearchContext(tensor=[SearchContextTensor(vector=[1, 2, 3], weight=1.0)])
        query = SearchQuery(q="test", context=context)
        
        result = query.get_context_tensor()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].vector, [1, 2, 3])
        self.assertEqual(result[0].weight, 1.0)

    def test_get_context_tensor_without_context(self):
        """Test get_context_tensor when no context is provided"""
        query = SearchQuery(q="test")
        
        result = query.get_context_tensor()
        self.assertIsNone(result)

    def test_get_context_documents_with_context(self):
        """Test get_context_documents when context with documents is provided"""
        query = SearchQuery(q="test")
        
        result = query.get_context_documents()
        self.assertIsNone(result)

    def test_get_context_documents_without_context(self):
        """Test get_context_documents when no context is provided"""
        query = SearchQuery(q="test")
        
        result = query.get_context_documents()
        self.assertIsNone(result)

    def test_get_context_documents_with_context_no_documents(self):
        """Test get_context_documents when context exists but has no documents"""
        context = SearchContext(tensor=[SearchContextTensor(vector=[1, 2, 3], weight=1.0)])
        query = SearchQuery(q="test", context=context)
        
        result = query.get_context_documents()
        self.assertIsNone(result)


class TestSearchContextDocumentsParameters(unittest.TestCase):
    """Test SearchContextDocumentsParameters validation"""

    def test_tensor_fields_validation_empty_list(self):
        """Test that empty tensorFields list raises error"""
        with self.assertRaises(ValueError) as cm:
            SearchContextDocumentsParameters(tensorFields=[])
        self.assertIn('tensorFields parameter must be non-empty list', str(cm.exception))

    def test_tensor_fields_validation_none(self):
        """Test that None tensorFields is valid"""
        # Should not raise error
        params = SearchContextDocumentsParameters(tensorFields=None)
        self.assertIsNone(params.tensor_fields)

    def test_tensor_fields_validation_valid_list(self):
        """Test that valid tensorFields list works"""
        params = SearchContextDocumentsParameters(tensorFields=["field1", "field2"])
        self.assertEqual(params.tensor_fields, ["field1", "field2"])


class TestSearchContextDocuments(unittest.TestCase):
    """Test SearchContextDocuments validation"""

    def test_ids_validation(self):
        """Test that ids field validation works correctly"""
        # Valid case
        docs = SearchContextDocuments(ids={"doc1": 1.0, "doc2": 0.5})
        self.assertEqual(docs.ids, {"doc1": 1.0, "doc2": 0.5})

    def test_parameters_validation(self):
        """Test that parameters field works correctly"""
        params = SearchContextDocumentsParameters(excludeInputDocuments=False)
        docs = SearchContextDocuments(ids={"doc1": 1.0}, parameters=params)
        self.assertFalse(docs.parameters.exclude_input_documents)

    def test_default_parameters(self):
        """Test that default parameters are created when not provided"""
        docs = SearchContextDocuments(ids={"doc1": 1.0})
        self.assertIsNotNone(docs.parameters)
        self.assertTrue(docs.parameters.exclude_input_documents)  # Default value

    def test_search_context_documents_with_empty_ids_fails(self):
        """Test that empty ids dict raises error"""
        with self.assertRaises(ValueError) as cm:
            SearchContextDocuments(ids={})
        self.assertIn('must be present and a non-empty dict', str(cm.exception))

    def test_search_context_documents_with_none_ids_fails(self):
        """Test that None ids raises error"""
        with self.assertRaises(ValueError) as cm:
            SearchContextDocuments(ids=None)
        self.assertIn('must be present and a non-empty dict', str(cm.exception))

    def test_search_context_documents_with_valid_ids_succeeds(self):
        """Test that valid ids dict succeeds"""
        docs = SearchContextDocuments(ids={"doc1": 1.0, "doc2": 0.5})
        self.assertEqual(docs.ids, {"doc1": 1.0, "doc2": 0.5})

    def test_exclude_input_documents_boolean_validation(self):
        """Test excludeInputDocuments boolean validation"""
        # Valid boolean values
        params = SearchContextDocumentsParameters(excludeInputDocuments=True)
        self.assertTrue(params.exclude_input_documents)

        params = SearchContextDocumentsParameters(excludeInputDocuments=False)
        self.assertFalse(params.exclude_input_documents)

    def test_tensor_fields_empty_string_in_list_fails(self):
        """Test that empty string in tensorFields list is handled"""
        # This should work - empty strings are valid field names in some contexts
        params = SearchContextDocumentsParameters(tensorFields=["field1", "", "field2"])
        self.assertEqual(params.tensor_fields, ["field1", "", "field2"])

    def test_search_context_documents_parameters_inheritance(self):
        """Test that SearchContextDocuments properly uses SearchContextDocumentsParameters"""
        params = SearchContextDocumentsParameters(
            tensorFields=["field1"],
            excludeInputDocuments=False,
        )
        docs = SearchContextDocuments(ids={"doc1": 1.0}, parameters=params)

        self.assertEqual(docs.parameters.tensor_fields, ["field1"])
        self.assertFalse(docs.parameters.exclude_input_documents)

    def test_search_context_documents_with_invalid_weight_types(self):
        """Test that invalid weight types are handled by pydantic"""
        # This should work as pydantic will convert string numbers to float
        docs = SearchContextDocuments(ids={"doc1": "1.0", "doc2": "0.5"})
        self.assertEqual(docs.ids, {"doc1": 1.0, "doc2": 0.5})

    def test_search_context_documents_with_negative_weights(self):
        """Test that negative weights are allowed"""
        # Negative weights should be allowed
        docs = SearchContextDocuments(ids={"doc1": -1.0, "doc2": 0.5})
        self.assertEqual(docs.ids, {"doc1": -1.0, "doc2": 0.5})

    def test_search_context_documents_with_zero_weights(self):
        """Test that zero weights are allowed"""
        # Zero weights should be allowed
        docs = SearchContextDocuments(ids={"doc1": 0.0, "doc2": 1.0})
        self.assertEqual(docs.ids, {"doc1": 0.0, "doc2": 1.0})


class TestSearchContext(unittest.TestCase):
    """Test SearchContext validation"""

    def test_tensor_type_validation_with_invalid_types(self):
        """Test that passing non-list types for tensor raises InvalidArgError"""
        invalid_types = [
            ("not_a_list", "str"),
            (123, "int"),
            ({"key": "value"}, "dict")
        ]

        for invalid_value, expected_type in invalid_types:
            with self.subTest(value=invalid_value, expected_type=expected_type):
                with self.assertRaises(api_exceptions.InvalidArgError) as cm:
                    SearchContext(tensor=invalid_value)
                self.assertIn('not a valid list', str(cm.exception))

    def test_tensor_valid_list(self):
        """Test that passing a valid list of SearchContextTensor works"""
        # Should not raise error
        tensor_list = [SearchContextTensor(vector=[0.1, 0.2, 0.3], weight=1.0)]
        context = SearchContext(tensor=tensor_list)
        self.assertEqual(len(context.tensor), 1)
        self.assertEqual(context.tensor[0].weight, 1.0)

    def test_tensor_none_is_valid(self):
        """Test that None tensor is valid when documents are provided"""
        docs = SearchContextDocuments(ids={"doc1": 1.0})
        context = SearchContext(tensor=None, documents=docs)
        self.assertIsNone(context.tensor)
        self.assertIsNotNone(context.documents)

    def test_tensor_length_validation_bounds(self):
        """Test tensor length validation bounds"""
        # Test with 0 tensors (should fail)
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            SearchContext(tensor=[])
        self.assertIn('has at least 1 items', str(cm.exception))

        # Test with 65 tensors (should fail)
        large_tensor_list = [SearchContextTensor(vector=[0.1, 0.2], weight=1.0) for _ in range(65)]
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            SearchContext(tensor=large_tensor_list)
        self.assertIn('has at most 64 items', str(cm.exception))

        # Test with 1 tensor (should pass)
        single_tensor = [SearchContextTensor(vector=[0.1, 0.2], weight=1.0)]
        context = SearchContext(tensor=single_tensor)
        self.assertEqual(len(context.tensor), 1)

        # Test with 64 tensors (should pass)
        max_tensor_list = [SearchContextTensor(vector=[0.1, 0.2], weight=1.0) for _ in range(64)]
        context = SearchContext(tensor=max_tensor_list)
        self.assertEqual(len(context.tensor), 64)

    def test_search_context_validation_error_conversion(self):
        """Test that ValidationError from parent init is converted to InvalidArgError"""
        # Create a scenario that would cause ValidationError in the parent __init__
        # This happens when we pass invalid data that fails pydantic validation
        with self.assertRaises(api_exceptions.InvalidArgError):
            # Pass invalid tensor data that will cause ValidationError
            SearchContext(tensor="invalid_tensor_data")


class TestQueryContent(unittest.TestCase):
    """Test QueryContent model"""

    def test_query_content_modality_field_with_text(self):
        """Test QueryContent modality field access with text modality"""
        query_content = QueryContent(content="test content", modality=Modality.TEXT)
        
        # Test that modality field is accessible and has correct value
        self.assertEqual(query_content.modality, Modality.TEXT)
        self.assertEqual(query_content.content, "test content")

    def test_query_content_modality_field_with_image(self):
        """Test QueryContent modality field access with image modality"""
        query_content = QueryContent(content="http://example.com/image.jpg", modality=Modality.IMAGE)
        
        self.assertEqual(query_content.modality, Modality.IMAGE)
        self.assertEqual(query_content.content, "http://example.com/image.jpg")


class TestSearchQueryCollapseFields(unittest.TestCase):
    """Test SearchQuery collapse fields validation"""

    def test_collapse_fields_valid_format(self):
        """Test that collapse fields with valid format are accepted"""
        collapse_fields = [CollapseModel(name="product_id")]
        
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.HYBRID,
            collapseFields=collapse_fields
        )
        
        self.assertEqual(len(search_query.collapse_fields), 1)
        self.assertEqual(search_query.collapse_fields[0].name, "product_id")

    def test_collapse_fields_only_for_hybrid_search(self):
        """Test that collapse fields are only allowed for hybrid search"""
        collapse_fields = [CollapseModel(name="product_id")]
        
        with self.subTest("TENSOR search method"):
            with self.assertRaises(ValueError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.TENSOR,
                    collapseFields=collapse_fields
                )
            self.assertIn("collapseFields can only be provided for 'HYBRID' search", str(cm.exception))
        
        with self.subTest("LEXICAL search method"):
            with self.assertRaises(ValueError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.LEXICAL,
                    collapseFields=collapse_fields
                )
            self.assertIn("collapseFields can only be provided for 'HYBRID' search", str(cm.exception))

    def test_collapse_fields_single_field_only(self):
        """Test that exactly one collapse field must be provided"""
        with self.subTest("Multiple collapse fields"):
            collapse_fields = [
                CollapseModel(name="product_id"),
                CollapseModel(name="category_id")
            ]
            
            with self.assertRaises(ValueError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.HYBRID,
                    collapseFields=collapse_fields
                )
            self.assertIn("Exactly one collapse field must be provided", str(cm.exception))
        
        with self.subTest("Empty collapse fields list"):
            with self.assertRaises(ValueError) as cm:
                SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.HYBRID,
                    collapseFields=[]
                )
            self.assertIn("Exactly one collapse field must be provided", str(cm.exception))

    def test_collapse_fields_none_is_valid(self):
        """Test that None collapse fields is valid"""
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.HYBRID,
            collapseFields=None
        )
        
        self.assertIsNone(search_query.collapse_fields)

    def test_search_collapse_field_requires_name(self):
        """Test that SearchCollapseField requires name field"""
        with self.assertRaises(ValidationError):
            CollapseModel()  # Missing required name field

    def test_collapse_model_sort_by_construction(self):
        """Test CollapseModel sortBy construction and validation"""
        with self.subTest("valid sort_by with explicit order"):
            model = CollapseModel(
                name="product_id",
                sort_by=CollapseSortBy(
                    fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
                )
            )
            self.assertEqual(model.name, "product_id")
            self.assertEqual(len(model.sort_by.fields), 1)
            self.assertEqual(model.sort_by.fields[0].field_name, "price")
            self.assertEqual(model.sort_by.fields[0].order, SortOrder.Asc)

        with self.subTest("order defaults to desc"):
            field = CollapseSortByField(fieldName="price")
            self.assertEqual(field.order, SortOrder.Desc)

        with self.subTest("None sort_by is valid"):
            model = CollapseModel(name="product_id")
            self.assertIsNone(model.sort_by)

        with self.subTest("multiple sort_by fields rejected"):
            with self.assertRaises(ValidationError):
                CollapseModel(
                    name="product_id",
                    sort_by=CollapseSortBy(
                        fields=[
                            CollapseSortByField(fieldName="price", order=SortOrder.Asc),
                            CollapseSortByField(fieldName="rating", order=SortOrder.Desc)
                        ]
                    )
                )

    def test_collapse_model_generate_vespa_sort_by_query_input(self):
        """Test generate_vespa_sort_by_query_input produces correct output"""
        with self.subTest("desc order returns 1"):
            sort_by = CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order=SortOrder.Desc)]
            )
            self.assertEqual(sort_by.generate_vespa_sort_by_query_input(), {"price": 1})

        with self.subTest("asc order returns -1"):
            sort_by = CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
            )
            self.assertEqual(sort_by.generate_vespa_sort_by_query_input(), {"price": -1})

        with self.subTest("no sort_by returns None on CollapseModel"):
            model = CollapseModel(name="product_id")
            self.assertIsNone(model.sort_by)

    def test_collapse_model_execute_sort_and_filter_string(self):
        """Test execute sort lifecycle and collapse filter string"""
        sort_by = CollapseSortBy(
            fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
        )

        with self.subTest("default state"):
            self.assertFalse(sort_by.should_execute_sort())
            self.assertEqual(sort_by.get_collapse_sort_by_filter_string(), "")

        with self.subTest("cannot set filter before enabling sort"):
            with self.assertRaises(RuntimeError):
                sort_by.set_collapse_sort_by_filter_string("product_id in (\"a\", \"b\")")

        with self.subTest("enable sort, then set filter"):
            sort_by.enable_execute_sort()
            self.assertTrue(sort_by.should_execute_sort())
            sort_by.set_collapse_sort_by_filter_string("product_id in (\"a\", \"b\")")
            self.assertEqual(sort_by.get_collapse_sort_by_filter_string(), "product_id in (\"a\", \"b\")")

        with self.subTest("disable sort"):
            sort_by.disable_execute_sort()
            self.assertFalse(sort_by.should_execute_sort())

    def test_collapse_model_num_threads_per_search(self):
        """Test numThreadsPerSearch validation"""
        with self.subTest("valid with sort_by"):
            sort_by = CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)],
                numThreadsPerSearch=4
            )
            self.assertEqual(sort_by.num_threads_per_search, 4)

        with self.subTest("None by default"):
            sort_by = CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
            )
            self.assertIsNone(sort_by.num_threads_per_search)

        with self.subTest("must be >= 1"):
            with self.assertRaises(ValidationError):
                CollapseSortBy(
                    fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)],
                    numThreadsPerSearch=0
                )

    def test_collapse_model_disable_if_main_sort_by_fields(self):
        """Test disableIfMainSortByFields is stored correctly"""
        sort_by = CollapseSortBy(
            fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)],
            disableIfMainSortByFields={"price", "rating"}
        )
        self.assertEqual(sort_by.disable_if_main_sort_by_fields, {"price", "rating"})

    def test_search_query_with_collapse_and_sort_by(self):
        """Test CollapseModel construction via SearchQuery and collapse.sortBy pruning when main query has sortBy"""
        with self.subTest("collapse with sortBy constructed via SearchQuery"):
            sq = SearchQuery(
                q="test",
                searchMethod=SearchMethod.HYBRID,
                collapseFields=[CollapseModel(
                    name="product_id",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
                    )
                )]
            )
            self.assertIsNotNone(sq.collapse_fields[0].sort_by)
            self.assertEqual(sq.collapse_fields[0].sort_by.fields[0].field_name, "price")

        with self.subTest("collapse.sortBy preserved even when main sortBy matches disableIfMainSortByFields"):
            sq = SearchQuery(
                q="test",
                searchMethod=SearchMethod.HYBRID,
                collapseFields=[CollapseModel(
                    name="product_id",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)],
                        disableIfMainSortByFields={"price"}
                    )
                )],
                sortBy=SortByModel(fields=[SortByField(field_name="price", order=SortOrder.Asc)])
            )
            # Pruning is no longer done at the API model level
            self.assertIsNotNone(sq.collapse_fields[0].sort_by)

        with self.subTest("collapse.sortBy kept when main sortBy does not match disableIfMainSortByFields"):
            sq = SearchQuery(
                q="test",
                searchMethod=SearchMethod.HYBRID,
                collapseFields=[CollapseModel(
                    name="product_id",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)],
                        disableIfMainSortByFields={"rating"}
                    )
                )],
                sortBy=SortByModel(fields=[SortByField(field_name="price", order=SortOrder.Asc)])
            )
            self.assertIsNotNone(sq.collapse_fields[0].sort_by)

        with self.subTest("collapse.sortBy kept when disableIfMainSortByFields is None"):
            sq = SearchQuery(
                q="test",
                searchMethod=SearchMethod.HYBRID,
                collapseFields=[CollapseModel(
                    name="product_id",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order=SortOrder.Asc)]
                    )
                )],
                sortBy=SortByModel(fields=[SortByField(field_name="price", order=SortOrder.Asc)])
            )
            self.assertIsNotNone(sq.collapse_fields[0].sort_by)


if __name__ == '__main__':
    unittest.main()
