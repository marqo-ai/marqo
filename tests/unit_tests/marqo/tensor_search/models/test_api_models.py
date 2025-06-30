import unittest
from pydantic.v1 import ValidationError

from marqo.api import exceptions as api_exceptions
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery
from marqo.tensor_search.models.search import (
    SearchContext, 
    SearchContextTensor, 
    SearchContextDocuments,
    SearchContextDocumentsParameters
)


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

    def test_concurrency_validation(self):
        """Test concurrency parameter validation"""
        # Valid positive integer
        params = SearchContextDocumentsParameters(concurrency=5)
        self.assertEqual(params.concurrency, 5)

        # None should be valid
        params = SearchContextDocumentsParameters(concurrency=None)
        self.assertIsNone(params.concurrency)

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
            concurrency=10
        )
        docs = SearchContextDocuments(ids={"doc1": 1.0}, parameters=params)

        self.assertEqual(docs.parameters.tensor_fields, ["field1"])
        self.assertFalse(docs.parameters.exclude_input_documents)
        self.assertEqual(docs.parameters.concurrency, 10)

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


if __name__ == '__main__':
    unittest.main()
