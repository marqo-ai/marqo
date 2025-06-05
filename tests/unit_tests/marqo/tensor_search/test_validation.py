import unittest
from unittest.mock import patch

from marqo.api.exceptions import InvalidArgError
from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import CustomVectorQuery


class TestValidateQuery(unittest.TestCase):
    """Unit tests for the validate_query function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_string_query = "test query"
        self.valid_dict_query = {"dogs": 1.0, "cats": 0.5}
        self.valid_custom_vector_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        )

    def test_validate_query_string_and_none_queries(self):
        """Test that string and None queries are returned unchanged for all search methods."""
        test_cases = [
            ("string", self.valid_string_query),
            ("none", None)
        ]

        for query_type, query_value in test_cases:
            for search_method in [SearchMethod.TENSOR, SearchMethod.LEXICAL, SearchMethod.HYBRID]:
                with self.subTest(query_type=query_type, search_method=search_method):
                    result = validation.validate_query(query_value, search_method)
                    if query_value is None:
                        self.assertIsNone(result)
                    else:
                        self.assertEqual(result, query_value)

    def test_validate_query_custom_vector_by_search_method(self):
        """Test CustomVectorQuery validation for different search methods."""
        test_cases = [
            (SearchMethod.TENSOR, True, "should succeed for TENSOR"),
            (SearchMethod.HYBRID, True, "should succeed for HYBRID"),
            (SearchMethod.LEXICAL, False, "should fail for LEXICAL")
        ]

        for search_method, should_succeed, description in test_cases:
            with self.subTest(search_method=search_method, description=description):
                if should_succeed:
                    result = validation.validate_query(self.valid_custom_vector_query, search_method)
                    self.assertEqual(result, self.valid_custom_vector_query)
                else:
                    with self.assertRaises(InvalidArgError) as cm:
                        validation.validate_query(self.valid_custom_vector_query, search_method)
                    error_msg = str(cm.exception)
                    self.assertIn("Custom vector search is only supported", error_msg)
                    self.assertIn("search_method=\"HYBRID\"", error_msg)
                    self.assertIn("search_method=\"TENSOR\"", error_msg)

    def test_validate_query_dict_by_search_method(self):
        """Test dictionary query validation for different search methods."""
        test_cases = [
            (SearchMethod.TENSOR, True, "should succeed for TENSOR"),
            (SearchMethod.LEXICAL, False, "Multi-term query is not supported for search_method=\"LEXICAL\""),
            (SearchMethod.HYBRID, False, "To use multi-term query with search_method=\"HYBRID\"")
        ]

        for search_method, should_succeed, expected_error_fragment in test_cases:
            with self.subTest(search_method=search_method):
                if should_succeed:
                    result = validation.validate_query(self.valid_dict_query, search_method)
                    self.assertEqual(result, self.valid_dict_query)
                else:
                    with self.assertRaises(InvalidArgError) as cm:
                        validation.validate_query(self.valid_dict_query, search_method)
                    error_msg = str(cm.exception)
                    self.assertIn(expected_error_fragment, error_msg)
                    if search_method == SearchMethod.HYBRID:
                        self.assertIn("hybrid_parameters.queryTensor", error_msg)

    def test_validate_query_dict_validation_rules(self):
        """Test dictionary validation rules for structure and content."""
        test_cases = [
            # (query, should_succeed, expected_error_fragment, description)
            ({}, False, "Multi-term query requires at least one query", "empty dict"),
            ({123: 1.0, "cats": 0.5}, False, "Found key of type `<class 'int'>` instead of string", "invalid key type"),
            ({"dogs": "not_a_number", "cats": 0.5}, False, "Found value of type `<class 'str'>` instead of float",
             "invalid value type"),
            ({"dogs": 1, "cats": 2}, True, None, "valid int values"),
            ({"dogs": 1.5, "cats": 2.7}, True, None, "valid float values"),
            ({"dogs": 1, "cats": 2.5, "birds": 0}, True, None, "mixed numeric values"),
            ({"dogs": 0.0, "cats": -1.5, "birds": float('inf')}, True, None, "special float values"),
            ({"dogs": -1, "cats": -2.5}, True, None, "negative values"),
            ({"query": 1.0}, True, None, "single item dictionary")
        ]

        for query, should_succeed, expected_error_fragment, description in test_cases:
            with self.subTest(description=description):
                if should_succeed:
                    result = validation.validate_query(query, SearchMethod.TENSOR)
                    self.assertEqual(result, query)
                else:
                    with self.assertRaises(InvalidArgError) as cm:
                        validation.validate_query(query, SearchMethod.TENSOR)
                    error_msg = str(cm.exception)
                    self.assertIn(expected_error_fragment, error_msg)

    def test_validate_query_invalid_types(self):
        """Test that queries with invalid types fail."""
        invalid_queries = [
            (123, "int"),
            (123.45, "float"),
            ([1, 2, 3], "list"),
            ({"a", "b"}, "set"),
            (object(), "arbitrary object")
        ]

        for invalid_query, description in invalid_queries:
            with self.subTest(query_type=description):
                with self.assertRaises(InvalidArgError) as cm:
                    validation.validate_query(invalid_query, SearchMethod.TENSOR)

                error_msg = str(cm.exception)
                self.assertIn("'q' must be a 'string', a 'dict', or 'None'", error_msg)
                self.assertIn(f"Received q of type `{type(invalid_query)}`", error_msg)

    def test_validate_query_case_insensitive_search_methods(self):
        """Test that search method validation is case insensitive for all query types."""
        search_method_cases = [
            ("tensor", SearchMethod.TENSOR),
            ("TENSOR", SearchMethod.TENSOR),
            ("Tensor", SearchMethod.TENSOR),
            ("lexical", SearchMethod.LEXICAL),
            ("LEXICAL", SearchMethod.LEXICAL),
            ("Lexical", SearchMethod.LEXICAL),
            ("hybrid", SearchMethod.HYBRID),
            ("HYBRID", SearchMethod.HYBRID),
            ("Hybrid", SearchMethod.HYBRID),
        ]

        query_cases = [
            ("string", "test", True, True, True),  # valid for all methods
            ("custom_vector", self.valid_custom_vector_query, True, False, True),  # valid for tensor/hybrid only
            ("dict", self.valid_dict_query, True, False, False)  # valid for tensor only
        ]

        for search_method_str, search_method_enum in search_method_cases:
            for query_type, query, valid_tensor, valid_lexical, valid_hybrid in query_cases:
                should_succeed = (
                        (search_method_enum == SearchMethod.TENSOR and valid_tensor) or
                        (search_method_enum == SearchMethod.LEXICAL and valid_lexical) or
                        (search_method_enum == SearchMethod.HYBRID and valid_hybrid)
                )

                with self.subTest(search_method=search_method_str, query_type=query_type):
                    if should_succeed:
                        # Test string version
                        result = validation.validate_query(query, search_method_str)
                        self.assertEqual(result, query)

                        # Test enum version for comparison
                        result_enum = validation.validate_query(query, search_method_enum)
                        self.assertEqual(result, result_enum)
                    else:
                        with self.assertRaises(InvalidArgError):
                            validation.validate_query(query, search_method_str)


if __name__ == '__main__':
    unittest.main()
