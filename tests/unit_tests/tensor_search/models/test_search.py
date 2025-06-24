import unittest
from marqo.tensor_search.models.search import SearchContextDocumentsParameters, SearchContextDocuments, SearchContext, SearchContextTensor
from marqo.api.exceptions import InvalidArgError
from pydantic.v1 import ValidationError


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

    # Error scenario tests
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
                with self.assertRaises(InvalidArgError) as cm:
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
        with self.assertRaises(InvalidArgError) as cm:
            SearchContext(tensor=[])
        self.assertIn('has at least 1 items', str(cm.exception))
        
        # Test with 65 tensors (should fail)
        large_tensor_list = [SearchContextTensor(vector=[0.1, 0.2], weight=1.0) for _ in range(65)]
        with self.assertRaises(InvalidArgError) as cm:
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
        with self.assertRaises(InvalidArgError):
            # Pass invalid tensor data that will cause ValidationError
            SearchContext(tensor="invalid_tensor_data")


if __name__ == '__main__':
    unittest.main() 