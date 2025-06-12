import unittest
from unittest.mock import Mock, patch
from pydantic.v1 import ValidationError

from marqo.api.exceptions import InvalidArgError, BadRequestError
from marqo.tensor_search.enums import SearchMethod, MappingsObjectType
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search import validation
from marqo.tensor_search.models.mappings_object import text_field_language_mappings_schema
from marqo.core.models.marqo_index import Field, FieldType, FieldFeature, SemiStructuredMarqoIndex
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import SemiStructuredAddDocumentsHandler
from marqo.core.models.marqo_query import MarqoLexicalQuery, MarqoHybridQuery


class TestLanguageMappingValidation(unittest.TestCase):
    """Test language field mapping validation"""

    def test_validate_text_field_language_mappings_object_valid(self):
        """Test validation of valid text field language mapping"""
        valid_mapping = {
            "type": "text_field_language",
            "language": "es"
        }
        
        result = validation.validate_text_field_language_mappings_object(valid_mapping)
        self.assertEqual(result, valid_mapping)

    def test_validate_text_field_language_mappings_object_invalid_language_empty(self):
        """Test validation fails for empty language"""
        invalid_mapping = {
            "type": "text_field_language",
            "language": ""
        }
        
        with self.assertRaises(InvalidArgError) as cm:
            validation.validate_text_field_language_mappings_object(invalid_mapping)
        
        self.assertIn("Invalid language code", str(cm.exception))

    def test_validate_text_field_language_mappings_object_invalid_language_too_long(self):
        """Test validation fails for language code too long"""
        invalid_mapping = {
            "type": "text_field_language",
            "language": "spanish"
        }
        
        with self.assertRaises(InvalidArgError) as cm:
            validation.validate_text_field_language_mappings_object(invalid_mapping)
        
        self.assertIn("Invalid language code", str(cm.exception))

    def test_validate_text_field_language_mappings_object_invalid_language_uppercase(self):
        """Test validation fails for uppercase language code"""
        invalid_mapping = {
            "type": "text_field_language",
            "language": "ES"
        }
        
        with self.assertRaises(InvalidArgError) as cm:
            validation.validate_text_field_language_mappings_object(invalid_mapping)
        
        self.assertIn("Invalid language code", str(cm.exception))

    def test_validate_text_field_language_mappings_object_invalid_language_numeric(self):
        """Test validation fails for numeric language code"""
        invalid_mapping = {
            "type": "text_field_language",
            "language": "e5"
        }
        
        with self.assertRaises(InvalidArgError) as cm:
            validation.validate_text_field_language_mappings_object(invalid_mapping)
        
        self.assertIn("Invalid language code", str(cm.exception))

    def test_validate_text_field_language_mappings_object_valid_three_char(self):
        """Test validation succeeds for 3-character language codes"""
        valid_mapping = {
            "type": "text_field_language",
            "language": "spa"
        }
        
        result = validation.validate_text_field_language_mappings_object(valid_mapping)
        self.assertEqual(result, valid_mapping)

    def test_validate_text_field_language_mappings_object_missing_language(self):
        """Test validation fails when language is missing"""
        invalid_mapping = {
            "type": "text_field_language"
        }
        
        with self.assertRaises(InvalidArgError):
            validation.validate_text_field_language_mappings_object(invalid_mapping)


class TestSearchQueryLanguageValidation(unittest.TestCase):
    """Test search query model language validation"""

    def test_search_query_model_language_lexical_valid(self):
        """Test that model.language is allowed for lexical search"""
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.LEXICAL,
            model={"language": "es"}
        )
        
        self.assertEqual(search_query.model["language"], "es")
        self.assertEqual(search_query.searchMethod, SearchMethod.LEXICAL)

    def test_search_query_model_language_hybrid_valid(self):
        """Test that model.language is allowed for hybrid search"""
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.HYBRID,
            model={"language": "fr"}
        )
        
        self.assertEqual(search_query.model["language"], "fr")
        self.assertEqual(search_query.searchMethod, SearchMethod.HYBRID)

    def test_search_query_model_language_tensor_invalid(self):
        """Test that model.language is not allowed for tensor search"""
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test query",
                searchMethod=SearchMethod.TENSOR,
                model={"language": "es"}
            )
        
        self.assertIn("model.language parameter is not supported for TENSOR search method", str(cm.exception))

    def test_search_query_model_language_none_valid(self):
        """Test that model parameter without language is valid"""
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.TENSOR,
            model={"some_other_param": "value"}
        )
        
        self.assertEqual(search_query.model["some_other_param"], "value")

    def test_search_query_no_model_valid(self):
        """Test that search query without model parameter is valid"""
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.TENSOR
        )
        
        self.assertIsNone(search_query.model)


class TestFieldModelLanguageProperty(unittest.TestCase):
    """Test Field model language property"""

    def test_field_with_language_property(self):
        """Test creating Field with language property"""
        field = Field(
            name="title",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch],
            lexical_field_name="marqo__title",
            language="es"
        )
        
        self.assertEqual(field.language, "es")
        self.assertEqual(field.name, "title")
        self.assertEqual(field.type, FieldType.Text)

    def test_field_without_language_property(self):
        """Test creating Field without language property (defaults to None)"""
        field = Field(
            name="title",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch],
            lexical_field_name="marqo__title"
        )
        
        self.assertIsNone(field.language)


class TestMarqoQueryLanguageProperty(unittest.TestCase):
    """Test MarqoLexicalQuery and MarqoHybridQuery language property"""

    def test_marqo_lexical_query_with_language(self):
        """Test creating MarqoLexicalQuery with language"""
        query = MarqoLexicalQuery(
            index_name="test_index",
            or_phrases=["hello", "world"],
            and_phrases=["test"],
            limit=10,
            language="es"
        )
        
        self.assertEqual(query.language, "es")
        self.assertEqual(query.index_name, "test_index")

    def test_marqo_lexical_query_without_language(self):
        """Test creating MarqoLexicalQuery without language (defaults to None)"""
        query = MarqoLexicalQuery(
            index_name="test_index",
            or_phrases=["hello", "world"],
            and_phrases=["test"],
            limit=10
        )
        
        self.assertIsNone(query.language)

    def test_marqo_hybrid_query_inherits_language(self):
        """Test that MarqoHybridQuery inherits language property from MarqoLexicalQuery"""
        from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
        
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF
        )
        
        query = MarqoHybridQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            or_phrases=["hello"],
            and_phrases=["test"],
            limit=10,
            hybrid_parameters=hybrid_params,
            language="fr"
        )
        
        self.assertEqual(query.language, "fr")


class TestSemiStructuredAddDocumentHandlerLanguage(unittest.TestCase):
    """Test semi-structured add document handler language functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vespa_client = Mock()
        self.mock_index_management = Mock()
        self.mock_inference = Mock()
        
        # Create a mock semi-structured index
        self.mock_index = Mock(spec=SemiStructuredMarqoIndex)
        self.mock_index.name = "test_index"
        self.mock_index.field_map = {}
        self.mock_index.lexical_fields = []
        self.mock_index.clear_cache = Mock()

    def test_get_field_language_with_mapping(self):
        """Test extracting language from mappings"""
        add_docs_params = AddDocsParams(
            docs=[{"_id": "1", "title": "test"}],
            index_name="test_index",
            device="cpu",
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "es"
                }
            }
        )
        
        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=self.mock_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference
        )
        
        language = handler._get_field_language("title")
        self.assertEqual(language, "es")

    def test_get_field_language_without_mapping(self):
        """Test extracting language when no mapping exists"""
        add_docs_params = AddDocsParams(
            docs=[{"_id": "1", "title": "test"}],
            index_name="test_index",
            device="cpu",
            mappings={}
        )
        
        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=self.mock_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference
        )
        
        language = handler._get_field_language("title")
        self.assertIsNone(language)

    def test_get_field_language_wrong_mapping_type(self):
        """Test extracting language when mapping is different type"""
        add_docs_params = AddDocsParams(
            docs=[{"_id": "1", "title": "test"}],
            index_name="test_index",
            device="cpu",
            mappings={
                "title": {
                    "type": "custom_vector"
                }
            }
        )
        
        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=self.mock_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference
        )
        
        language = handler._get_field_language("title")
        self.assertIsNone(language)

    def test_validate_language_mapping_for_text_field_valid(self):
        """Test validation passes for text field with language mapping"""
        add_docs_params = AddDocsParams(
            docs=[{"_id": "1", "title": "test"}],
            index_name="test_index",
            device="cpu",
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "es"
                }
            }
        )
        
        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=self.mock_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference
        )
        
        # Should not raise exception
        handler._validate_language_mapping_for_field("title", "test content")

    def test_validate_language_mapping_for_non_text_field_invalid(self):
        """Test validation fails for non-text field with language mapping"""
        add_docs_params = AddDocsParams(
            docs=[{"_id": "1", "price": 10.5}],
            index_name="test_index",
            device="cpu",
            mappings={
                "price": {
                    "type": "text_field_language",
                    "language": "es"
                }
            }
        )
        
        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=self.mock_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference
        )
        
        with self.assertRaises(BadRequestError) as cm:
            handler._validate_language_mapping_for_field("price", 10.5)
        
        self.assertIn("Language mapping for field 'price' can only be used with text", str(cm.exception))


class TestMappingsValidationIntegration(unittest.TestCase):
    """Test integration of language mappings with existing validation"""

    def test_validate_mappings_object_with_language_field(self):
        """Test mappings validation includes language field"""
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "es"
            }
        }
        
        result = validation.validate_mappings_object(mappings)
        self.assertEqual(result, mappings)

    def test_validate_mappings_object_language_field_structured_index_error(self):
        """Test that language mappings are rejected for structured indexes"""
        from marqo.core.models.marqo_index import StructuredMarqoIndex
        
        mock_structured_index = Mock(spec=StructuredMarqoIndex)
        
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "es"
            }
        }
        
        with self.assertRaises(InvalidArgError) as cm:
            validation.validate_mappings_object(mappings, mock_structured_index)
        
        self.assertIn("Language field mapping 'title' cannot be used with structured indexes", str(cm.exception))

    def test_validate_mappings_object_language_field_invalid_schema(self):
        """Test mappings validation rejects invalid language field schema"""
        mappings = {
            "title": {
                "type": "text_field_language",
                "invalid_field": "value"
            }
        }
        
        with self.assertRaises(InvalidArgError):
            validation.validate_mappings_object(mappings)


if __name__ == '__main__':
    unittest.main()