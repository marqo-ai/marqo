from marqo.core.models.marqo_index import CollapseField
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument
from tests.unit_tests.marqo_test import MarqoTestCase


class TestSemiStructuredVespaDocument(MarqoTestCase):
    """Unit tests for SemiStructuredVespaDocument class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collapse_field = CollapseField(name="product_id", minGroups=100)
        
    def test_from_vespa_document_with_collapse_field(self):
        """Test that from_vespa_document correctly handles collapse fields by adding them to text_fields."""
        # Create a semi-structured index with collapse fields
        marqo_index = self.semi_structured_marqo_index(
            name='test_index',
            collapse_fields=[self.collapse_field],
            lexical_field_names=['title']
        )
        
        # Mock Vespa document with collapse field
        vespa_doc = {
            "id": "test_doc_id",
            "fields": {
                "product_id": "group_123",
                "marqo__lexical_title": "Test Title",
                "marqo__id": "test_doc_id"
            }
        }
        
        # Convert from Vespa document
        semi_structured_doc = SemiStructuredVespaDocument.from_vespa_document(vespa_doc, marqo_index)
        
        # Assert collapse field is added to text_fields
        self.assertIn("product_id", semi_structured_doc.text_fields)
        self.assertEqual("group_123", semi_structured_doc.text_fields["product_id"])
        
        # Assert regular lexical field is also handled correctly
        self.assertIn("title", semi_structured_doc.text_fields)
        self.assertEqual("Test Title", semi_structured_doc.text_fields["title"])

    def test_from_marqo_document_with_collapse_field(self):
        """Test that from_marqo_document correctly handles collapse fields without field_map validation."""
        # Create a semi-structured index with collapse fields
        marqo_index = self.semi_structured_marqo_index(
            name='test_index',
            collapse_fields=[self.collapse_field],
            lexical_field_names=['title']
        )
        
        # Create Marqo document with collapse field
        marqo_doc = {
            "_id": "test_doc_id",
            "product_id": "group_456",
            "title": "Test Title Content"
        }
        
        # Convert from Marqo document
        semi_structured_doc = SemiStructuredVespaDocument.from_marqo_document(marqo_doc, marqo_index)
        
        # Assert collapse field was processed correctly and added to text_fields
        self.assertIn("product_id", semi_structured_doc.text_fields)
        self.assertEqual("group_456", semi_structured_doc.text_fields["product_id"])
        
        # Assert regular field was processed through normal field_map validation
        self.assertIn("marqo__lexical_title", semi_structured_doc.text_fields)
        self.assertEqual("Test Title Content", semi_structured_doc.text_fields["marqo__lexical_title"])

    def test_from_marqo_document_collapse_field_not_in_field_map(self):
        """Test that collapse fields don't need to be in field_map to be processed."""
        # Create a semi-structured index where collapse field is NOT in lexical_fields
        marqo_index = self.semi_structured_marqo_index(
            name='test_index',
            collapse_fields=[self.collapse_field],
            lexical_field_names=['title']  # collapse field not included here
        )
        
        # Create Marqo document with collapse field
        marqo_doc = {
            "_id": "test_doc_id",
            "product_id": "group_789"  # This field is not in field_map but is a collapse field
        }
        
        # Should not raise an error even though product_id is not in field_map
        semi_structured_doc = SemiStructuredVespaDocument.from_marqo_document(marqo_doc, marqo_index)
        
        # Assert collapse field was processed successfully
        self.assertIn("product_id", semi_structured_doc.text_fields)
        self.assertEqual("group_789", semi_structured_doc.text_fields["product_id"])

    def test_from_marqo_document_non_collapse_field_not_in_field_map_raises_error(self):
        """Test that non-collapse fields not in field_map raise MarqoDocumentParsingError."""
        from marqo.core.exceptions import MarqoDocumentParsingError
        
        # Create a semi-structured index with collapse fields
        marqo_index = self.semi_structured_marqo_index(
            name='test_index',
            collapse_fields=[self.collapse_field],
            lexical_field_names=['title']
        )
        
        # Create Marqo document with unknown field (not collapse, not in field_map)
        marqo_doc = {
            "_id": "test_doc_id",
            "unknown_field": "some_value"  # This field is not in field_map and not a collapse field
        }
        
        # Should raise MarqoDocumentParsingError
        with self.assertRaises(MarqoDocumentParsingError) as cm:
            SemiStructuredVespaDocument.from_marqo_document(marqo_doc, marqo_index)
        
        self.assertIn("Field unknown_field is not in index test_index", str(cm.exception))