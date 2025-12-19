"""Unit tests for recency_score handling in SemiStructuredVespaDocument."""
import unittest
from unittest.mock import MagicMock

from marqo.core import constants
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.semi_structured_document import (
    SemiStructuredVespaDocument,
    SemiStructuredVespaDocumentFields
)


class TestSemiStructuredDocumentRecency(unittest.TestCase):
    """Tests for recency_score handling in SemiStructuredVespaDocument."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal mock index
        self.mock_index = MagicMock(spec=SemiStructuredMarqoIndex)
        self.mock_index.schema_name = "test_schema"
        self.mock_index.name = "test_index"
        self.mock_index.tensor_field_map = {}

    def test_recency_score_field_in_vespa_document_fields(self):
        """Test that recency_score field exists in SemiStructuredVespaDocumentFields."""
        # Test with None (default)
        fields = SemiStructuredVespaDocumentFields(marqo__id="doc1")
        self.assertIsNone(fields.recency_score)

        # Test with a value
        fields_with_score = SemiStructuredVespaDocumentFields(
            marqo__id="doc1",
            **{common.VESPA_DOC_RECENCY_SCORE: 0.85}
        )
        self.assertEqual(fields_with_score.recency_score, 0.85)

    def test_to_marqo_document_with_recency_score(self):
        """Test to_marqo_document() includes recency_score when present."""
        test_cases = [
            ("with_high_recency_score", 0.95),
            ("with_medium_recency_score", 0.5),
            ("with_low_recency_score", 0.1),
            ("with_zero_recency_score", 0.0),
            ("with_one_recency_score", 1.0),
        ]

        for test_name, recency_score_value in test_cases:
            with self.subTest(test_name):
                vespa_doc = {
                    "id": "test_id",
                    "fields": {
                        common.VESPA_FIELD_ID: "doc1",
                        common.VESPA_DOC_RECENCY_SCORE: recency_score_value
                    }
                }

                semi_structured_doc = SemiStructuredVespaDocument.from_vespa_document(
                    vespa_doc, marqo_index=self.mock_index
                )
                marqo_doc = semi_structured_doc.to_marqo_document(marqo_index=self.mock_index)

                self.assertIn(constants.MARQO_DOC_RECENCY_SCORE, marqo_doc)
                self.assertEqual(marqo_doc[constants.MARQO_DOC_RECENCY_SCORE], recency_score_value)

    def test_to_marqo_document_without_recency_score(self):
        """Test to_marqo_document() excludes recency_score when not present."""
        vespa_doc = {
            "id": "test_id",
            "fields": {
                common.VESPA_FIELD_ID: "doc1"
            }
        }

        semi_structured_doc = SemiStructuredVespaDocument.from_vespa_document(
            vespa_doc, marqo_index=self.mock_index
        )
        marqo_doc = semi_structured_doc.to_marqo_document(marqo_index=self.mock_index)

        self.assertNotIn(constants.MARQO_DOC_RECENCY_SCORE, marqo_doc)

    def test_recency_score_with_hybrid_scores(self):
        """Test recency_score coexists with hybrid search scores."""
        vespa_doc = {
            "id": "test_id",
            "fields": {
                common.VESPA_FIELD_ID: "doc1",
                common.VESPA_DOC_HYBRID_RAW_TENSOR_SCORE: 0.8,
                common.VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE: 0.6,
                common.VESPA_DOC_RECENCY_SCORE: 0.9
            }
        }

        semi_structured_doc = SemiStructuredVespaDocument.from_vespa_document(
            vespa_doc, marqo_index=self.mock_index
        )
        marqo_doc = semi_structured_doc.to_marqo_document(marqo_index=self.mock_index)

        # All three scores should be present
        self.assertIn(constants.MARQO_DOC_HYBRID_TENSOR_SCORE, marqo_doc)
        self.assertIn(constants.MARQO_DOC_HYBRID_LEXICAL_SCORE, marqo_doc)
        self.assertIn(constants.MARQO_DOC_RECENCY_SCORE, marqo_doc)

        self.assertEqual(marqo_doc[constants.MARQO_DOC_HYBRID_TENSOR_SCORE], 0.8)
        self.assertEqual(marqo_doc[constants.MARQO_DOC_HYBRID_LEXICAL_SCORE], 0.6)
        self.assertEqual(marqo_doc[constants.MARQO_DOC_RECENCY_SCORE], 0.9)

    def test_correct_constant_mapping(self):
        """Test that the correct constants are used for recency score mapping."""
        # Verify the constant mapping from Vespa to Marqo
        self.assertEqual(common.VESPA_DOC_RECENCY_SCORE, "marqo__recency_score")
        self.assertEqual(constants.MARQO_DOC_RECENCY_SCORE, "_recency_score")

        # Verify the mapping works in practice
        vespa_doc = {
            "id": "test_id",
            "fields": {
                common.VESPA_FIELD_ID: "doc1",
                "marqo__recency_score": 0.75  # Using the actual Vespa field name
            }
        }

        semi_structured_doc = SemiStructuredVespaDocument.from_vespa_document(
            vespa_doc, marqo_index=self.mock_index
        )
        marqo_doc = semi_structured_doc.to_marqo_document(marqo_index=self.mock_index)

        self.assertIn("_recency_score", marqo_doc)  # Using the actual Marqo field name
        self.assertEqual(marqo_doc["_recency_score"], 0.75)


if __name__ == '__main__':
    unittest.main()
