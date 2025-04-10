from unittest.mock import MagicMock
from unit_tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument


class TestSemiStructuredPartialUpdate(MarqoTestCase):
    def _semistructured_index_creation_helper(self, marqo_version: str):
        """Only used for testing the vespa document creation. You can only have one 'test_field'
        field in the document."""
        return SemiStructuredMarqoIndex(
            name="test_index",
            schema_name="test_schema",
            model = MagicMock(spec=Model),
            normalize_embeddings=True,
            text_preprocessing=MagicMock(spec=TextPreProcessing),
            image_preprocessing=MagicMock(spec=ImagePreProcessing),
            distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=MagicMock(spec=HnswConfig),
            created_at=1,
            updated_at=2,
            version=None,
            treat_urls_and_pointers_as_images=False,
            filter_string_max_length=100,
            lexical_fields=[Field(name="test_field", type=FieldType.Text)],
            tensor_fields=[],
            marqo_version=marqo_version,
        )

    def test_index_supports_partial_updates_return_true_after_2160(self):
        index_version_216 = self._semistructured_index_creation_helper("2.16.0")
        self.assertTrue(index_version_216.index_supports_partial_updates)

    def test_index_supports_partial_updates_return_false_before_2160(self):
        index_version_216 = self._semistructured_index_creation_helper("2.13.5")
        self.assertFalse(index_version_216.index_supports_partial_updates)

    def test_uuid_is_not_included_for_index_before_2160(self):
        index_version_2150 = self._semistructured_index_creation_helper("2.15.0")
        marqo_document ={
            "_id": "test_id",
            "test_field": "test text",
        }
        vespa_document = (SemiStructuredVespaDocument.from_marqo_document(marqo_document, index_version_2150).
                          to_vespa_document())
        self.assertNotIn("marqo__version_uuid", vespa_document["fields"])

    def test_uuid_is_not_included_for_index_after_2160(self):
        index_version_21612 = self._semistructured_index_creation_helper("2.16.12")
        marqo_document ={
            "_id": "test_id",
            "test_field": "test text",
        }
        vespa_document = (SemiStructuredVespaDocument.from_marqo_document(marqo_document, index_version_21612).
                          to_vespa_document())
        self.assertIn("marqo__version_uuid", vespa_document["fields"])