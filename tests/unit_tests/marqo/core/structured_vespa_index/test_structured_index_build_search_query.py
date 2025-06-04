from magic import Magic
from unittest.mock import MagicMock

from marqo.core.exceptions import AddDocumentsError
from marqo.core.inference.api import Inference, Modality
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import FieldType
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.models.marqo_query import MarqoLexicalQuery



class TestStructuredIndexBuildLexicalSearchQuery(MarqoTestCase):
    """
    Test the build_lexical_search_query method of StructuredVespaIndex.
    Note that this method is also used in semi-structured Vespa index.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.maxDiff = None
        cls.structured_vespa_index = StructuredVespaIndex(
            marqo_index=cls.structured_marqo_index(
                name='index1',
                schema_name='index1',
                fields=[
                    Field(name='lexical_field_1', type=FieldType.Text, features=[FieldFeature.LexicalSearch],
                          lexical_field_name='marqo__lexical_lexical_field_1'),
                    Field(name='lexical_field_2', type=FieldType.Text, features=[FieldFeature.LexicalSearch],
                          lexical_field_name='marqo__lexical_lexical_field_2'),
                    Field(name='lexical_field_3', type=FieldType.Text, features=[FieldFeature.LexicalSearch],
                          lexical_field_name='marqo__lexical_lexical_field_3'),
                    Field(name='lexical_field_4', type=FieldType.Text, features=[FieldFeature.LexicalSearch],
                          lexical_field_name='marqo__lexical_lexical_field_4')
                ],
                tensor_fields=[]
            )
        )

    def _help_create_test_lexical_query_object(self, or_phrases: List[str], and_phrases: List[str],
                                               searchable_attributes: List[str] = None) -> MarqoLexicalQuery:
        """
        Helper function to create a MarqoLexicalQuery object with the given parameters.
        """
        return MarqoLexicalQuery(
            index_name='index1',
            limit=10,
            offset=0,
            searchable_attributes=searchable_attributes,
            or_phrases=or_phrases,
            and_phrases=and_phrases
        )

    def test_generate_and_terms_for_lexical_search(self):
        """Test to ensure that the AND terms are correctly built when searchable attributes are specified.
        """
        test_cases = [
            (
                ["required_1"], ['lexical_field_1'],
                '(marqo__lexical_lexical_field_1 contains "required_1")',
                "Simple 1 required term, 1 field"
            ),
            (
                ["required_1", "required_2"], ['lexical_field_1'],
                '(marqo__lexical_lexical_field_1 contains "required_1") AND '
                '(marqo__lexical_lexical_field_1 contains "required_2")',
                "2 required terms, 1 field"
            ),
            (
                ["required_1", "required_2"], ['lexical_field_1', 'lexical_field_2'],
                '(marqo__lexical_lexical_field_1 contains "required_1" '
                'OR marqo__lexical_lexical_field_2 contains "required_1") AND '
                '(marqo__lexical_lexical_field_1 contains "required_2" '
                'OR marqo__lexical_lexical_field_2 contains "required_2")',
                "2 required terms, 2 fields"
            ),
            (
                ["long required phrase", "short required phrase"], ['lexical_field_1', 'lexical_field_4'],
                '(marqo__lexical_lexical_field_1 contains "long required phrase" '
                'OR marqo__lexical_lexical_field_4 contains "long required phrase") AND '
                '(marqo__lexical_lexical_field_1 contains "short required phrase" OR '
                'marqo__lexical_lexical_field_4 contains "short required phrase")',
                "2 multiple words required phrases, 2 fields"
            ),
            (
                ["term1", "term2"], None,
                'default contains "term1" AND default contains "term2"',
                "2 required terms, no fields"
            ),
            (
                ["term1", "term2", "term3", "term4"], ['lexical_field_1', 'lexical_field_2', 'lexical_field_3'],
                '(marqo__lexical_lexical_field_1 contains "term1" OR marqo__lexical_lexical_field_2 contains "term1" '
                'OR marqo__lexical_lexical_field_3 contains "term1") AND (marqo__lexical_lexical_field_1 contains "term2" '
                'OR marqo__lexical_lexical_field_2 contains "term2" OR marqo__lexical_lexical_field_3 contains "term2") AND '
                '(marqo__lexical_lexical_field_1 contains "term3" OR marqo__lexical_lexical_field_2 contains "term3" OR '
                'marqo__lexical_lexical_field_3 contains "term3") AND (marqo__lexical_lexical_field_1 contains "term4" '
                'OR marqo__lexical_lexical_field_2 contains "term4" OR marqo__lexical_lexical_field_3 contains "term4")',
                "4 required terms, 3 fields"
            )
        ]
        for required_phrases, searchable_attributes, expected_and_terms, msg in test_cases:
            with self.subTest(f"{msg}"):
                test_marqo_query = self._help_create_test_lexical_query_object(
                    or_phrases=[],
                    and_phrases=required_phrases,
                    searchable_attributes=searchable_attributes
                )
                generated_and_terms = self.structured_vespa_index._get_lexical_search_term(
                    test_marqo_query
                )
                self.assertEqual(generated_and_terms, expected_and_terms)