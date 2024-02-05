import unittest

from marqo.core import constants
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.core.models import MarqoIndex, UnstructuredMarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest, MarqoIndexRequest
from marqo.core.unstructured_vespa_index.unstructured_vespa_schema import UnstructuredVespaSchema
from marqo.core.vespa_schema import VespaSchema



class VespaSchemaImplementation(VespaSchema):
    def generate_schema(self) -> (str, MarqoIndex):
        pass


class TestVespaSchema(unittest.TestCase):

    def setUp(self):
        self.vespa_schema = VespaSchemaImplementation()

    def test_normal_case(self):
        index_name = "validSchemaName123"
        expected = "validSchemaName123"
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)

    def test_encoding_underscore(self):
        index_name = "schema_name"
        expected = f"{constants.MARQO_RESERVED_PREFIX}schema_00name"
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)

    def test_encoding_hyphen(self):
        index_name = "schema-name"
        expected = f"{constants.MARQO_RESERVED_PREFIX}schema_01name"
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)

    def test_empty_string(self):
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(''), '')

    def test_long_string_with_special_characters(self):
        index_name = "_" * 1000  # Very long string of underscores
        expected = f"{constants.MARQO_RESERVED_PREFIX}" + "_00" * 1000
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)

    def test_special_characters_only(self):
        index_name = "_-_"
        expected = f"{constants.MARQO_RESERVED_PREFIX}_00_01_00"
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)

    def test_mixed_characters(self):
        index_name = "test_schema-name"
        expected = f"{constants.MARQO_RESERVED_PREFIX}test_00schema_01name"
        self.assertEqual(self.vespa_schema._get_vespa_schema_name(index_name), expected)


class TestUnstructuredVespaSchema(unittest.TestCase):

    INDEX_NAME = "test_unstructured_schema"

    TEST_MARQO_INDEX_REQUEST: UnstructuredMarqoIndexRequest = IndexSettings(
        type="unstructured",
        model="random/small"
    ).to_marqo_index_request(INDEX_NAME)

    TEST_UNSTRUCTURED_SCHEMA_OBJECT = UnstructuredVespaSchema(TEST_MARQO_INDEX_REQUEST)

    GENERATED_SCHEMA, _ = TEST_UNSTRUCTURED_SCHEMA_OBJECT.generate_schema()

    def assertFieldInSchema(self, field_name, field_type, schema, additional_conditions=None):
        """
        Utility method to assert that a field with a specific configuration is present in the schema.
        :param field_name: Name of the field to check
        :param field_type: Expected type of the field
        :param schema: The schema string
        :param additional_conditions: List of additional string conditions that should be present in the field definition
        """
        self.assertIn(f"field {field_name} type {field_type}", schema)
        if additional_conditions:
            for condition in additional_conditions:
                self.assertIn(condition, schema)

    def test_fields_in_schema(self):
        """
        Test that all fields are correctly defined in the schema.
        """
        self.assertFieldInSchema("marqo__id", "string", self.GENERATED_SCHEMA, ["indexing: attribute | summary"])
        self.assertFieldInSchema("marqo__strings", "array<string>", self.GENERATED_SCHEMA,
                                 ["indexing: index", "index: enable-bm25"])
        self.assertFieldInSchema("marqo__long_string_fields", "map<string, string>", self.GENERATED_SCHEMA,
                                 ["indexing: summary"])
        self.assertFieldInSchema("marqo__short_string_fields", "map<string, string>", self.GENERATED_SCHEMA,
                                 ["indexing: summary"])
        self.assertFieldInSchema("marqo__string_array", "array<string>", self.GENERATED_SCHEMA,
                                 ["indexing: attribute | summary", "attribute: fast-search", "rank: filter"])
        self.assertFieldInSchema("marqo__multimodal_params", "map<string, string>", self.GENERATED_SCHEMA,
                                 ["indexing: summary"])
        self.assertFieldInSchema("marqo__int_fields", "map<string, int>", self.GENERATED_SCHEMA, ["indexing: summary"])
        self.assertFieldInSchema("marqo__bool_fields", "map<string, byte>", self.GENERATED_SCHEMA,
                                 ["indexing: summary"])
        self.assertFieldInSchema("marqo__float_fields", "map<string, float>", self.GENERATED_SCHEMA,
                                 ["indexing: summary"])
        self.assertFieldInSchema("marqo__score_modifiers", "tensor<float>(p{})", self.GENERATED_SCHEMA,
                                 ["indexing: attribute | summary"])
        self.assertFieldInSchema("marqo__chunks", "array<string>", self.GENERATED_SCHEMA, ["indexing: summary"])
        self.assertFieldInSchema("marqo__vector_count", "int", self.GENERATED_SCHEMA, ["indexing: attribute | summary"])
        self.assertFieldInSchema("marqo__embeddings", f"tensor<float>(p{{}}, x[{self.TEST_MARQO_INDEX_REQUEST.model.get_dimension()}])",
                                 self.GENERATED_SCHEMA,
                                 ["indexing: attribute | index | summary"])

    def test_rank_profiles_in_schema(self):
        """
        Test that all rank profiles are correctly defined in the schema.
        """
        self.assertIn("rank-profile embedding_similarity inherits default", self.GENERATED_SCHEMA)
        self.assertIn("rank-profile bm25 inherits default", self.GENERATED_SCHEMA)
        self.assertIn("rank-profile modifiers inherits default", self.GENERATED_SCHEMA)
        self.assertIn("rank-profile bm25_modifiers inherits modifiers", self.GENERATED_SCHEMA)
        self.assertIn("rank-profile embedding_similarity_modifiers inherits modifiers", self.GENERATED_SCHEMA)

    def test_all_non_vector_summary(self):
        """Test to verify that all fields are correctly defined in the all-non-vector-summary document-summary."""
        non_vector_summary_fields = [
            ("marqo__id", "string"),
            ("marqo__strings", "array<string>"),
            ("marqo__long_string_fields", "map<string, string>"),
            ("marqo__short_string_fields", "map<string, string>"),
            ("marqo__string_array", "array<string>"),
            ("marqo__bool_fields", "map<string, byte>"),
            ("marqo__int_fields", "map<string, int>"),
            ("marqo__float_fields", "map<string, float>"),
            ("marqo__chunks", "array<string>"),
        ]

        extracted_section = self.GENERATED_SCHEMA.split("document-summary all-non-vector-summary", 1)[1]
        for field, field_type in non_vector_summary_fields:
            self.assertIn(f"summary {field} type {field_type}", extracted_section)

    def test_all_vector_summary(self):
        """Test to verify that all fields are correctly defined in the all-vector-summary document-summary."""
        vector_summary_fields = [
            ("marqo__id", "string"),
            ("marqo__strings", "array<string>"),
            ("marqo__long_string_fields", "map<string, string>"),
            ("marqo__short_string_fields", "map<string, string>"),
            ("marqo__string_array", "array<string>"),
            ("marqo__bool_fields", "map<string, byte>"),
            ("marqo__int_fields", "map<string, int>"),
            ("marqo__float_fields", "map<string, float>"),
            ("marqo__chunks", "array<string>"),
            ("marqo__embeddings", "tensor<float>(p{}, x[32])"),
        ]

        extracted_section = self.GENERATED_SCHEMA.split("document-summary all-vector-summary", 1)[1]
        for field, field_type in vector_summary_fields:
            self.assertIn(f"summary {field} type {field_type}", extracted_section)