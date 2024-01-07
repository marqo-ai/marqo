import unittest

from marqo.core import constants
from marqo.core.models import MarqoIndex
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
