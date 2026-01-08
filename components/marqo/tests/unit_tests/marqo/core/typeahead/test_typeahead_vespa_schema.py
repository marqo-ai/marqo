import os

from marqo.core.models.marqo_index import *
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
from tests.unit_tests.marqo_test import MarqoTestCase


class TestTypeaheadVespaSchema(MarqoTestCase):
    def _read_schema_from_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            schema = f.read()

        return schema

    def _remove_empty_lines_in_schema(self, schema: str) -> str:
        return '\n'.join([line for line in schema.splitlines() if line.strip()])

    def test_typeahead_schema_generation(self):
        """Test that TypeaheadVespaSchema generates the correct schema."""
        # Create a semi-structured index for testing
        test_marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            model=Model(name='hf/e5-small'),
        )

        # Generate the typeahead schema
        typeahead_schema = TypeaheadVespaSchema(test_marqo_index)
        generated_schema, updated_index = typeahead_schema.generate_schema()

        # Read expected schema
        expected_schema = self._read_schema_from_file('test_schemas/typeahead_vespa_schema.sd')

        # Verify the generated schema matches expected
        self.maxDiff = None
        self.assertEqual(
            self._remove_empty_lines_in_schema(expected_schema),
            self._remove_empty_lines_in_schema(generated_schema)
        )

        # Verify the updated index has the correct typeahead schema name
        expected_typeahead_schema_name = f"{test_marqo_index.schema_name}_typeahead"
        self.assertEqual(updated_index.typeahead_schema_name, expected_typeahead_schema_name)