import os
import re

from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.unstructured_vespa_index.unstructured_vespa_schema import UnstructuredVespaSchema
from marqo.tensor_search.models.index_settings import IndexSettings
from tests.marqo_test import MarqoTestCase


class TestUnstructuredVespaSchema(MarqoTestCase):
    def _read_schema_from_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            schema = f.read()

        return schema

    def _remove_whitespace_in_schema(self, schema: str) -> str:
        """
        This function removes as much whitespace as possible from a schema without affecting its semantics.
        It is intended to help compare schemas independent of non-consequential syntactical differences such as
        new lines and indentation. Note, however, that not every new line can be removed without breaking the schema.
        """
        chars = re.escape('{}=+-<>():,;[]|')

        # Replace whitespace (including newlines) before or after any of the chars
        pattern = rf"(\s*([{chars}])\s*)"
        schema = re.sub(pattern, r"\2", schema)

        # Replace multiple spaces with a single space
        schema = re.sub(r' +', ' ', schema)

        # Replace leading whitespace and blank lines
        schema = re.sub(r'^\s+', '', schema, flags=re.MULTILINE)

        return schema

    def test_unstructured_index_schema_random_model(self):
        """A test for the unstructured Vespa schema generation with a random model."""
        index_name = "test_unstructured_schema"

        test_marqo_index_request: MarqoIndexRequest = IndexSettings(
            type="unstructured",
            model="random/small"
        ).to_marqo_index_request(index_name)

        test_unstructured_schema_object = UnstructuredVespaSchema(test_marqo_index_request)

        generated_schema, _ = test_unstructured_schema_object.generate_schema()

        expected_schema = self._read_schema_from_file('test_schemas/unstructured_vespa_index_schema.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(generated_schema)
        )