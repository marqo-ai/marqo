import unittest
import os
from marqo.core import constants
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.core.models import MarqoIndex, UnstructuredMarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest, MarqoIndexRequest
from marqo.core.unstructured_vespa_index.unstructured_vespa_schema import UnstructuredVespaSchema
from tests.core.structured_vespa_index.test_structured_vespa_schema import TestStructuredVespaSchema


class TestUnstructuredVespaSchema(TestStructuredVespaSchema):
    def _read_schema_from_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            schema = f.read()

        return schema

    def test_unstructured_index_schema(self):
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