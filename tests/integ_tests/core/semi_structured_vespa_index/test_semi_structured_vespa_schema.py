import os
from typing import cast

from marqo.core.models.marqo_index import *
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from tests.integ_tests.marqo_test import MarqoTestCase


class TestSemiStructuredVespaSchema(MarqoTestCase):
    def _read_schema_from_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            schema = f.read()

        return schema

    def _remove_empty_lines_in_schema(self, schema: str) -> str:
        return '\n'.join([line for line in schema.splitlines() if line.strip()])

    def test_semi_structured_index_schema_random_model(self):
        test_cases = [
            # test_case_name, lexical_fields, tensor_fields, expected schema file, string_array_fields
            ('no_field', [], [], 'semi_structured_vespa_index_schema_no_field.sd', []),
            ('one_lexical_field', ['text_field'], [], 'semi_structured_vespa_index_schema_one_lexical_field.sd', []),
             ('one_tensor_field', [], ['tensor_field'], 'semi_structured_vespa_index_schema_one_tensor_field.sd', []),
            ('one_string_array_field', [], [], 'semi_structured_vespa_index_schema_one_string_array_field.sd', ['string_array_1']),
             ('one_lexical_one_tensor_field', ['text_field'], ['tensor_field'], 'semi_structured_vespa_index_schema_one_lexical_one_tensor_field.sd', []),
            ('one_lexical_one_tensor_field_one_string_array_field', ['text_field'], ['tensor_field'], 'semi_structured_vespa_index_schema_one_lexical_one_tensor_one_string_array_field.sd', ['string_array_1']),
            ('multiple_lexical_tensor_fields', ['text_field1', 'text_field2'], ['tensor_field1', 'tensor_field2'], 'semi_structured_vespa_index_schema_multiple_lexical_tensor_and_string_array_fields.sd', ['string_array_1', 'string_array_2']),
        ]

        for test_case in test_cases:
            with (self.subTest(msg=test_case[0])):
                lexical_fields = test_case[1]
                tensor_fields = test_case[2]
                string_array_fields = test_case[4]
                expected_schema = self._read_schema_from_file(f'test_schemas/{test_case[3]}')

                test_marqo_index_request = self.unstructured_marqo_index_request(
                    name="test_semi_structured_schema",
                    hnsw_config=HnswConfig(ef_construction=512, m=16),
                    distance_metric=DistanceMetric.PrenormalizedAngular
                )

                _, index = SemiStructuredVespaSchema(test_marqo_index_request).generate_schema()
                marqo_index = cast(SemiStructuredMarqoIndex, index)

                for lexical_field in lexical_fields:
                    marqo_index.lexical_fields.append(
                        Field(name=lexical_field, type=FieldType.Text,
                              features=[FieldFeature.LexicalSearch],
                              lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{lexical_field}'))
                for tensor_field in tensor_fields:
                    marqo_index.tensor_fields.append(TensorField(
                        name=tensor_field,
                        chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{tensor_field}',
                        embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{tensor_field}',
                    ))
                for string_array_field in string_array_fields:
                    marqo_index.string_array_fields.append(StringArrayField(
                        name=string_array_field,
                        type=FieldType.ArrayText,
                        string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{string_array_field}',
                        features=[]
                    ))
                marqo_index.clear_cache()
                generated_schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)

                self.maxDiff = None
                self.assertEqual(
                    self._remove_empty_lines_in_schema(expected_schema),
                    self._remove_empty_lines_in_schema(generated_schema)
                )