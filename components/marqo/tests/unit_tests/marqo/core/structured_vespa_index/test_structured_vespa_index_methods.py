import re
import time
import unittest
from typing import List, Set, Optional

from marqo import version
from marqo.core.models import MarqoTensorQuery, MarqoLexicalQuery, MarqoQuery
from marqo.core.models.marqo_index import StructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod, \
    ImagePreProcessing, HnswConfig, VectorNumericType, DistanceMetric, Field, FieldType, FieldFeature, TensorField, \
    StringArrayField
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.structured_vespa_index.structured_vespa_schema import StructuredVespaSchema
from marqo.core.exceptions import InvalidFieldNameError


class TestStructuredVespaIndexGetFilterString(unittest.TestCase):
    def setUp(self):
        # Create a dummy structured index
        marqo_index = self._structured_marqo_index(name='index1', text_field_names=['title'],
                                                   tensor_field_names=['title'])
        self.vespa_index = StructuredVespaIndex(marqo_index)

    def _structured_marqo_index(self, name: str, 
                               text_field_names: List[str] = [],
                               tensor_field_names: List[str] = []) -> StructuredMarqoIndex:
        fields = []
        # Add numeric field
        fields.append(
            Field(
                name='number',
                type=FieldType.Int,
                features=[FieldFeature.Filter],
                filter_field_name='number'
            )
        )
        for field_name in text_field_names:
            fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=field_name,
                    filter_field_name=field_name
                )
            )

        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=field_name,
                    chunk_field_name=f'chunks_{field_name}'
                )
            )

        return StructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version='2.0.0',
            created_at=time.time(),
            updated_at=time.time(),
            fields=fields,
            tensor_fields=tensor_fields,
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(
                patch_method=None
            )
        )

    def test_get_filter_string_escaped_characters(self):
        test_cases = [
            # Direct field access with escaped values
            ('title:hello', 'title contains "hello"'),
            ('title:hel"l\\o', 'title contains "hel\\"lo"'),
            ('title:hel\\"l\\\\o', 'title contains "hel\\"l\\\\o"'),
            
            # Range test cases
            ('number:[10 TO 20]', 'number >= 10 AND number <= 20'),
            
            # IN terms with escaped values
            ('title IN (hel\\"lo, wor\\\\ld)', 'title in ("hel\\"lo", "wor\\\\ld")'),
            ('title IN ((mul\\\\ti wor"d phrase), wor\\\\ld)', 'title in ("mul\\\\ti wor\\"d phrase", "wor\\\\ld")'),
            ('number:[1 TO 100] OR title IN (val\\"1, val\\\\2)',
             '(number >= 1 AND number <= 100) OR title in ("val\\"1", "val\\\\2")'),
            
            # Invalid field name cases (should raise InvalidFieldNameError)
            ('escaped\\"field:[1 TO 5]', 'Field validation should fail'),
            ('invalid\\\\field:value', 'Field validation should fail'),
            ('bad"field:test', 'Field validation should fail'),
            ('ti\\"t\\\\le IN ("hello\\"", "wor\\\\ld")', 'Field validation should fail'),
            ('field\\\\:name IN ("has\\"quote", "back\\\\slash", "normal")', 
             'Field validation should fail')
        ]

        for filter_string, expected_result in test_cases:
            with self.subTest(msg=f"Testing filter string: {filter_string}"):
                marqo_query = MarqoQuery(
                    index_name=self.vespa_index._marqo_index.name,
                    limit=10,
                    filter=filter_string,
                    score_modifiers=[],
                    expose_facets=False
                )
                if 'Field validation should fail' in expected_result:
                    with self.assertRaises(InvalidFieldNameError):
                        self.vespa_index._get_filter_term(marqo_query)
                else:
                    result_filter_string = self.vespa_index._get_filter_term(marqo_query)
                    self.assertIn(expected_result, result_filter_string)