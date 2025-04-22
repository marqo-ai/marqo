import re
import time
import unittest
from typing import List, Set, Optional

from marqo import version
from marqo.core.models import MarqoTensorQuery, MarqoLexicalQuery, MarqoQuery
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod, \
    ImagePreProcessing, HnswConfig, VectorNumericType, DistanceMetric, Field, FieldType, FieldFeature, TensorField, \
    StringArrayField
from marqo.core.semi_structured_vespa_index.common import STRING_ARRAY, BOOL_FIELDS, INT_FIELDS, FLOAT_FIELDS, \
    VESPA_FIELD_ID
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema


class TestSemiStructuredVespaIndexGetFilterString(unittest.TestCase):
    def setUp(self):
        # Create a dummy semi-structured index
        marqo_index = self._semi_structured_marqo_index(name='index1', lexical_field_names=['title'],
                                                        tensor_field_names=['title'])
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _semi_structured_marqo_index(self, name,
                                     lexical_field_names: List[str] = [],
                                     tensor_field_names: List[str] = [],
                                     marqo_version: Optional[str] = None,
                                     string_array_fields: List[str] = []) -> SemiStructuredMarqoIndex:
        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(
                patch_method=None
            ),
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(
                ef_construction=128,
                m=16
            ),
            marqo_version=version.get_version() if marqo_version is None else marqo_version,
            created_at=time.time(),
            updated_at=time.time(),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True,
            filter_string_max_length=100,
            lexical_fields=[
                Field(name=field_name, type=FieldType.Text,
                      features=[FieldFeature.LexicalSearch],
                      lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}')
                for field_name in lexical_field_names
            ],
            tensor_fields=[
                TensorField(
                    name=field_name,
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                )
                for field_name in tensor_field_names
            ],
            string_array_fields=[
                StringArrayField(
                    name=field_name,
                    type=FieldType.ArrayText,
                    string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}',
                    features=[]
                )
                for field_name in string_array_fields
            ]
        )

    def test_get_filter_string_escaped_characters(self):
        """
        Ensure the \ character is added to vespa query before all special characters (\ and ")
        """
        test_cases = [
            # Equality terms
            # no escaped characters
            ('title:hello',
             'key contains "title", value contains "hello"'),
            # Unescaped backslash gets ignored (double quote does not need to be escaped by user)
            ('title:hel"l\\o',
             'key contains "title", value contains "hel\\"lo"'),
            # Escaped backslash is also escaped in vespa query
            ('title:hel\\"l\\\\o',
             'key contains "title", value contains "hel\\"l\\\\o"'),
             ('ti\\"t\\\\le:hello',
              'key contains "ti\\"t\\\\le", value contains "hello"'),
            # Range terms
              ('nu\\"m\\\\ber:[1 TO 100]',
               'key contains "nu\\"m\\\\ber", value >= 1, value <= 100'),
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
                result_filter_string = self.vespa_index._get_filter_term(marqo_query)
                self.assertIn(expected_result, result_filter_string,)