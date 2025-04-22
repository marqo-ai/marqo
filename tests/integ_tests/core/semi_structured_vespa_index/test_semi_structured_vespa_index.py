import re
import time
import unittest
from typing import List, Set, Optional

from marqo import version
from marqo.core.models import MarqoTensorQuery, MarqoLexicalQuery
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod, \
    ImagePreProcessing, HnswConfig, VectorNumericType, DistanceMetric, Field, FieldType, FieldFeature, TensorField, \
    StringArrayField
from marqo.core.semi_structured_vespa_index.common import STRING_ARRAY, BOOL_FIELDS, INT_FIELDS, FLOAT_FIELDS, \
    VESPA_FIELD_ID
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema


class TestSemiStructuredVespaIndexToVespaQuery(unittest.TestCase):

    def test_to_vespa_query_should_include_static_fields_when_attributes_to_retrieve_is_not_empty(self):
        marqo_index = self._semi_structured_marqo_index(name='index1', lexical_field_names=['title'],
                                                        tensor_field_names=['title'])
        vespa_index = SemiStructuredVespaIndex(marqo_index)

        for marqo_query in [
            MarqoTensorQuery(index_name=marqo_index.name, limit=10, offset=0,
                             attributes_to_retrieve=['title'], vector_query=[1.0] * 10),
            MarqoLexicalQuery(index_name=marqo_index.name, limit=10, offset=0,
                              attributes_to_retrieve=['title'], and_phrases=['hello'], or_phrases=['world']),
            # MarqoHybridSearch yql is just a placeholder and is generated in customer search component.
        ]:
            with self.subTest(test_query=marqo_query):
                query = vespa_index.to_vespa_query(marqo_query)
                fields = self._extract_fields_from_yql(query['yql'])

                self.assertSetEqual({VESPA_FIELD_ID, 'title', 'marqo__chunks_title',
                                     BOOL_FIELDS, INT_FIELDS, FLOAT_FIELDS}, fields)

    def test_to_vespa_query_should_include_string_array_fields_when_string_array_attributes_to_retrieve_is_not_empty(self):
        """Tests that string array fields are correctly included in the Vespa query when requested.
        
        This test verifies that when string array fields are included in the attributes_to_retrieve
        parameter of a query, both the original field names and their corresponding Vespa internal
        field names (with the marqo__string_array_ prefix) are included in the generated YQL query.
        
        The test checks this behavior for both tensor queries and lexical queries to ensure
        consistent handling across query types.
        """
        marqo_index = self._semi_structured_marqo_index(name='index1', lexical_field_names=['title'],
                                                        tensor_field_names=['title'], string_array_fields = ['string_array_field1', 'string_array_field2'])
        vespa_index = SemiStructuredVespaIndex(marqo_index)

        for marqo_query in [
            MarqoTensorQuery(index_name=marqo_index.name, limit=10, offset=0,
                             attributes_to_retrieve=['title', 'string_array_field1', 'string_array_field2'], vector_query=[1.0] * 10),
            MarqoLexicalQuery(index_name=marqo_index.name, limit=10, offset=0,
                              attributes_to_retrieve=['title', 'string_array_field1', 'string_array_field2'], and_phrases=['hello'], or_phrases=['world']),
            # MarqoHybridSearch yql is just a placeholder and is generated in customer search component.
        ]:
            with self.subTest(test_query=marqo_query):
                query = vespa_index.to_vespa_query(marqo_query)
                fields = self._extract_fields_from_yql(query['yql'])

                self.assertSetEqual({VESPA_FIELD_ID, 'title', 'marqo__chunks_title',
                                     BOOL_FIELDS, INT_FIELDS, FLOAT_FIELDS,
                                     'string_array_field1', 'string_array_field2', 'marqo__string_array_string_array_field1', 'marqo__string_array_string_array_field2'}, fields)

    def test_to_vespa_query_should_include_static_fields_when_attributes_to_retrieve_is_not_empty_pre_2_16_indexes(self):
        """
        Test that a vespa query is correctly generated for a schema with marqo version < 2.16.0
        Returns:

        """
        marqo_index = self._semi_structured_marqo_index(name='index1', lexical_field_names=['title'],
                                                        tensor_field_names=['title'], marqo_version = '2.15.0')
        vespa_index = SemiStructuredVespaIndex(marqo_index)

        for marqo_query in [
            MarqoTensorQuery(index_name=marqo_index.name, limit=10, offset=0,
                             attributes_to_retrieve=['title'], vector_query=[1.0] * 10),
            MarqoLexicalQuery(index_name=marqo_index.name, limit=10, offset=0,
                              attributes_to_retrieve=['title'], and_phrases=['hello'], or_phrases=['world']),
            # MarqoHybridSearch yql is just a placeholder and is generated in customer search component.
        ]:
            with self.subTest(test_query=marqo_query):
                query = vespa_index.to_vespa_query(marqo_query)
                fields = self._extract_fields_from_yql(query['yql'])

                self.assertSetEqual({VESPA_FIELD_ID, 'title', 'marqo__chunks_title', STRING_ARRAY,
                                     BOOL_FIELDS, INT_FIELDS, FLOAT_FIELDS}, fields)


    def test_to_vespa_query_should_not_include_static_fields_when_attributes_to_retrieve_is_empty(self):
        marqo_index = self._semi_structured_marqo_index(name='index1', lexical_field_names=['title'],
                                                        tensor_field_names=['title'])
        vespa_index = SemiStructuredVespaIndex(marqo_index)

        for marqo_query in [
            MarqoTensorQuery(index_name=marqo_index.name, limit=10, offset=0,
                             attributes_to_retrieve=[], vector_query=[1.0] * 10),
            MarqoLexicalQuery(index_name=marqo_index.name, limit=10, offset=0,
                              attributes_to_retrieve=[], and_phrases=['hello'], or_phrases=['world']),
            # MarqoHybridSearch yql is just a placeholder and is generated in customer search component.
        ]:
            with self.subTest(test_query=marqo_query):
                query = vespa_index.to_vespa_query(marqo_query)
                fields = self._extract_fields_from_yql(query['yql'])

                self.assertSetEqual({VESPA_FIELD_ID}, fields)

    def _extract_fields_from_yql(self, yql: str) -> Set[str]:
        # Define the regex pattern to extract fields from the SELECT clause
        pattern = r"select\s+(.*?)\s+from"

        # Search for the fields between SELECT and FROM
        match = re.search(pattern, yql, re.IGNORECASE)

        if match:
            # Extract the fields and split them by commas, trimming any extra spaces
            fields = match.group(1).split(',')
            return set([field.strip() for field in fields])
        else:
            raise ValueError("No fields found in the query.")

    def _semi_structured_marqo_index(self, name='index_name',
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