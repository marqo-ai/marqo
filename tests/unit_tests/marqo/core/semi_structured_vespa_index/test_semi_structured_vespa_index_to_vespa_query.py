import time
import unittest
from typing import List

from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoHybridQuery
from marqo.core.models.marqo_index import (
    SemiStructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric, Field, FieldType,
    FieldFeature, TensorField, StringArrayField
)
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema


class TestSemiStructuredVespaIndexToVespaQuery(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with a semi-structured index that supports both tensor and lexical search."""
        # Create a semi-structured index with both lexical and tensor fields
        marqo_index = self._create_semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title', 'description'], 
            tensor_field_names=['title', 'description'],
            string_array_field_names=['tags']
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _create_semi_structured_marqo_index(
        self, 
        name: str,
        lexical_field_names: List[str] = [],
        tensor_field_names: List[str] = [],
        string_array_field_names: List[str] = []
    ) -> SemiStructuredMarqoIndex:
        """Helper method to create a semi-structured Marqo index for testing."""
        
        # Create lexical fields
        lexical_fields = []
        for field_name in lexical_field_names:
            lexical_fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}',
                    filter_field_name=f'{field_name}_filter'
                )
            )

        # Create tensor fields
        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}'
                )
            )

        # Create string array fields
        string_array_fields = []
        for field_name in string_array_field_names:
            string_array_fields.append(
                StringArrayField(
                    name=field_name,
                    type=FieldType.ArrayText,
                    features=[FieldFeature.Filter],
                    string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}'
                )
            )

        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version='2.16.0',  # Version that supports hybrid search and partial updates
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(
                patch_method=None
            ),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=string_array_fields
        )

    def test_to_vespa_query_tensor_mode_approximate_threshold(self):
        """Test that to_vespa_query correctly sets approximate threshold for tensor queries."""
        threshold_values = [0.75, 0.85, 0.95, None]
        
        for threshold in threshold_values:
            with self.subTest(approximate_threshold=threshold):
                marqo_query = MarqoTensorQuery(
                    index_name='test_index',
                    limit=10,
                    offset=0,
                    vector_query=[0.1, 0.2, 0.3, 0.4],
                    approximate_threshold=threshold,
                    approximate=True
                )

                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                if threshold is not None:
                    # Verify approximate threshold is set correctly
                    self.assertEqual(vespa_query['ranking.matching.approximateThreshold'], threshold)
                else:
                    # When threshold is None, it should not be included in the query
                    self.assertNotIn('ranking.matching.approximateThreshold', vespa_query)
                
                # Verify other key fields are present
                self.assertIn('yql', vespa_query)
                self.assertIn('ranking', vespa_query)
                self.assertEqual(vespa_query['hits'], 10)

    def test_to_vespa_query_hybrid_mode_approximate_threshold(self):
        """Test that to_vespa_query correctly sets approximate threshold for hybrid queries."""
        threshold_values = [0.70, 0.80, 0.90, None]
        
        for threshold in threshold_values:
            with self.subTest(approximate_threshold=threshold):
                hybrid_parameters = HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0.7,
                    rrfK=100
                )
                
                marqo_query = MarqoHybridQuery(
                    index_name='test_index',
                    limit=15,
                    offset=0,
                    vector_query=[0.2, 0.3, 0.4, 0.5],
                    or_phrases=['search', 'query'],
                    and_phrases=['required'],
                    hybrid_parameters=hybrid_parameters,
                    approximate_threshold=threshold,
                    approximate=True
                )

                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                if threshold is not None:
                    # Verify approximate threshold is set correctly
                    self.assertEqual(vespa_query['ranking.matching.approximateThreshold'], threshold)
                else:
                    # When threshold is None, it should not be included in the query
                    self.assertNotIn('ranking.matching.approximateThreshold', vespa_query)
                
                # Verify hybrid-specific fields are present for semi-structured indexes
                self.assertEqual(vespa_query['hits'], 15)
                self.assertIn('ranking', vespa_query)


if __name__ == '__main__':
    unittest.main() 