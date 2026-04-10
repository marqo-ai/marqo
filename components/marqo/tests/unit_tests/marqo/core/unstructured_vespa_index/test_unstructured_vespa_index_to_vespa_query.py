import time
import unittest

from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoHybridQuery
from marqo.core.models.marqo_index import (
    UnstructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric
)
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.models import MarqoQuery
from marqo.exceptions import InvalidArgumentError


class TestUnstructuredVespaIndexToVespaQuery(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with an unstructured index that supports both tensor and lexical search."""
        # Create an unstructured index
        marqo_index = self._create_unstructured_marqo_index(name='test_index')
        self.vespa_index = UnstructuredVespaIndex(marqo_index)

    def _create_unstructured_marqo_index(self, name: str) -> UnstructuredMarqoIndex:
        """Helper method to create an unstructured Marqo index for testing."""
        return UnstructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version='2.12.0',  # Version that supports hybrid search
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
            filter_string_max_length=50
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
                
                # Verify hybrid-specific fields are present
                self.assertEqual(vespa_query['hits'], 15)
                self.assertIn('searchChain', vespa_query)
                self.assertEqual(vespa_query['searchChain'], 'marqo')
                self.assertIn('marqo__hybrid.retrievalMethod', vespa_query)
                self.assertIn('marqo__hybrid.rankingMethod', vespa_query)


    def test_contains_filter_raises_error(self):
        """CONTAINS filter is not supported for legacy unstructured indexes."""
        marqo_query = MarqoQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            filter='title CONTAINS hello',
            score_modifiers=[],
            expose_facets=False
        )
        with self.assertRaises(InvalidArgumentError):
            self.vespa_index._get_filter_term(marqo_query)


if __name__ == '__main__':
    unittest.main()