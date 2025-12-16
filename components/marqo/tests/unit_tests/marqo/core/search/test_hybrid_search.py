import os
from unittest import TestCase
from unittest.mock import Mock, patch
from pydantic.v1 import ValidationError
import semver

from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.models.facets_parameters import (
    FacetsParameters, FieldFacetsConfiguration
)
from marqo.core.search.hybrid_search import HybridSearch
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.tensor_search.models.api_models import ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.search import SearchContext, SearchContextDocuments, SearchContextTensor
from marqo.config import Config
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints
from marqo.tensor_search.enums import EnvVars


class TestHybridSearch(TestCase):

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_search_creates_correct_marqo_hybrid_query(
        self, mock_metrics, mock_gather_docs, mock_parse_lexical,
        mock_vectorise, mock_vespa_factory
    ):
        """Test that HybridSearch.search creates MarqoHybridQuery with all relevant parameters."""
        
        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response
        
        # Mock marqo_index
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.21.0")
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        
        # Mock vespa_index
        mock_vespa_index = Mock(spec=SemiStructuredVespaIndex)
        mock_vespa_query = {"query": "test"}
        mock_vespa_index.to_vespa_query.return_value = mock_vespa_query
        mock_vespa_index.gather_facets_from_response.return_value = {"facets": {"test_field": {}}}
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock vectorisation pipeline
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}
        
        # Mock lexical query parsing
        mock_parse_lexical.return_value = (["required"], ["optional"])
        
        # Mock metrics store
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value = mock_context_manager
        
        # Mock gather_documents_from_response
        mock_gather_docs.return_value = {
            "hits": [{"_id": "1", "doc": {"field": "value"}}]
        }
        
        # Patch MarqoHybridQuery to capture its creation
        with patch('marqo.core.search.hybrid_search.MarqoHybridQuery') as mock_marqo_query:
            mock_query_instance = Mock()
            mock_marqo_query.return_value = mock_query_instance
            
            # Setup test parameters
            score_modifiers = Mock(spec=ScoreModifierLists)
            score_modifiers.to_marqo_score_modifiers.return_value = [
                ScoreModifier(field="field1", weight=1.0, type=ScoreModifierType.Add)
            ]
            
            score_modifiers_lexical = Mock(spec=ScoreModifierLists)
            score_modifiers_lexical.to_marqo_score_modifiers.return_value = [
                ScoreModifier(field="field2", weight=2.0, type=ScoreModifierType.Multiply)
            ]
            
            score_modifiers_tensor = Mock(spec=ScoreModifierLists)
            score_modifiers_tensor.to_marqo_score_modifiers.return_value = [
                ScoreModifier(field="field3", weight=0.5, type=ScoreModifierType.Add)
            ]
            
            hybrid_parameters = HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.7,
                rrfK=100,
                scoreModifiersLexical=score_modifiers_lexical,
                scoreModifiersTensor=score_modifiers_tensor
            )
            
            facets = FacetsParameters(
                fields={"test_field": FieldFacetsConfiguration(type="string")}
            )
            
            # Execute the search
            hybrid_search = HybridSearch()
            result = hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test query",
                result_count=10,
                offset=5,
                rerank_depth=50,
                ef_search=100,
                approximate=False,
                approximate_threshold=0.95,
                searchable_attributes=["field1", "field2"],
                filter_string="field:value",
                attributes_to_retrieve=["field1", "field3"],
                score_modifiers=score_modifiers,
                hybrid_parameters=hybrid_parameters,
                facets=facets,
                track_total_hits=True,
                language="en"
            )
            
            # Verify MarqoHybridQuery was created with correct parameters
            mock_marqo_query.assert_called_once()
            call_args = mock_marqo_query.call_args[1]  # Get keyword arguments
            
            # Verify all important parameters
            self.assertEqual(call_args['index_name'], "test_index")
            self.assertEqual(call_args['vector_query'], [0.1, 0.2, 0.3])
            self.assertEqual(call_args['filter'], "field:value")
            self.assertEqual(call_args['limit'], 10)
            self.assertEqual(call_args['ef_search'], 100)
            self.assertFalse(call_args['approximate'])
            self.assertEqual(call_args['approximate_threshold'], 0.95)
            self.assertEqual(call_args['offset'], 5)
            self.assertEqual(call_args['global_rerank_depth'], 50)
            self.assertEqual(call_args['or_phrases'], ["optional"])
            self.assertEqual(call_args['and_phrases'], ["required"])
            self.assertEqual(call_args['attributes_to_retrieve'], ["field1", "field3"])
            self.assertEqual(call_args['searchable_attributes'], ["field1", "field2"])
            self.assertEqual(call_args['hybrid_parameters'], hybrid_parameters)
            self.assertEqual(call_args['facets'], facets)
            self.assertTrue(call_args['track_total_hits'])
            self.assertEqual(call_args['language'], "en")
            
            # Verify score_modifiers are processed correctly
            self.assertIsNotNone(call_args['score_modifiers'])
            self.assertIsNotNone(call_args['score_modifiers_lexical'])
            self.assertIsNotNone(call_args['score_modifiers_tensor'])
            
            # Verify the search executed successfully
            self.assertIsNotNone(result) 

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_search_custom_vector_query_with_existing_context_tensor_none(
        self, mock_metrics, mock_gather_docs, mock_parse_lexical, 
        mock_vectorise, mock_vespa_factory
    ):
        """Test CustomVectorQuery handling when context exists but context.tensor is None (line 200)."""
        
        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response
        
        # Mock marqo_index
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.21.0")
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        
        # Mock vespa_index
        mock_vespa_index = Mock(spec=SemiStructuredVespaIndex)
        mock_vespa_query = {"query": "test"}
        mock_vespa_index.to_vespa_query.return_value = mock_vespa_query
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock vectorisation pipeline
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}
        
        # Mock lexical query parsing
        mock_parse_lexical.return_value = (["required"], ["optional"])
        
        # Mock metrics store
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value = mock_context_manager
        
        # Mock gather_documents_from_response
        mock_gather_docs.return_value = {
            "hits": [{"_id": "1", "doc": {"field": "value"}}]
        }
        
        # Create CustomVectorQuery
        custom_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.5, 0.6, 0.7]
            )
        )
        
        # Create context with tensor=None (this triggers line 200)
        context = SearchContext(
            tensor=None,  # This is key - tensor is None
            documents=SearchContextDocuments(ids={"doc1": 1.0})
        )
        
        # Execute the search
        hybrid_search = HybridSearch()
        result = hybrid_search.search(
            config=config,
            marqo_index=marqo_index,
            query=custom_query,
            context=context,
            hybrid_parameters=HybridParameters()
        )
        
        # Verify the search executed successfully
        self.assertIsNotNone(result)
        
        # Verify that context.tensor was created (line 202)
        self.assertIsNotNone(context.tensor)
        self.assertEqual(len(context.tensor), 1)
        self.assertEqual(context.tensor[0].vector, [0.5, 0.6, 0.7])
        self.assertEqual(context.tensor[0].weight, 1)

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_search_custom_vector_query_with_existing_context_tensor_exists(
        self, mock_metrics, mock_gather_docs, mock_parse_lexical, 
        mock_vectorise, mock_vespa_factory
    ):
        """Test CustomVectorQuery handling when context.tensor already exists (append scenario)."""
        
        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response
        
        # Mock marqo_index
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.21.0")
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        
        # Mock vespa_index
        mock_vespa_index = Mock(spec=SemiStructuredVespaIndex)
        mock_vespa_query = {"query": "test"}
        mock_vespa_index.to_vespa_query.return_value = mock_vespa_query
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock vectorisation pipeline
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}
        
        # Mock lexical query parsing
        mock_parse_lexical.return_value = (["required"], ["optional"])
        
        # Mock metrics store
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value = mock_context_manager
        
        # Mock gather_documents_from_response
        mock_gather_docs.return_value = {
            "hits": [{"_id": "1", "doc": {"field": "value"}}]
        }
        
        # Create CustomVectorQuery
        custom_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.5, 0.6, 0.7]
            )
        )
        
        # Create context with existing tensor (this triggers the append scenario)
        existing_tensor = SearchContextTensor(vector=[0.1, 0.2, 0.3], weight=0.5)
        context = SearchContext(
            tensor=[existing_tensor]  # Already has a tensor
        )
        
        # Execute the search
        hybrid_search = HybridSearch()
        result = hybrid_search.search(
            config=config,
            marqo_index=marqo_index,
            query=custom_query,
            context=context,
            hybrid_parameters=HybridParameters()
        )
        
        # Verify the search executed successfully
        self.assertIsNotNone(result)
        
        # Verify that the new tensor was appended
        self.assertEqual(len(context.tensor), 2)
        self.assertEqual(context.tensor[0].vector, [0.1, 0.2, 0.3])  # Original tensor
        self.assertEqual(context.tensor[0].weight, 0.5)
        self.assertEqual(context.tensor[1].vector, [0.5, 0.6, 0.7])  # Appended tensor
        self.assertEqual(context.tensor[1].weight, 1)

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_search_custom_vector_query_with_capped_total_hits(
            self, mock_metrics, mock_gather_docs, mock_parse_lexical,
            mock_vectorise, mock_vespa_factory
    ):
        """Test CustomVectorQuery handling when context.tensor already exists (append scenario)."""

        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response

        # Mock marqo_index
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.21.0")
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""

        # Mock vespa_index
        mock_vespa_index = Mock(spec=SemiStructuredVespaIndex)
        mock_vespa_query = {"query": "test"}
        mock_vespa_index.to_vespa_query.return_value = mock_vespa_query
        mock_vespa_factory.return_value = mock_vespa_index

        # Mock vectorisation pipeline
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}

        # Mock lexical query parsing
        mock_parse_lexical.return_value = (["required"], ["optional"])

        # Mock metrics store
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value = mock_context_manager

        # Mock gather_documents_from_response
        mock_gather_docs.return_value = {
            "hits": [{"_id": "1", "doc": {"field": "value"}}]
        }
        mock_vespa_index.gather_facets_from_response.return_value = {"totalHits": 20_000}

        # Execute the search
        hybrid_search = HybridSearch()
        result = hybrid_search.search(
            config=config,
            marqo_index=marqo_index,
            query="test",
            track_total_hits=True,
            hybrid_parameters=HybridParameters()
        )

        # Verify the search executed successfully
        self.assertIsNotNone(result)
        capped_total_hits = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
        self.assertEqual(
            capped_total_hits, result["totalHits"],
            f"The total hits should be capped to the env var value {capped_total_hits}, "
            f"but got {result['totalHits']}"
        )
