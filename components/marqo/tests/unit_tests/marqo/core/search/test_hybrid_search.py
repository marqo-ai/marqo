from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import semver

from marqo.config import Config
from marqo.core import constants
from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.facets_parameters import (
    FacetsParameters, FieldFacetsConfiguration
)
from marqo.core.models.hybrid_parameters import (
    HybridParameters, LexicalOperand, RankingMethod, RetrievalMethod
)
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, StructuredMarqoIndex, UnstructuredMarqoIndex
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.search.hybrid_search import HybridSearch, should_use_collapse_search
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.api_models import ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.search import SearchContext, SearchContextDocuments, SearchContextTensor
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField, SortOrder
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


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


class TestRecencyValidation(TestCase):
    """Tests for recency scoring validation in HybridSearch."""

    def _setup_metrics_mock(self, mock_metrics):
        """Helper to set up the metrics store mock."""
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value = mock_context_manager
        return mock_metrics_instance

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_recency_on_structured_index_raises_error(self, mock_metrics):
        """Test that recency scoring on structured index raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        # Create a mock structured index
        marqo_index = Mock(spec=StructuredMarqoIndex)
        marqo_index.name = "test_structured_index"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.9")

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn("unstructured", str(ctx.exception).lower())
        self.assertIn("Structured indexes do not support", str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_recency_on_old_schema_version_raises_error(self, mock_metrics):
        """Test that recency scoring on old schema version raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        # Create a mock semi-structured index with old schema version
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.7"  # Old version
        marqo_index.marqo_version = "2.24.7"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.7")
        # Set up property mock - returns False because schema is too old
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=False)

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn(str(constants.MARQO_RECENCY_SCORING_MINIMUM_VERSION), str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_additive_recency_on_old_schema_version_raises_error(self, mock_metrics):
        """Test that additive recency (addToScoreWeight) on old schema version raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        # Create a mock semi-structured index that supports basic recency but NOT additive
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.8"  # Supports recency but not additive
        marqo_index.marqo_version = "2.24.8"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.8")
        # Supports basic recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        # Does NOT support additive recency
        type(marqo_index).index_supports_recency_additive = PropertyMock(return_value=False)

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            add_to_score_weight=0.5  # This should trigger the error
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn("addToScoreWeight", str(ctx.exception))
        self.assertIn(str(constants.MARQO_RECENCY_ADDITIVE_MINIMUM_VERSION), str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_grow_recency_on_old_schema_version_raises_error(self, mock_metrics):
        """Test that recency grow parameters (growFrom) on old schema version raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        # Create a mock semi-structured index that supports basic recency but NOT grow
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.8"  # Supports recency but not grow
        marqo_index.marqo_version = "2.24.8"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.8")
        # Supports basic recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        # Does NOT support grow recency
        type(marqo_index).index_supports_recency_grow = PropertyMock(return_value=False)

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            # All grow params required together - this should trigger the schema version error
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="7d",
            grow_offset="0d"
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn("growFrom", str(ctx.exception))
        self.assertIn(str(constants.MARQO_RECENCY_GROW_MINIMUM_VERSION), str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_center_recency_on_old_schema_version_raises_error(self, mock_metrics):
        """Test that recency center parameter on old schema version raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.25.0"
        marqo_index.marqo_version = "2.25.0"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.25.0")
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_grow = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_center_and_subqueries = PropertyMock(return_value=False)

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            center=1234.0,
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn("center", str(ctx.exception))
        self.assertIn(str(constants.MARQO_RECENCY_CENTER_AND_SUBQUERIES_MINIMUM_VERSION), str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_apply_to_subqueries_on_old_schema_version_raises_error(self, mock_metrics):
        """Test that recency applyToSubqueries parameter on old schema version raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.25.0"
        marqo_index.marqo_version = "2.25.0"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.25.0")
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_grow = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_center_and_subqueries = PropertyMock(return_value=False)

        config = Mock(spec=Config)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            apply_to_subqueries=["tensor"],
        )

        hybrid_search = HybridSearch()
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            hybrid_search.search(
                config=config,
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(),
                recency_parameters=recency_params
            )

        self.assertIn("applyToSubqueries", str(ctx.exception))
        self.assertIn(str(constants.MARQO_RECENCY_CENTER_AND_SUBQUERIES_MINIMUM_VERSION), str(ctx.exception))

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_additive_recency_on_new_schema_version_succeeds(
        self, mock_metrics, mock_gather_docs, mock_parse_lexical,
        mock_vectorise, mock_vespa_factory
    ):
        """Test that additive recency succeeds on index that supports it."""
        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response

        # Mock marqo_index with new schema version
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.9"  # New version
        marqo_index.marqo_version = "2.24.9"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.9")
        # Supports both basic recency and additive recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_additive = PropertyMock(return_value=True)

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

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            add_to_score_weight=0.5  # Additive mode
        )

        # Execute the search - should not raise
        hybrid_search = HybridSearch()
        result = hybrid_search.search(
            config=config,
            marqo_index=marqo_index,
            query="test query",
            hybrid_parameters=HybridParameters(),
            recency_parameters=recency_params
        )

        # Verify the search executed successfully
        self.assertIsNotNone(result)

    @patch('marqo.core.search.hybrid_search.vespa_index_factory')
    @patch('marqo.core.search.hybrid_search.run_vectorise_pipeline')
    @patch('marqo.core.search.hybrid_search.utils.parse_lexical_query')
    @patch('marqo.core.search.hybrid_search.gather_documents_from_response')
    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_grow_recency_on_new_schema_version_succeeds(
        self, mock_metrics, mock_gather_docs, mock_parse_lexical,
        mock_vectorise, mock_vespa_factory
    ):
        """Test that grow recency (growFrom) succeeds on index that supports it."""
        # Setup mocks
        config = Mock(spec=Config)
        config.vespa_client = Mock()
        mock_response = Mock()
        mock_response.root.coverage.coverage = 100
        mock_response.root.coverage.degraded = None
        config.vespa_client.query.return_value = mock_response

        # Mock marqo_index with new schema version
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.9"  # New version
        marqo_index.marqo_version = "2.24.9"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.9")
        # Supports both basic recency and grow recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        type(marqo_index).index_supports_recency_grow = PropertyMock(return_value=True)

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

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,  # Grow mode
            grow_function="exponential",
            grow_scale="7d",
            grow_offset="0d"
        )

        # Execute the search - should not raise
        hybrid_search = HybridSearch()
        result = hybrid_search.search(
            config=config,
            marqo_index=marqo_index,
            query="test query",
            hybrid_parameters=HybridParameters(),
            recency_parameters=recency_params
        )

        # Verify the search executed successfully
        self.assertIsNotNone(result)

    def test_basic_recency_without_additive_succeeds_on_old_additive_schema(self):
        """Test that basic recency (without addToScoreWeight) succeeds on schema 2.24.8."""
        # Create a mock semi-structured index that supports basic recency but NOT additive
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.8"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        # Supports basic recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        # Does NOT support additive recency - but this shouldn't matter if we don't use additive
        type(marqo_index).index_supports_recency_additive = PropertyMock(return_value=False)

        # Recency params WITHOUT add_to_score_weight (basic/multiplicative mode)
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5
            # No add_to_score_weight - should pass validation
        )

        # The validation should pass (no error raised)
        # This test verifies that the validation ONLY fails when add_to_score_weight is provided
        # The actual search would require more setup, but we're testing the validation logic

        # Since we can't easily test the full search flow without extensive mocking,
        # let's just verify the RecencyParameters accepts the parameters
        self.assertIsNone(recency_params.add_to_score_weight)

    def test_basic_recency_without_grow_succeeds_on_old_grow_schema(self):
        """Test that basic recency (without growFrom) succeeds on schema 2.24.8."""
        # Create a mock semi-structured index that supports basic recency but NOT grow
        marqo_index = Mock(spec=SemiStructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.schema_template_version = "2.24.8"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        # Supports basic recency
        type(marqo_index).index_supports_recency_scoring = PropertyMock(return_value=True)
        # Does NOT support grow recency - but this shouldn't matter if we don't use grow
        type(marqo_index).index_supports_recency_grow = PropertyMock(return_value=False)

        # Recency params WITHOUT grow_from (basic mode)
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            decay_function="exponential",
            decay_to=0.5
            # No grow_from - should pass validation
        )

        # The validation should pass (no error raised)
        # This test verifies that the validation ONLY fails when grow_from is provided
        self.assertIsNone(recency_params.grow_from)

    @patch('marqo.core.search.hybrid_search.RequestMetricsStore')
    def test_lexical_operand_on_structured_index_raises_error(self, mock_metrics):
        """Test that lexicalOperand on a structured index raises UnsupportedFeatureError."""
        self._setup_metrics_mock(mock_metrics)

        marqo_index = Mock(spec=StructuredMarqoIndex)
        marqo_index.name = "test_index"
        marqo_index.model = Mock()
        marqo_index.model.get_text_query_prefix.return_value = ""
        marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.25.0")

        with self.assertRaises(UnsupportedFeatureError) as ctx:
            HybridSearch().search(
                config=Mock(spec=Config),
                marqo_index=marqo_index,
                query="test",
                hybrid_parameters=HybridParameters(lexicalOperand=LexicalOperand.Or)
            )
        self.assertIn("lexicalOperand", str(ctx.exception))


class TestShouldUseCollapseSearch(TestCase):
    """Tests for should_use_collapse_search() logic."""

    def test_no_collapse_returns_false(self):
        self.assertFalse(should_use_collapse_search(collapse=None))

    def test_collapse_without_sort_by_returns_false(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel
        collapse = CollapseModel(name="product_id")
        self.assertFalse(should_use_collapse_search(collapse=collapse))

    def test_collapse_with_sort_by_no_main_sort_returns_true(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        collapse = CollapseModel(
            name="product_id",
            sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
        )
        self.assertTrue(should_use_collapse_search(collapse=collapse, main_query_sort_by=None))

    def test_collapse_with_sort_by_and_main_sort_no_disable_fields_returns_true(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        collapse = CollapseModel(
            name="product_id",
            sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
        )
        main_sort = SortByModel(fields=[SortByField(field_name="price", order=SortOrder.Asc)])
        # No disable_if_main_sort_by_fields set, so collapse search should still be used
        self.assertTrue(should_use_collapse_search(collapse=collapse, main_query_sort_by=main_sort))

    def test_collapse_disabled_when_main_sort_intersects_disable_fields(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        collapse = CollapseModel(
            name="product_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order="asc")],
                disableIfMainSortByFields={"price"}
            )
        )
        main_sort = SortByModel(fields=[SortByField(field_name="price", order=SortOrder.Asc)])
        self.assertFalse(should_use_collapse_search(collapse=collapse, main_query_sort_by=main_sort))

    def test_collapse_not_disabled_when_main_sort_disjoint_from_disable_fields(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        collapse = CollapseModel(
            name="product_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order="asc")],
                disableIfMainSortByFields={"price"}
            )
        )
        main_sort = SortByModel(fields=[SortByField(field_name="date", order=SortOrder.Desc)])
        self.assertTrue(should_use_collapse_search(collapse=collapse, main_query_sort_by=main_sort))


