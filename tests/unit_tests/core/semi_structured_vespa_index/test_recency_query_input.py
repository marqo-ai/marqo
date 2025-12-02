"""Unit tests for _get_recency_query_input() method in SemiStructuredVespaIndex."""
import unittest
from unittest.mock import MagicMock, patch

from marqo.core import constants
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.tensor_search.models.recency_parameters import RecencyParameters


class TestRecencyQueryInput(unittest.TestCase):
    """Tests for _get_recency_query_input() method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal mock index
        self.mock_index = MagicMock(spec=SemiStructuredMarqoIndex)
        self.mock_index.schema_name = "test_schema"
        self.vespa_index = SemiStructuredVespaIndex(self.mock_index)

    def test_decay_function_mapping(self):
        """Test decay function to numeric mapping."""
        decay_functions = [
            ("exponential", 0),
            ("linear", 1),
            ("gaussian", 2),
            ("binary", 3),
        ]

        for function_name, expected_code in decay_functions:
            with self.subTest(function_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    decay_function=function_name
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE],
                    expected_code
                )

    def test_duration_to_seconds_conversion(self):
        """Test conversion of duration strings to seconds."""
        duration_cases = [
            # (scale, offset, expected_scale_seconds, expected_offset_seconds)
            ("7d", "0d", 604800, 0),
            ("1d", "1d", 86400, 86400),
            ("14d", "2d", 1209600, 172800),
            ("24h", "12h", 86400, 43200),
            ("1.5d", "0.5d", 129600, 43200),
            ("0.5h", "0h", 1800, 0),
        ]

        for scale, offset, expected_scale, expected_offset in duration_cases:
            with self.subTest(scale=scale, offset=offset):
                params = RecencyParameters(
                    recency_field="created_at",
                    scale=scale,
                    offset=offset
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_SCALE_SECONDS],
                    expected_scale
                )
                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS],
                    expected_offset
                )

    def test_decay_to_parameter(self):
        """Test decay_to parameter is passed correctly."""
        decay_to_values = [0.01, 0.3, 0.5, 0.75, 0.99, 1.0]

        for decay_to in decay_to_values:
            with self.subTest(decay_to=decay_to):
                params = RecencyParameters(
                    recency_field="created_at",
                    decay_to=decay_to
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_DECAY_TO],
                    decay_to
                )

    def test_recency_field_to_timestamp_key(self):
        """Test recency_field is mapped to timestamp key correctly."""
        field_names = ["created_at", "updated_at", "publish_date", "custom_timestamp"]

        for field_name in field_names:
            with self.subTest(field_name=field_name):
                params = RecencyParameters(recency_field=field_name)
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY],
                    {field_name: 1.0}
                )

    def test_should_calculate_score_always_one(self):
        """Test should_calculate_score is always 1."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE],
            1
        )

    def test_apply_in_ranking_phase_all(self):
        """Test apply_in_ranking_phase='all' sets should_apply_score to 1."""
        params = RecencyParameters(
            recency_field="created_at",
            apply_in_ranking_phase="all"
        )
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE],
            1
        )

    def test_apply_in_ranking_phase_only_global(self):
        """Test apply_in_ranking_phase='only-global' sets should_apply_score to 0."""
        params = RecencyParameters(
            recency_field="created_at",
            apply_in_ranking_phase="only-global"
        )
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE],
            0
        )

    def test_apply_in_ranking_phase_exclude_global(self):
        """Test apply_in_ranking_phase='exclude-global' sets should_apply_score to 1."""
        params = RecencyParameters(
            recency_field="created_at",
            apply_in_ranking_phase="exclude-global"
        )
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE],
            1
        )

    def test_all_query_input_constants_present(self):
        """Test that all 7 expected query input constants are present."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        expected_keys = [
            constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE,
            constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE,
            constants.QUERY_INPUT_RECENCY_SCALE_SECONDS,
            constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS,
            constants.QUERY_INPUT_RECENCY_DECAY_TO,
            constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY,
            constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE,
        ]

        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, result)

    def test_complete_parameter_combination(self):
        """Test complete parameter combination produces expected output."""
        test_cases = [
            (
                "exponential_7d",
                {
                    "recency_field": "created_at",
                    "decay_function": "exponential",
                    "scale": "7d",
                    "offset": "0d",
                    "decay_to": 0.5,
                    "apply_in_ranking_phase": "all"
                },
                {
                    constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 604800,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.5,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"created_at": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 0,
                }
            ),
            (
                "linear_14d_with_offset",
                {
                    "recency_field": "updated_at",
                    "decay_function": "linear",
                    "scale": "14d",
                    "offset": "1d",
                    "decay_to": 0.3,
                    "apply_in_ranking_phase": "only-global"
                },
                {
                    constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 0,
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 1209600,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 86400,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.3,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"updated_at": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 1,
                }
            ),
            (
                "gaussian_24h",
                {
                    "recency_field": "publish_date",
                    "decay_function": "gaussian",
                    "scale": "24h",
                    "offset": "12h",
                    "decay_to": 0.75,
                    "apply_in_ranking_phase": "exclude-global"
                },
                {
                    constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 86400,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 43200,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.75,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"publish_date": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 2,
                }
            ),
            (
                "binary_1d",
                {
                    "recency_field": "event_time",
                    "decay_function": "binary",
                    "scale": "1d",
                    "offset": "0d",
                    "decay_to": 0.01,
                    "apply_in_ranking_phase": "all"
                },
                {
                    constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 86400,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.01,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"event_time": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 3,
                }
            ),
        ]

        for test_name, input_params, expected_output in test_cases:
            with self.subTest(test_name):
                params = RecencyParameters(**input_params)
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(result, expected_output)

    def test_global_phase_parameter_for_all_apply_modes(self):
        """Test marqo__recency_apply_in_global_ranking_phase is set correctly for all modes.

        This tests _to_vespa_hybrid_query() to verify the global phase parameter is set correctly.
        The bug was using 'exclude_global' (underscore) instead of 'exclude-global' (hyphen).
        """
        test_cases = [
            # (apply_in_ranking_phase, expected_global_phase_value)
            ("all", True),           # Apply in all phases including global
            ("only-global", True),   # Apply only in global phase
            ("exclude-global", False),  # Exclude from global phase
        ]

        for apply_mode, expected_global_phase in test_cases:
            with self.subTest(apply_mode=apply_mode):
                query = self._create_hybrid_query_with_recency(apply_mode)

                with patch('marqo.core.structured_vespa_index.structured_vespa_index.StructuredVespaIndex._to_vespa_hybrid_query') as mock_parent:
                    mock_parent.return_value = {'query_features': {}}
                    result = self.vespa_index._to_vespa_hybrid_query(query)

                # Verify recency is enabled
                self.assertTrue(result['marqo__recency_enabled'])

                # Verify global phase parameter is set correctly
                self.assertEqual(
                    result['marqo__recency_apply_in_global_ranking_phase'],
                    expected_global_phase,
                    f"apply_in_ranking_phase='{apply_mode}' should set "
                    f"marqo__recency_apply_in_global_ranking_phase to {expected_global_phase}"
                )

    def _create_hybrid_query_with_recency(self, apply_in_ranking_phase: str) -> MarqoHybridQuery:
        """Helper to create a MarqoHybridQuery with recency parameters."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            apply_in_ranking_phase=apply_in_ranking_phase
        )

        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF
        )

        query = MarqoHybridQuery(
            index_name="test_index",
            or_phrases=["test", "query"],
            and_phrases=[],
            vector_query=None,
            limit=10,
            offset=0,
            hybrid_parameters=hybrid_params,
            recency_parameters=recency_params
        )
        return query

if __name__ == '__main__':
    unittest.main()
