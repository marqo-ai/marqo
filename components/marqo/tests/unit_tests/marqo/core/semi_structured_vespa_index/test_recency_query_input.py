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
        """Test that all 8 expected query input constants are present."""
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
            constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT,
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
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 604800.0,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 0.0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.5,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"created_at": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: 0.0,
                    constants.QUERY_INPUT_RECENCY_CENTER_SECONDS: 0,
                    # Grow parameters (disabled by default)
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 604800.0,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
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
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 1209600.0,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 86400.0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.3,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"updated_at": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 1,
                    constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: 0.0,
                    constants.QUERY_INPUT_RECENCY_CENTER_SECONDS: 0,
                    # Grow parameters (disabled by default)
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 1209600.0,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
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
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 86400.0,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 43200.0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.75,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"publish_date": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 2,
                    constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: 0.0,
                    constants.QUERY_INPUT_RECENCY_CENTER_SECONDS: 0,
                    # Grow parameters (disabled by default)
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 86400.0,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
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
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 86400.0,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 0.0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.01,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"event_time": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 3,
                    constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: 0.0,
                    constants.QUERY_INPUT_RECENCY_CENTER_SECONDS: 0,
                    # Grow parameters (disabled by default)
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 86400.0,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
                }
            ),
            (
                "with_add_to_score_weight",
                {
                    "recency_field": "created_at",
                    "decay_function": "exponential",
                    "scale": "7d",
                    "offset": "0d",
                    "decay_to": 0.5,
                    "apply_in_ranking_phase": "all",
                    "add_to_score_weight": 0.5
                },
                {
                    constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 1,
                    constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: 604800.0,
                    constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: 0.0,
                    constants.QUERY_INPUT_RECENCY_DECAY_TO: 0.5,
                    constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {"created_at": 1.0},
                    constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: 0.5,
                    constants.QUERY_INPUT_RECENCY_CENTER_SECONDS: 0,
                    # Grow parameters (disabled by default)
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 604800.0,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
                }
            ),
        ]

        for test_name, input_params, expected_output in test_cases:
            with self.subTest(test_name):
                params = RecencyParameters(**input_params)
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(result, expected_output)

    def test_add_to_score_weight_defaults_to_zero(self):
        """Test add_to_score_weight defaults to 0.0 when not provided."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT],
            0.0
        )

    def test_add_to_score_weight_passed_correctly(self):
        """Test add_to_score_weight is passed correctly when provided."""
        weight_values = [0.1, 0.5, 1.0, 10.0]

        for weight in weight_values:
            with self.subTest(weight=weight):
                params = RecencyParameters(
                    recency_field="created_at",
                    add_to_score_weight=weight
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT],
                    weight
                )

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

                # Stub the base query so we don't run index-dependent code (version checks, etc.).
                # The test only cares that _to_vespa_hybrid_query sets recency params on the result.
                with patch.object(
                    self.vespa_index,
                    '_get_base_vespa_hybrid_query',
                    return_value={'query_features': {}},
                ):
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

    # ============= Grow Parameters Tests =============

    def test_grow_parameters_disabled_by_default(self):
        """Test grow parameters are disabled when grow_from is not specified."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_GROW_ENABLED],
            0
        )

    def test_grow_parameters_enabled_when_grow_from_specified(self):
        """Test grow parameters are enabled when all grow params are specified."""
        params = RecencyParameters(
            recency_field="created_at",
            grow_from=0.5,
            grow_function="exponential",
            grow_scale="7d",
            grow_offset="0d"
        )
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_GROW_ENABLED],
            1
        )
        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_GROW_FROM],
            0.5
        )

    def test_grow_from_values(self):
        """Test grow_from values are passed correctly."""
        grow_from_values = [0.01, 0.3, 0.5, 0.75, 0.99, 1.0]

        for grow_from in grow_from_values:
            with self.subTest(grow_from=grow_from):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=grow_from,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset="0d"
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_GROW_FROM],
                    grow_from
                )

    def test_grow_function_explicit_mapping(self):
        """Test grow_function to numeric mapping when explicitly specified."""
        # All grow params must be provided together
        grow_functions = [
            ("exponential", 0),
            ("linear", 1),
            ("gaussian", 2),
            ("binary", 3),
        ]

        for func_name, expected_code in grow_functions:
            with self.subTest(func_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    decay_function="exponential",
                    grow_from=0.5,
                    grow_function=func_name,
                    grow_scale="7d",
                    grow_offset="0d"
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE],
                    expected_code
                )

    def test_grow_scale_to_seconds_conversion(self):
        """Test grow_scale conversion to seconds when specified."""
        duration_cases = [
            ("7d", 604800),
            ("1d", 86400),
            ("24h", 86400),
            ("14d", 1209600),
            ("1.5d", 129600),
            ("0.5h", 1800),
        ]

        for grow_scale, expected_seconds in duration_cases:
            with self.subTest(grow_scale=grow_scale):
                params = RecencyParameters(
                    recency_field="created_at",
                    scale="7d",
                    grow_from=0.5,
                    grow_function="exponential",
                    grow_scale=grow_scale,
                    grow_offset="0d"
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS],
                    expected_seconds
                )

    def test_grow_offset_to_seconds_conversion(self):
        """Test grow_offset conversion to seconds when specified."""
        duration_cases = [
            ("0d", 0),
            ("1d", 86400),
            ("12h", 43200),
            ("2d", 172800),
            ("0.5d", 43200),
        ]

        for grow_offset, expected_seconds in duration_cases:
            with self.subTest(grow_offset=grow_offset):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=0.5,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset=grow_offset
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS],
                    expected_seconds
                )

    def test_all_grow_query_input_constants_present(self):
        """Test that all grow query input constants are present when grow is enabled."""
        params = RecencyParameters(
            recency_field="created_at",
            grow_from=0.5,
            grow_function="exponential",
            grow_scale="7d",
            grow_offset="0d"
        )
        result = self.vespa_index._get_recency_query_input(params)

        expected_grow_keys = [
            constants.QUERY_INPUT_RECENCY_GROW_ENABLED,
            constants.QUERY_INPUT_RECENCY_GROW_FROM,
            constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE,
            constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS,
            constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS,
        ]

        for key in expected_grow_keys:
            with self.subTest(key=key):
                self.assertIn(key, result)

    def test_complete_grow_parameter_combination(self):
        """Test complete grow parameter combination produces expected output."""
        test_cases = [
            (
                "grow_with_all_params",
                {
                    "recency_field": "created_at",
                    "decay_function": "exponential",
                    "scale": "7d",
                    "offset": "0d",
                    "decay_to": 0.5,
                    "grow_from": 0.3,
                    "grow_function": "linear",
                    "grow_scale": "14d",
                    "grow_offset": "1d"
                },
                {
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 1,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 0.3,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 1,  # linear
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 1209600,  # 14d
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 86400,  # 1d
                }
            ),
            (
                "grow_with_gaussian",
                {
                    "recency_field": "created_at",
                    "decay_function": "gaussian",
                    "scale": "24h",
                    "grow_from": 0.5,
                    "grow_function": "gaussian",
                    "grow_scale": "24h",
                    "grow_offset": "0d"
                },
                {
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 1,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 0.5,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 2,  # gaussian
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 86400,  # 24h
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
                }
            ),
            (
                "grow_disabled",
                {
                    "recency_field": "created_at",
                    "decay_function": "exponential",
                    "scale": "7d",
                    # No grow params specified - grow disabled
                },
                {
                    constants.QUERY_INPUT_RECENCY_GROW_ENABLED: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_FROM: 1.0,
                    constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE: 0,
                    constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS: 604800,
                    constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS: 0,
                }
            ),
        ]

        for test_name, input_params, expected_grow_output in test_cases:
            with self.subTest(test_name):
                params = RecencyParameters(**input_params)
                result = self.vespa_index._get_recency_query_input(params)

                for key, expected_value in expected_grow_output.items():
                    self.assertEqual(
                        result[key],
                        expected_value,
                        f"Key {key}: expected {expected_value}, got {result.get(key)}"
                    )


    # ============= Center Parameter Tests =============

    def test_center_default_zero(self):
        """Test center defaults to 0 (sentinel for 'use now()') when None."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        self.assertEqual(
            result[constants.QUERY_INPUT_RECENCY_CENTER_SECONDS],
            0
        )

    def test_center_passed_correctly(self):
        """Test center values are passed through correctly."""
        center_values = [1709232000.0, 0, 1000000000, 9999999999.9]

        for center in center_values:
            with self.subTest(center=center):
                params = RecencyParameters(
                    recency_field="created_at",
                    center=center
                )
                result = self.vespa_index._get_recency_query_input(params)

                self.assertEqual(
                    result[constants.QUERY_INPUT_RECENCY_CENTER_SECONDS],
                    center
                )

    def test_center_in_all_query_constants(self):
        """Test that center_seconds constant is present in output."""
        params = RecencyParameters(recency_field="created_at")
        result = self.vespa_index._get_recency_query_input(params)

        self.assertIn(constants.QUERY_INPUT_RECENCY_CENTER_SECONDS, result)

    # ============= ApplyToSubqueries Parameter Tests =============

    def test_apply_to_subqueries_default(self):
        """Test apply_to_subqueries default (None) sets both flags to True as top-level query properties."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
        )

        query = self._create_hybrid_query_with_recency("all")
        # Override to use default apply_to_subqueries (None)
        query.recency_parameters = recency_params

        with patch.object(self.vespa_index, '_get_base_vespa_hybrid_query', return_value={'query_features': {}}):
            result = self.vespa_index._to_vespa_hybrid_query(query)

        # Flags should be top-level query properties, not in query_features
        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR], True)
        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL], True)
        self.assertNotIn(constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR, result['query_features'])
        self.assertNotIn(constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL, result['query_features'])

    def test_apply_to_subqueries_tensor_only(self):
        """Test apply_to_subqueries=['tensor'] sets only tensor flag to True as top-level query properties."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            apply_to_subqueries=["tensor"]
        )

        query = self._create_hybrid_query_with_recency("all")
        query.recency_parameters = recency_params

        with patch.object(self.vespa_index, '_get_base_vespa_hybrid_query', return_value={'query_features': {}}):
            result = self.vespa_index._to_vespa_hybrid_query(query)

        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR], True)
        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL], False)

    def test_apply_to_subqueries_lexical_only(self):
        """Test apply_to_subqueries=['lexical'] sets only lexical flag to True as top-level query properties."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            apply_to_subqueries=["lexical"]
        )

        query = self._create_hybrid_query_with_recency("all")
        query.recency_parameters = recency_params

        with patch.object(self.vespa_index, '_get_base_vespa_hybrid_query', return_value={'query_features': {}}):
            result = self.vespa_index._to_vespa_hybrid_query(query)

        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR], False)
        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL], True)

    def test_apply_to_subqueries_empty_list(self):
        """Test apply_to_subqueries=[] sets both flags to False as top-level query properties."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            apply_to_subqueries=[]
        )

        query = self._create_hybrid_query_with_recency("all")
        query.recency_parameters = recency_params

        with patch.object(self.vespa_index, '_get_base_vespa_hybrid_query', return_value={'query_features': {}}):
            result = self.vespa_index._to_vespa_hybrid_query(query)

        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR], False)
        self.assertEqual(result[constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL], False)


if __name__ == '__main__':
    unittest.main()
