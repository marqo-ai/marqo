from unittest import TestCase

from pydantic.v1 import ValidationError

from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffMethod,
    RelativeMaxScoreParameters,
    MeanStdParameters,
    RelevanceCutoffModel
)


class TestRelevanceCutoffModel(TestCase):

    def test_relative_max_score_valid(self):
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.75)
        m = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params
        )
        self.assertEqual(m.method, RelevanceCutoffMethod.RelativeMaxScore)
        self.assertIsInstance(m.parameters, RelativeMaxScoreParameters)
        self.assertEqual(m.probe_depth, 1000)

    def test_relative_max_score_missing_parameters(self):
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(method=RelevanceCutoffMethod.RelativeMaxScore)
        self.assertIn("relativeScoreFactor", str(cm.exception))

    def test_relative_max_score_wrong_parameter_type(self):
        bad_params = MeanStdParameters(stdDevFactor=1.2)
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=bad_params
            )
        self.assertIn("relativeScoreFactor", str(cm.exception))

    def test_mean_std_dev_valid(self):
        params = MeanStdParameters(stdDevFactor=2.5)
        m = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params
        )
        self.assertEqual(m.method, RelevanceCutoffMethod.MeanStdDev)
        self.assertIsInstance(m.parameters, MeanStdParameters)

    def test_mean_std_dev_missing_parameters(self):
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(method=RelevanceCutoffMethod.MeanStdDev)
        self.assertIn("stdDevFactor", str(cm.exception))

    def test_mean_std_dev_wrong_parameter_type(self):
        bad_params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.MeanStdDev,
                parameters=bad_params
            )
        self.assertIn("stdDevFactor", str(cm.exception))

    def test_gap_detection_valid(self):
        # no parameters required
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        self.assertEqual(m.method, RelevanceCutoffMethod.GapDetection)
        self.assertIsNone(m.parameters)

    def test_gap_detection_with_parameters(self):
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.3)
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                parameters=params
            )
        self.assertIn("does not require any parameters", str(cm.exception))

    def test_probe_depth_validation(self):
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        with self.assertRaises(ValidationError):
            # probeDepth must be >= 1
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=params,
                probeDepth=0
            )

    def test_relative_score_parameter_constraints(self):
        # relativeScoreFactor must be >0 and <=1
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=0)
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=1.5)

    def test_std_dev_parameter_constraints(self):
        # stdDevFactor must be >0
        with self.assertRaises(ValidationError):
            MeanStdParameters(stdDevFactor=0)