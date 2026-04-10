from unittest import TestCase

from pydantic.v1 import ValidationError

from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffMethod,
    RelativeMaxScoreParameters,
    MeanStdParameters,
    RelevanceCutoffModel,
    ApplyInRetrieval
)
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.enums import SearchMethod


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
        # relativeScoreFactor must be >=0 and <=1
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=-1)
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=1.5)

    def test_std_dev_parameter_constraints(self):
        # stdDevFactor must be a numeric value
        with self.assertRaises(ValidationError):
            MeanStdParameters(stdDevFactor="test")

    def test_relevance_cutoff_with_tensor_search_fails(self):
        """Test that relevance cutoff fails with TENSOR search method"""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.75)
        )

        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test query",
                searchMethod=SearchMethod.TENSOR,
                relevanceCutoff=relevance_cutoff
            )

        self.assertIn("relevanceCutoff can only be provided for 'HYBRID' search", str(cm.exception))
        self.assertIn("TENSOR", str(cm.exception))

    def test_relevance_cutoff_with_lexical_search_fails(self):
        """Test that relevance cutoff fails with LEXICAL search method"""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection
        )

        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test query",
                searchMethod=SearchMethod.LEXICAL,
                relevanceCutoff=relevance_cutoff
            )

        self.assertIn("relevanceCutoff can only be provided for 'HYBRID' search", str(cm.exception))
        self.assertIn("LEXICAL", str(cm.exception))

    def test_lexical_operand(self):
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.RelativeMaxScore, parameters=params)
        self.assertIsNone(m.lexical_operand)

        for operand in ['or', 'and', 'weakAnd']:
            with self.subTest(operand=operand):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.RelativeMaxScore,
                    parameters=params,
                    lexicalOperand=operand
                )
                self.assertEqual(m.lexical_operand, operand)

        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=params,
                lexicalOperand="invalid"
            )

    def test_apply_in_retrieval_model_validation(self):
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)

        # Default is None
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.RelativeMaxScore, parameters=params)
        self.assertIsNone(m.apply_in_retrieval)

        # Valid values
        for value in ['tensor', 'both']:
            with self.subTest(value=value):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.RelativeMaxScore,
                    parameters=params,
                    applyInRetrieval=value
                )
                self.assertEqual(m.apply_in_retrieval, value)

        # Invalid value
        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=params,
                applyInRetrieval="invalid"
            )

        # 'lexical' is recognised as a valid enum value but explicitly blocked
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=params,
                applyInRetrieval='lexical'
            )
        self.assertIn("not currently supported", str(cm.exception))

    def test_apply_in_retrieval_requires_disjunction_when_set(self):
        """applyInRetrieval is rejected when explicitly set and retrievalMethod is not disjunction."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        for value, retrieval_method, ranking_method in [
            ('tensor', RetrievalMethod.Tensor, 'tensor'),
            ('tensor', RetrievalMethod.Lexical, 'lexical'),
            ('both', RetrievalMethod.Tensor, 'tensor'),
            ('both', RetrievalMethod.Lexical, 'lexical'),
        ]:
            with self.subTest(applyInRetrieval=value, retrievalMethod=retrieval_method):
                with self.assertRaises(ValidationError) as cm:
                    SearchQuery(
                        q="test query",
                        searchMethod=SearchMethod.HYBRID,
                        relevanceCutoff=RelevanceCutoffModel(
                            method=RelevanceCutoffMethod.RelativeMaxScore,
                            parameters=params,
                            applyInRetrieval=value
                        ),
                        hybridParameters=HybridParameters(
                            retrievalMethod=retrieval_method,
                            rankingMethod=ranking_method
                        )
                    )
                self.assertIn("applyInRetrieval", str(cm.exception))

    def test_apply_in_retrieval_accepted_with_disjunction(self):
        """applyInRetrieval is accepted when retrievalMethod is disjunction."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        for value in ['tensor', 'both']:
            with self.subTest(applyInRetrieval=value):
                sq = SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.HYBRID,
                    relevanceCutoff=RelevanceCutoffModel(
                        method=RelevanceCutoffMethod.RelativeMaxScore,
                        parameters=params,
                        applyInRetrieval=value
                    ),
                    hybridParameters=HybridParameters(
                        retrievalMethod='disjunction',
                        rankingMethod='rrf'
                    )
                )
                self.assertEqual(value, sq.relevance_cutoff.apply_in_retrieval)

    def test_apply_in_retrieval_default_none_accepted_with_any_retrieval_method(self):
        """When applyInRetrieval is not set, relevanceCutoff works with any retrievalMethod.
        For disjunction it resolves to 'both'; for others it stays None (not applicable)."""
        cases = [
            (RetrievalMethod.Disjunction, 'rrf', ApplyInRetrieval.Both),
            (RetrievalMethod.Tensor, 'tensor', None),
            (RetrievalMethod.Lexical, 'lexical', None),
        ]
        for retrieval_method, ranking_method, expected in cases:
            with self.subTest(retrievalMethod=retrieval_method):
                relevance_cutoff = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.RelativeMaxScore,
                    parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5)
                )
                sq = SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.HYBRID,
                    relevanceCutoff=relevance_cutoff,
                    hybridParameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method
                    )
                )
                self.assertEqual(expected, sq.relevance_cutoff.apply_in_retrieval)

    def test_apply_in_retrieval_incompatible_with_override_sort_candidates(self):
        """applyInRetrieval='tensor' cannot be combined with overrideSortCandidatesWithRelevantCandidates."""
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5),
                applyInRetrieval='tensor',
                overrideSortCandidatesWithRelevantCandidates=True
            )
        self.assertIn("applyInRetrieval", str(cm.exception))

    def test_apply_in_retrieval_both_compatible_with_override_sort_candidates(self):
        """applyInRetrieval='both' (explicit or default None) is compatible with overrideSortCandidatesWithRelevantCandidates."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.5)
        for apply_in_retrieval in ['both', None]:
            with self.subTest(applyInRetrieval=apply_in_retrieval):
                kwargs = dict(
                    method=RelevanceCutoffMethod.RelativeMaxScore,
                    parameters=params,
                    overrideSortCandidatesWithRelevantCandidates=True
                )
                if apply_in_retrieval is not None:
                    kwargs['applyInRetrieval'] = apply_in_retrieval
                m = RelevanceCutoffModel(**kwargs)
                self.assertTrue(m.override_sort_candidates_with_relevant_candidates)
