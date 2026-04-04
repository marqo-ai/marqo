from unittest import TestCase

from pydantic.v1 import ValidationError

from marqo.core.models.hybrid_parameters import LexicalOperand, HybridParameters, RetrievalMethod
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

    def test_lexical_operand_valid_values(self):
        """Test that valid lexicalOperand values are accepted."""
        for operand in ['or', 'and', 'weakAnd']:
            with self.subTest(operand=operand):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.GapDetection,
                    lexicalOperand=operand
                )
                self.assertEqual(m.lexical_operand, operand)

    def test_lexical_operand_none_by_default(self):
        """Test that lexicalOperand defaults to None."""
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        self.assertIsNone(m.lexical_operand)

    def test_lexical_operand_invalid_value(self):
        """Test that invalid lexicalOperand values are rejected."""
        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                lexicalOperand="invalid"
            )

    def test_apply_in_retrieval_valid_values(self):
        """Test that valid applyInRetrieval values are accepted."""
        for value in ['lexical', 'tensor', 'both']:
            with self.subTest(value=value):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.GapDetection,
                    applyInRetrieval=value
                )
                self.assertEqual(m.apply_in_retrieval, value)

    def test_apply_in_retrieval_none_by_default(self):
        """Test that applyInRetrieval defaults to None."""
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        self.assertIsNone(m.apply_in_retrieval)

    def test_apply_in_retrieval_invalid_value(self):
        """Test that invalid applyInRetrieval values are rejected."""
        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                applyInRetrieval="invalid"
            )

    def test_apply_in_retrieval_requires_disjunction_retrieval_method(self):
        """Test that applyInRetrieval is rejected when retrievalMethod is not disjunction."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval='lexical'
        )

        for retrieval_method in [RetrievalMethod.Tensor, RetrievalMethod.Lexical]:
            with self.subTest(retrievalMethod=retrieval_method):
                with self.assertRaises(ValidationError) as cm:
                    SearchQuery(
                        q="test query",
                        searchMethod=SearchMethod.HYBRID,
                        relevanceCutoff=relevance_cutoff,
                        hybridParameters=HybridParameters(
                            retrievalMethod=retrieval_method,
                            rankingMethod='tensor'
                        )
                    )
                self.assertIn("applyInRetrieval", str(cm.exception))

    def test_apply_in_retrieval_accepted_with_disjunction(self):
        """Test that applyInRetrieval is accepted when retrievalMethod is disjunction."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval='lexical'
        )
        sq = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.HYBRID,
            relevanceCutoff=relevance_cutoff,
            hybridParameters=HybridParameters(
                retrievalMethod='disjunction',
                rankingMethod='rrf'
            )
        )
        self.assertEqual(sq.relevance_cutoff.apply_in_retrieval, 'lexical')

    def test_apply_in_retrieval_none_accepted_with_any_retrieval_method(self):
        """Test that applyInRetrieval=None works with any retrieval method."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection
        )
        sq = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.HYBRID,
            relevanceCutoff=relevance_cutoff,
            hybridParameters=HybridParameters(
                retrievalMethod='disjunction',
                rankingMethod='rrf'
            )
        )
        self.assertIsNone(sq.relevance_cutoff.apply_in_retrieval)

    def test_apply_in_retrieval_incompatible_with_override_sort_candidates(self):
        """applyInRetrieval targeting a specific leg cannot be combined with
        overrideSortCandidatesWithRelevantCandidates."""
        for value in ['lexical', 'tensor']:
            with self.subTest(applyInRetrieval=value):
                with self.assertRaises(ValidationError) as cm:
                    RelevanceCutoffModel(
                        method=RelevanceCutoffMethod.GapDetection,
                        applyInRetrieval=value,
                        overrideSortCandidatesWithRelevantCandidates=True
                    )
                self.assertIn("applyInRetrieval", str(cm.exception))

    def test_apply_in_retrieval_both_allowed_with_override_sort_candidates(self):
        """applyInRetrieval='both' is compatible with overrideSortCandidatesWithRelevantCandidates."""
        m = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval='both',
            overrideSortCandidatesWithRelevantCandidates=True
        )
        self.assertEqual(m.apply_in_retrieval, 'both')
        self.assertTrue(m.override_sort_candidates_with_relevant_candidates)

    def test_apply_in_retrieval_none_allowed_with_override_sort_candidates(self):
        """applyInRetrieval=None (default) is compatible with overrideSortCandidatesWithRelevantCandidates."""
        m = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            overrideSortCandidatesWithRelevantCandidates=True
        )
        self.assertIsNone(m.apply_in_retrieval)
        self.assertTrue(m.override_sort_candidates_with_relevant_candidates)