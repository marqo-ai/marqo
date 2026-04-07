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
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=params,
                probeDepth=0
            )

    def test_relative_score_parameter_constraints(self):
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=-1)
        with self.assertRaises(ValidationError):
            RelativeMaxScoreParameters(relativeScoreFactor=1.5)

    def test_std_dev_parameter_constraints(self):
        with self.assertRaises(ValidationError):
            MeanStdParameters(stdDevFactor="test")

    def test_relevance_cutoff_rejected_for_non_hybrid_search_methods(self):
        """relevanceCutoff is only valid for HYBRID search."""
        relevance_cutoff = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        for search_method in [SearchMethod.TENSOR, SearchMethod.LEXICAL]:
            with self.subTest(searchMethod=search_method):
                with self.assertRaises(ValidationError) as cm:
                    SearchQuery(
                        q="test query",
                        searchMethod=search_method,
                        relevanceCutoff=relevance_cutoff
                    )
                self.assertIn("relevanceCutoff can only be provided for 'HYBRID' search", str(cm.exception))
                self.assertIn(search_method, str(cm.exception))

    def test_lexical_operand_valid_values(self):
        for operand in ['or', 'and', 'weakAnd']:
            with self.subTest(operand=operand):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.GapDetection,
                    lexicalOperand=operand
                )
                self.assertEqual(m.lexical_operand, operand)

    def test_lexical_operand_none_by_default(self):
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        self.assertIsNone(m.lexical_operand)

    def test_lexical_operand_invalid_value(self):
        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                lexicalOperand="invalid"
            )

    def test_apply_in_retrieval_valid_values(self):
        for value in ['tensor', 'both']:
            with self.subTest(value=value):
                m = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.GapDetection,
                    applyInRetrieval=value
                )
                self.assertEqual(m.apply_in_retrieval, value)

    def test_apply_in_retrieval_both_by_default(self):
        m = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        self.assertEqual(m.apply_in_retrieval, ApplyInRetrieval.Both)

    def test_apply_in_retrieval_invalid_value(self):
        with self.assertRaises(ValidationError):
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                applyInRetrieval="invalid"
            )

    def test_apply_in_retrieval_lexical_not_supported(self):
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                applyInRetrieval='lexical'
            )
        self.assertIn("not currently supported", str(cm.exception))

    def test_apply_in_retrieval_tensor_requires_disjunction_retrieval_method(self):
        """applyInRetrieval='tensor' is rejected when retrievalMethod is not disjunction."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval='tensor'
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

    def test_apply_in_retrieval_tensor_accepted_with_disjunction(self):
        """applyInRetrieval='tensor' is accepted when retrievalMethod is disjunction."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval='tensor'
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
        self.assertEqual(sq.relevance_cutoff.apply_in_retrieval, 'tensor')

    def test_apply_in_retrieval_both_default_accepted_with_any_retrieval_method(self):
        """applyInRetrieval defaults to 'both', which is accepted for any retrievalMethod."""
        relevance_cutoff = RelevanceCutoffModel(method=RelevanceCutoffMethod.GapDetection)
        cases = [
            (RetrievalMethod.Disjunction, 'rrf'),
            (RetrievalMethod.Tensor, 'tensor'),
            (RetrievalMethod.Lexical, 'lexical'),
        ]
        for retrieval_method, ranking_method in cases:
            with self.subTest(retrievalMethod=retrieval_method):
                sq = SearchQuery(
                    q="test query",
                    searchMethod=SearchMethod.HYBRID,
                    relevanceCutoff=relevance_cutoff,
                    hybridParameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method
                    )
                )
                self.assertEqual(sq.relevance_cutoff.apply_in_retrieval, ApplyInRetrieval.Both)

    def test_apply_in_retrieval_incompatible_with_override_sort_candidates(self):
        """applyInRetrieval='tensor' cannot be combined with overrideSortCandidatesWithRelevantCandidates."""
        with self.assertRaises(ValidationError) as cm:
            RelevanceCutoffModel(
                method=RelevanceCutoffMethod.GapDetection,
                applyInRetrieval='tensor',
                overrideSortCandidatesWithRelevantCandidates=True
            )
        self.assertIn("applyInRetrieval", str(cm.exception))

    def test_apply_in_retrieval_both_compatible_with_override_sort_candidates(self):
        """applyInRetrieval='both' (explicit or default) is compatible with overrideSortCandidatesWithRelevantCandidates."""
        for apply_in_retrieval in ['both', None]:
            with self.subTest(applyInRetrieval=apply_in_retrieval):
                kwargs = dict(
                    method=RelevanceCutoffMethod.GapDetection,
                    overrideSortCandidatesWithRelevantCandidates=True
                )
                if apply_in_retrieval is not None:
                    kwargs['applyInRetrieval'] = apply_in_retrieval
                m = RelevanceCutoffModel(**kwargs)
                self.assertEqual(m.apply_in_retrieval, ApplyInRetrieval.Both)
                self.assertTrue(m.override_sort_candidates_with_relevant_candidates)
