import pytest

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters, WeakAndParameters
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.collapse_model import CollapseModel
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierOperator
from tests.integ_tests.marqo_test import MarqoTestCase


@pytest.mark.skip_for_multinode
class TestSecondPhaseLexicalModifiers(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
    Note that these tests are skipped for multinode as the multinode setup as the rerankCount is applied to
    each content node separately.

    This test tests the following things:
    1. The current first phase lexical score modifier implementation works as expected, a relevant document without
          score modifier field is not returned when there are irrelevant documents with high score modifier values.
    2. The second phase lexical score modifier implementation works as expected, a relevant document without
            score modifier field is returned when there are irrelevant documents with high score modifier values if
            it is inside the rerankCount.
    3. However, if the rerankCount is too large, the relevant document can be squeezed out of the results.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.create_indexes([
            semi_structured_default_text_index,
        ])

        cls.index_name = semi_structured_default_text_index.name
        irrelevant_docs = [
            {
                '_id': f'{doc_id}',
                'text': 'Irrelevant documents but has a score modifier field.',
                "score_modifier_value": 100.0
            }
            for doc_id in range(10)
        ]

        relevant_docs = [
            {
                "_id": f'relevant_0',
                "text": "This is a relevant documents without score modifier.",
            }
        ]

        _ = cls.add_documents(
            config=cls.config, add_docs_params=AddDocsParams(
                index_name=cls.index_name,
                docs=irrelevant_docs + relevant_docs,
                tensor_fields=['text'],
            )
        )

    def setUp(self):
        # To override the default behavior of cleaning up indexes after each test
        self.assertEqual(11, self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents)

    def test_no_score_modifiers_returns_relevant_result_at_top(self):
        no_score_modifier_results = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
            ),
            result_count=10,
            offset=0
        )
        no_score_modifier_results_ids = [doc['_id'] for doc in no_score_modifier_results['hits']]
        # The relevant document should be in the results as there are no score modifiers
        self.assertEqual("relevant_0", no_score_modifier_results_ids[0])

    def test_first_phase_score_modifiers_exclude_result(self):
        first_phase_score_modifier_results = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                )
            ),
            result_count=10,
            offset=0
        )
        first_phase_score_modifier_results_ids = [doc['_id'] for doc in first_phase_score_modifier_results['hits']]
        # The relevant document should not be in the results as it has no score modifier field
        self.assertNotIn('relevant_0', first_phase_score_modifier_results_ids)

    def test_rerank_count_10_put_relevant_result_at_position_10(self):
        second_phase_score_modifier_results = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=True,
                rerankCount=10,
            ),
            result_count=10,
            offset=0
        )
        # The relevant document should be in the results as it is within the rerankCount
        second_phase_score_modifier_results_ids = [doc['_id'] for doc in second_phase_score_modifier_results['hits']]
        self.assertEqual("relevant_0", second_phase_score_modifier_results_ids[9])

    def test_rerank_count_too_large_squeezes_out_relevant_result(self):
        # Now test that if the relevant document can be squeezed if the rerankCount is too large
        second_phase_score_modifier_results_large_rerank_count = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=True,
                rerankCount=15,
            ),
            result_count=10,
            offset=0
        )

        second_phase_score_modifier_results_large_rerank_count_ids = [
            doc['_id'] for doc in second_phase_score_modifier_results_large_rerank_count['hits']
        ]
        # The relevant document should not be in the results as it is squeezed out
        self.assertNotIn("relevant_0", second_phase_score_modifier_results_large_rerank_count_ids)

    def test_rerank_1_makes_the_document_the_only_results(self):
        # Now test that if rerankCount is 1, the relevant document is the only result
        second_phase_score_modifier_results_rerank_1 = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=True,
                rerankCount=1,
            ),
            result_count=1,
            offset=0
        )

        second_phase_score_modifier_results_rerank_1_ids = [
            doc['_id'] for doc in second_phase_score_modifier_results_rerank_1['hits']
        ]
        # The relevant document should be the only result
        self.assertEqual("relevant_0", second_phase_score_modifier_results_rerank_1_ids[0])

    def test_second_phase_modifier_work_with_disjunction(self):
        """Ensure that second phase lexical modifiers work with disjunction retrieval method."""
        disjunction_result = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.01, # Set a low alpha to prioritise lexical scores
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=True,
                rerankCount=10,
            ),
            result_count=10,
            offset=0
        )

        ids = [doc['_id'] for doc in disjunction_result['hits']]
        self.assertEqual("relevant_0", ids[-1])


class TestUnsupportedScenarioForSecondPhaseLexicalModifiers(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        unstructured_index_with_collapse_field = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            collapse_fields=[CollapseField(name="category")],
        )

        cls.create_indexes([
            structured_default_text_index,
            unstructured_index_with_collapse_field,
        ])

        cls.structured_marqo_index = structured_default_text_index.name
        cls.unstructured_index_with_collapse_field = unstructured_index_with_collapse_field.name

    def test_structured_index_raises_error(self):
        with self.assertRaises(UnsupportedFeatureError) as cm:
            tensor_search.search(
                config=self.config,
                index_name=self.structured_marqo_index,
                text="test",
                search_method=SearchMethod.HYBRID,
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Lexical,
                    secondPhaseModifier=True,
                    rerankCount=10,
                ),
                result_count=10,
                offset=0
            )
        self.assertIn("is only supported for unstructured indexes", str(cm.exception))

    def test_unsupported_retrieval_and_ranking_combination_raises_error(self):
        test_cases = [
            (RetrievalMethod.Tensor, RankingMethod.Lexical, "retrievalMethod: tensor, rankingMethod: lexical"),
            (RetrievalMethod.Lexical, RankingMethod.Tensor, "retrievalMethod: lexical, rankingMethod: tensor"),
            (RetrievalMethod.Tensor, RankingMethod.Tensor, "retrievalMethod: tensor, rankingMethod: tensor"),
        ]
        for retrieval_method, ranking_method, description in test_cases:
            with self.subTest(description=description):
                with self.assertRaises(ValidationError) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=self.unstructured_index_with_collapse_field,
                        text="test",
                        search_method=SearchMethod.HYBRID,
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method,
                            rankingMethod=ranking_method,
                            secondPhaseModifier=True,
                            rerankCount=10,
                        ),
                        result_count=10,
                        offset=0
                    )
                self.assertIn(
                    "'secondPhaseModifier' can only be set to True when 'retrievalMethod' is 'disjunction' or both "
                    "'retrievalMethod' and 'rankingMethod' are 'lexical'", str(cm.exception)
                )


class TestRerankDepthLexicalAndWeakAndParameters(MarqoTestCase):
    """
    A test class to verify rerankDepthLexical and weakAndParameters.

    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='random/small')
        )

        cls.create_indexes([
            semi_structured_default_text_index,
        ])

        cls.index_name = semi_structured_default_text_index.name

    def test_invalid_retrieval_method_raises_error(self):
        with self.assertRaises(ValidationError) as cm:
            tensor_search.search(
                config=self.config,
                index_name=self.index_name,
                text="test",
                search_method=SearchMethod.HYBRID,
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Lexical,
                    rerankDepthLexical=10,
                ),
                result_count=10,
                offset=0
            )
        self.assertIn(
            "'rerankDepthLexical' can only be set when 'retrievalMethod' is "
            "'lexical' or 'disjunction'", str(cm.exception)
        )

    def test_weakand_parameters_stopwordlimit(self):
        """
        Test that very common stop words can be excluded in the retrieval using weakAndParameters.stopwordLimit.
        """
        irrelevant_docs = [
            {
                '_id': f'{doc_id}',
                'text': 'stop stop stop irrelevant',
                "score_modifier_value": 100.0
            }
            for doc_id in range(10)
        ]

        relevant_docs = [
            {
                "_id": f'relevant_0',
                "text": "relevant documents",
            }
        ]

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name,
                docs=irrelevant_docs + relevant_docs,
                tensor_fields=['text'],
            )
        )

        results_without_weakand_parameters = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="stop stop stop relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                rerankDepthLexical=100,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                )
            ),
            offset=0,
            result_count=10
        )

        results_without_weakand_parameters_ids = [
            doc['_id'] for doc in results_without_weakand_parameters['hits']
        ]
        self.assertNotIn('relevant_0', results_without_weakand_parameters_ids)


        results_with_stop_word_limit = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="stop stop stop relevant documents",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                rerankDepthLexical=10,
                weakAndParameters=WeakAndParameters(
                    stopwordLimit=0.2,
                    allowDropAll=False,
                    adjustTarget=0.8,
                ),
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="score_modifier_value",
                            weight=1.0
                        )
                    ]
                )
            ),
            offset=0,
            result_count=10
        )

        results_with_stop_word_limit_ids = [
            doc['_id'] for doc in results_with_stop_word_limit['hits']
        ]
        self.assertIn('relevant_0', results_with_stop_word_limit_ids)

    def test_weakand_parameters_drop_all(self):
        """
        Test that when all terms are dropped due to stopwordLimit, no results are returned if allowDropAll is True.
        """
        irrelevant_docs = [
            {
                '_id': f'{doc_id}',
                'text': 'stop stop stop irrelevant',
                "score_modifier_value": 100.0
            }
            for doc_id in range(10)
        ]

        relevant_docs = [
            {
                "_id": f'relevant_0',
                "text": "relevant documents",
            }
        ]

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name,
                docs=irrelevant_docs + relevant_docs,
                tensor_fields=['text'],
            )
        )
        results_with_stop_word_limit_and_drop_all = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="stop stop stop",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                rerankDepthLexical=10,
                weakAndParameters=WeakAndParameters(
                    stopwordLimit=0.2,
                    allowDropAll=True,
                    adjustTarget=0.8,
                ),
            ),
            offset=0,
            result_count=10
        )

        self.assertEqual([], results_with_stop_word_limit_and_drop_all['hits'])


@pytest.mark.skip_for_multinode
class TestSecondPhaseLexicalModifiersAndCollapseField(MarqoTestCase):
    """
    A test class to verify that using second phase lexical modifiers with collapse field raises an error.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            collapse_fields=[CollapseField(name="parent_id")],
        )

        cls.create_indexes([
            semi_structured_default_text_index,
        ])

        cls.index_name = semi_structured_default_text_index.name

        documents =[
            {
                "title": "red speedo goggles",
                "parent_id": "group_1",
                "_id": "doc_1_1"
            },
            {
                "title": "blue speedo goggles",
                "parent_id": "group_1",
                "_id": "doc_1_2"
            },
            {
                "title": "green speedo goggles",
                "parent_id": "group_1",
                "add_to_score": 100.0,
                "_id": "doc_1_3"
            },
            {
                "title": "yellow speedo goggles",
                "parent_id": "group_1",
                "_id": "doc_1_4"
            },
        ]



        _ = cls.add_documents(
            config=cls.config, add_docs_params=AddDocsParams(
                index_name=cls.index_name,
                docs=documents,
                tensor_fields=['title'],
            )
        )

    def setUp(self) -> None:
        # To override the default behavior of cleaning up indexes after each test
        self.assertEqual(4, self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents)

    def test_collapse_field_collapse_to_high_score_modifier_documents(self):
        """
        A test to verify that when using collapse field, the document with highest score modifier is returned
        due to first phase lexical scoring.
        """
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="red speedo goggles",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="add_to_score",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=False,
            ),
            collapse=CollapseModel(name="parent_id"),
            result_count=10,
            offset=0
        )

        ids = [doc['_id'] for doc in res['hits']]
        self.assertEqual('doc_1_3', ids[0])

    def test_collapse_field_collapse_to_high_relevance_documents(self):
        """
        A test to verify that when using collapse field, the document with highest relevance is returned
        due to second phase lexical scoring.
        """
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="red speedo goggles",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                scoreModifiersLexical=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(
                            field_name="add_to_score",
                            weight=1.0
                        )
                    ]
                ),
                secondPhaseModifier=True,
            ),
            collapse=CollapseModel(name="parent_id"),
            result_count=10,
            offset=0
        )

        ids = [doc['_id'] for doc in res['hits']]
        self.assertEqual('doc_1_1', ids[0])
