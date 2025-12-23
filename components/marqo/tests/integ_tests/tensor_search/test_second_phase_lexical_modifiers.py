from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import *
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierOperator
from tests.integ_tests.marqo_test import MarqoTestCase


class TestSecondPhaseLexicalModifiers(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
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

    def test_second_phase_modifiers_is_working(self):
        """
        This test tests the following things:
        1. The current first phase lexical score modifier implementation works as expected, a relevant document without
              score modifier field is not returned when there are irrelevant documents with high score modifier values.
        2. The second phase lexical score modifier implementation works as expected, a relevant document without
                score modifier field is returned when there are irrelevant documents with high score modifier values if
                it is inside the rerankCount.
        3. However, if the rerankCount is too large, the relevant document can be squeezed out of the results.
        """
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

        res = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name,
                docs=irrelevant_docs + relevant_docs,
                tensor_fields=['text'],
            )
        )

        self.assertEqual(11, self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents)

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
        self.assertIn("relevant_0", second_phase_score_modifier_results_ids)

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