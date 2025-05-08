import pytest
from integ_tests.marqo_test import MarqoTestCase

from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType


@pytest.mark.unittest
class TestMarqoHybridQuery(MarqoTestCase):
    def test_hybrid_query_with_score_modifiers_with_wrong_ranking_method_fails(self):
        """
        Test that score modifiers can only be defined in a hybrid query if rankingMethod is `RRF`
        """
        for retrieval_method, ranking_method in [
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Lexical)
        ]:
            with self.assertRaises(ValueError) as e:
                my_query = MarqoHybridQuery(
                    index_name="my_index",
                    limit=10,
                    score_modifiers=[
                        ScoreModifier(
                            field="my_field_1",
                            weight=1.0,
                            type=ScoreModifierType.Multiply
                        ),
                        ScoreModifier(
                            field="my_field_2",
                            weight=1.0,
                            type=ScoreModifierType.Add
                        )
                    ],
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method
                    )
                )

            self.assertIn("only supported for hybrid search if 'rankingMethod' is 'RRF'", str(e.exception))

    def test_hybrid_query_with_searchable_attributes_fails(self):
        """
        Test that searchable attributes cannot be defined in a hybrid query
        """
        with self.assertRaises(ValueError) as e:
            my_query = MarqoHybridQuery(
                index_name="my_index",
                limit=10,
                searchable_attributes=["my_field_1", "my_field_2"],
            )

        self.assertIn("'searchableAttributes' cannot be used for hybrid search", str(e.exception))
