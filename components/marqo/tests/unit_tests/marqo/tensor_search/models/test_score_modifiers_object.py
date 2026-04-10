import unittest

from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierOperator


class TestScoreModifierListsUsesCustomScoreRerank(unittest.TestCase):
    """ScoreModifierLists.uses_custom_score_rerank is True only when any modifier field starts with marqo__score_."""

    def test_uses_custom_score_rerank_true_add_to_score(self):
        mods = ScoreModifierLists(
            add_to_score=[ScoreModifierOperator(field_name="marqo__score_bm25_field_title", weight=1.0)]
        )
        self.assertTrue(mods.uses_custom_score_rerank)

    def test_uses_custom_score_rerank_true_multiply_score_by(self):
        mods = ScoreModifierLists(
            multiply_score_by=[ScoreModifierOperator(field_name="marqo__score_closeness_retrieval_vector_sum", weight=0.5)]
        )
        self.assertTrue(mods.uses_custom_score_rerank)

    def test_uses_custom_score_rerank_true_mixed(self):
        mods = ScoreModifierLists(
            add_to_score=[ScoreModifierOperator(field_name="popularity", weight=1.0)],
            multiply_score_by=[ScoreModifierOperator(field_name="marqo__score_bm25_sum", weight=1.0)],
        )
        self.assertTrue(mods.uses_custom_score_rerank)

    def test_uses_custom_score_rerank_false_ordinary_fields(self):
        mods = ScoreModifierLists(
            add_to_score=[ScoreModifierOperator(field_name="popularity", weight=1.0)],
            multiply_score_by=[ScoreModifierOperator(field_name="rating", weight=0.5)],
        )
        self.assertFalse(mods.uses_custom_score_rerank)

    def test_uses_custom_score_rerank_false_add_only_ordinary(self):
        mods = ScoreModifierLists(add_to_score=[ScoreModifierOperator(field_name="popularity", weight=1.0)])
        self.assertFalse(mods.uses_custom_score_rerank)

    def test_uses_custom_score_rerank_false_multiply_only_ordinary(self):
        mods = ScoreModifierLists(multiply_score_by=[ScoreModifierOperator(field_name="rating", weight=1.0)])
        self.assertFalse(mods.uses_custom_score_rerank)
