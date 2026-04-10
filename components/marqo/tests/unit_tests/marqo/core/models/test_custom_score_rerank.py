"""Unit tests for ParsedCustomScoreKey."""
import unittest

from marqo.core.models.custom_score_rerank import ParsedCustomScoreKey


class TestParsedCustomScoreKeyParse(unittest.TestCase):
    """Tests for ParsedCustomScoreKey.parse."""

    def test_bm25_per_field(self):
        for key, expected_field in [
            ("bm25_field_variantTitle", "variantTitle"),
            ("bm25_field_title", "title"),
            ("bm25_field_a_b_c", "a_b_c"),
        ]:
            with self.subTest(key=key):
                p = ParsedCustomScoreKey.parse(key)
                self.assertIsNotNone(p)
                assert p is not None
                self.assertEqual(p.score_type, "bm25")
                self.assertEqual(p.field_name, expected_field)
                self.assertIsNone(p.aggregate_type)

    def test_bm25_aggregates(self):
        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                p = ParsedCustomScoreKey.parse(f"bm25_{agg}")
                self.assertIsNotNone(p)
                assert p is not None
                self.assertEqual(p.score_type, "bm25")
                self.assertIsNone(p.field_name)
                self.assertEqual(p.aggregate_type, agg)

    def test_closeness_per_field(self):
        p = ParsedCustomScoreKey.parse("closeness_retrieval_vector_field_variantImage")
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.score_type, "closeness_retrieval_vector")
        self.assertEqual(p.field_name, "variantImage")
        self.assertIsNone(p.aggregate_type)

        p2 = ParsedCustomScoreKey.parse("closeness_retrieval_vector_field_my_embedding")
        self.assertIsNotNone(p2)
        assert p2 is not None
        self.assertEqual(p2.field_name, "my_embedding")

    def test_closeness_aggregates(self):
        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                p = ParsedCustomScoreKey.parse(f"closeness_retrieval_vector_{agg}")
                self.assertIsNotNone(p)
                assert p is not None
                self.assertEqual(p.score_type, "closeness_retrieval_vector")
                self.assertIsNone(p.field_name)
                self.assertEqual(p.aggregate_type, agg)

    def test_returns_none_invalid(self):
        for key in [
            "",
            "bm25",
            "bm25_",
            "bm25_field_",
            "closeness_ranking_vector_sum",
            "unknown_type_field_x",
            "closeness_retrieval_vector",
            "closeness_retrieval_vector_field_",
            "bm25_field",
            "bm25_sun",
        ]:
            with self.subTest(key=key):
                self.assertIsNone(
                    ParsedCustomScoreKey.parse(key),
                    msg=f"expected None for {key!r}",
                )

    def test_bm25_prefix_not_substring_of_closeness(self):
        """bm25_* must not match inside closeness_retrieval_vector_* incorrectly."""
        self.assertIsNone(ParsedCustomScoreKey.parse("x_bm25_field_title"))

    def test_model_equality_and_dict(self):
        a = ParsedCustomScoreKey.parse("bm25_sum")
        b = ParsedCustomScoreKey.parse("bm25_sum")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        assert a is not None and b is not None
        self.assertEqual(a, b)
        self.assertEqual(
            a.dict(),
            {"score_type": "bm25", "field_name": None, "aggregate_type": "sum"},
        )


if __name__ == "__main__":
    unittest.main()
