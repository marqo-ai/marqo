"""Unit tests for custom score rerank logic in vespa_index (LLD A)."""
import unittest
from typing import List

from marqo.core.constants import MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.vespa_index.vespa_index import VespaIndex
from marqo.exceptions import InvalidArgumentError


class TestParseCustomScoreKey(unittest.TestCase):
    """Tests for VespaIndex.parse_custom_score_key."""

    def test_bm25_field(self):
        self.assertEqual(
            VespaIndex.parse_custom_score_key("bm25_field_variantTitle"),
            ("bm25", "variantTitle", None),
        )
        self.assertEqual(
            VespaIndex.parse_custom_score_key("bm25_field_title"),
            ("bm25", "title", None),
        )

    def test_bm25_aggregates(self):
        for agg in ("sum", "max", "avg"):
            self.assertEqual(
                VespaIndex.parse_custom_score_key(f"bm25_{agg}"),
                ("bm25", None, agg),
            )

    def test_closeness_retrieval_vector_field(self):
        self.assertEqual(
            VespaIndex.parse_custom_score_key("closeness_retrieval_vector_field_variantImage"),
            ("closeness_retrieval_vector", "variantImage", None),
        )

    def test_closeness_retrieval_vector_aggregates(self):
        for agg in ("sum", "max", "avg"):
            self.assertEqual(
                VespaIndex.parse_custom_score_key(f"closeness_retrieval_vector_{agg}"),
                ("closeness_retrieval_vector", None, agg),
            )

    def test_unsupported_or_invalid_returns_none(self):
        self.assertIsNone(VespaIndex.parse_custom_score_key("closeness_ranking_vector_sum"))
        self.assertIsNone(VespaIndex.parse_custom_score_key("unknown_type_field_x"))
        self.assertIsNone(VespaIndex.parse_custom_score_key(""))
        self.assertIsNone(VespaIndex.parse_custom_score_key("bm25"))
        self.assertIsNone(VespaIndex.parse_custom_score_key("bm25_"))
        self.assertIsNone(VespaIndex.parse_custom_score_key("bm25_field_"))


class TestConvertHybridGlobalScoreModifiersToTensors(unittest.TestCase):
    """Tests for _convert_hybrid_global_score_modifiers_to_tensors (custom vs global split and keys)."""

    def _create_index_with_hybrid(self):
        import time
        from marqo.core.models.marqo_index import (
            StructuredMarqoIndex,
            Model,
            TextPreProcessing,
            TextSplitMethod,
            ImagePreProcessing,
            HnswConfig,
            DistanceMetric,
            Field,
            FieldType,
            FieldFeature,
            TensorField,
        )
        fields = [
            Field(
                name="title",
                type=FieldType.Text,
                features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="title_lexical",
                filter_field_name="title_filter",
            ),
        ]
        tensor_fields = [
            TensorField(
                name="title",
                embeddings_field_name="title_embeddings",
                chunk_field_name="title_chunks",
            ),
        ]
        idx = StructuredMarqoIndex(
            name="test",
            schema_name="test",
            model=Model(name="test"),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type="float",
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version="2.15.0",
            created_at=time.time(),
            updated_at=time.time(),
            fields=fields,
            tensor_fields=tensor_fields,
            text_preprocessing=TextPreProcessing(
                split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
        )
        from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
        return StructuredVespaIndex(idx)

    def test_custom_score_uses_custom_query_input_keys(self):
        """Custom score modifiers (marqo__score_ prefix) go to custom score rerank tensors."""
        vespa_index = self._create_index_with_hybrid()
        modifiers = [
            ScoreModifier(
                field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                weight=1.0,
                type=ScoreModifierType.Add,
            ),
            ScoreModifier(
                field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_max",
                weight=2.0,
                type=ScoreModifierType.Add,
            ),
        ]
        g_mult, g_add, c_mult, c_add = vespa_index._convert_hybrid_global_score_modifiers_to_tensors(
            modifiers
        )
        self.assertEqual(g_mult, {})
        self.assertEqual(g_add, {})
        self.assertEqual(c_mult, {})
        self.assertEqual(c_add, {"bm25_field_title": 1.0, "bm25_max": 2.0})

    def test_custom_score_invalid_format_raises_400(self):
        """Field starting with marqo__score_ but invalid format raises InvalidArgumentError (400)."""
        vespa_index = self._create_index_with_hybrid()
        modifiers = [
            ScoreModifier(
                field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}invalid_format",
                weight=1.0,
                type=ScoreModifierType.Add,
            ),
        ]
        with self.assertRaises(InvalidArgumentError) as ctx:
            vespa_index._convert_hybrid_global_score_modifiers_to_tensors(modifiers)
        self.assertIn("invalid format", str(ctx.exception))
        self.assertIn("invalid_format", str(ctx.exception))

    def test_global_score_modifiers_unchanged(self):
        """Non-custom score modifiers go to global mult/add only."""
        vespa_index = self._create_index_with_hybrid()
        modifiers = [
            ScoreModifier(field="some_doc_field", weight=0.5, type=ScoreModifierType.Multiply),
            ScoreModifier(field="other_field", weight=10.0, type=ScoreModifierType.Add),
        ]
        g_mult, g_add, c_mult, c_add = vespa_index._convert_hybrid_global_score_modifiers_to_tensors(
            modifiers
        )
        self.assertEqual(g_mult, {"some_doc_field": 0.5})
        self.assertEqual(g_add, {"other_field": 10.0})
        self.assertEqual(c_mult, {})
        self.assertEqual(c_add, {})

    def test_mixed_custom_and_global(self):
        vespa_index = self._create_index_with_hybrid()
        modifiers = [
            ScoreModifier(field="doc_field", weight=1.0, type=ScoreModifierType.Add),
            ScoreModifier(
                field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_sum",
                weight=0.5,
                type=ScoreModifierType.Multiply,
            ),
        ]
        g_mult, g_add, c_mult, c_add = vespa_index._convert_hybrid_global_score_modifiers_to_tensors(
            modifiers
        )
        self.assertEqual(g_mult, {})
        self.assertEqual(g_add, {"doc_field": 1.0})
        self.assertEqual(c_mult, {"bm25_sum": 0.5})
        self.assertEqual(c_add, {})


if __name__ == "__main__":
    unittest.main()
