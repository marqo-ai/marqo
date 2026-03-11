"""Unit tests for custom score rerank logic in vespa_index (LLD A)."""
import unittest
from typing import List
from unittest.mock import Mock

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


class TestValidateCustomScoreModifierFieldsAggregates(unittest.TestCase):
    """
    _validate_custom_score_modifier_fields must raise InvalidArgumentError (400) when requesting
    a BM25 aggregate (sum/max/avg) but the index has no lexically searchable fields, or when
    requesting a closeness aggregate but the index has no tensor fields.
    """

    def _create_semi_structured_vespa_index_with_empty_lexical_and_tensor(self):
        """SemiStructuredVespaIndex whose marqo_index has no lexical and no tensor fields."""
        from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
        mock_index = Mock()
        mock_index.lexically_searchable_fields_names = set()
        mock_index.tensor_field_map = {}
        mock_index.field_map = {}
        mock_index.index_supports_partial_updates = False
        return SemiStructuredVespaIndex(mock_index)

    def test_bm25_aggregate_with_no_lexically_searchable_fields_raises(self):
        """Requesting marqo__score_bm25_sum (or max/avg) with no lexical fields in index must raise 400."""
        vespa_index = self._create_semi_structured_vespa_index_with_empty_lexical_and_tensor()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                custom_score_keys = {f"{prefix}bm25_{agg}"}
                with self.assertRaises(InvalidArgumentError) as ctx:
                    vespa_index._validate_custom_score_modifier_fields(custom_score_keys)
                self.assertIn("BM25 aggregate", str(ctx.exception))
                self.assertIn("no lexically searchable fields", str(ctx.exception))

    def test_closeness_aggregate_with_no_tensor_fields_raises(self):
        """Requesting closeness_retrieval_vector sum/max/avg with no tensor fields in index must raise 400."""
        vespa_index = self._create_semi_structured_vespa_index_with_empty_lexical_and_tensor()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                custom_score_keys = {f"{prefix}closeness_retrieval_vector_{agg}"}
                with self.assertRaises(InvalidArgumentError) as ctx:
                    vespa_index._validate_custom_score_modifier_fields(custom_score_keys)
                self.assertIn("closeness aggregate", str(ctx.exception))
                self.assertIn("no tensor fields", str(ctx.exception))


class TestValidateCustomScoreModifierFieldsGeodegrees(unittest.TestCase):
    """_validate_custom_score_modifier_fields must raise 400 when using closeness_retrieval_vector with geodegrees."""

    def _create_semi_structured_vespa_index_geodegrees(self):
        """SemiStructuredVespaIndex with distance_metric=Geodegrees and one tensor field."""
        import time
        from marqo.core.models.marqo_index import (
            SemiStructuredMarqoIndex,
            Model,
            Field,
            FieldType,
            FieldFeature,
            TensorField,
            HnswConfig,
            DistanceMetric,
            TextPreProcessing,
            TextSplitMethod,
            ImagePreProcessing,
        )
        from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
        lexical_fields = [
            Field(
                name="title",
                type=FieldType.Text,
                features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="marqo__lexical_title",
                filter_field_name="title_filter",
            ),
        ]
        tensor_fields = [
            TensorField(
                name="title",
                embeddings_field_name="marqo__embeddings_title",
                chunk_field_name="marqo__chunks_title",
            ),
        ]
        marqo_index = SemiStructuredMarqoIndex(
            name="test",
            schema_name="test",
            model=Model(name="test"),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Geodegrees,
            vector_numeric_type="float",
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version="2.16.0",
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=[],
        )
        return SemiStructuredVespaIndex(marqo_index)

    def test_closeness_retrieval_vector_with_geodegrees_raises_400(self):
        """Using closeness_retrieval_vector (field or aggregate) with index distance_metric=geodegrees must raise InvalidArgumentError (400)."""
        vespa_index = self._create_semi_structured_vespa_index_geodegrees()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        for custom_score_keys in [
            {f"{prefix}closeness_retrieval_vector_field_title"},
            {f"{prefix}closeness_retrieval_vector_sum"},
        ]:
            with self.subTest(keys=custom_score_keys):
                with self.assertRaises(InvalidArgumentError) as ctx:
                    vespa_index._validate_custom_score_modifier_fields(custom_score_keys)
                self.assertIn("geodegrees", str(ctx.exception).lower())
                self.assertIn("closeness_retrieval_vector", str(ctx.exception).lower())
                self.assertIn("not supported", str(ctx.exception).lower())

    def test_bm25_with_geodegrees_index_succeeds(self):
        """BM25 custom score keys are allowed when index uses geodegrees (only closeness is rejected)."""
        vespa_index = self._create_semi_structured_vespa_index_geodegrees()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        vespa_index._validate_custom_score_modifier_fields({f"{prefix}bm25_field_title"})


class TestValidateCustomScoreModifierFieldsSingleField(unittest.TestCase):
    """_validate_custom_score_modifier_fields must raise for nonexistent or invalid single-field keys."""

    def _create_semi_structured_vespa_index_with_lexical_and_tensor(self):
        """SemiStructuredVespaIndex with title (lexical+tensor) and description (lexical only, no BM25)."""
        from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
        from marqo.core.models.marqo_index import (
            SemiStructuredMarqoIndex, Model, Field, FieldType, FieldFeature,
            TensorField, HnswConfig, DistanceMetric, TextPreProcessing,
            TextSplitMethod, ImagePreProcessing,
        )
        import time
        lexical_fields = [
            Field(
                name="title",
                type=FieldType.Text,
                features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="marqo__lexical_title",
                filter_field_name="title_filter",
            ),
            Field(
                name="description",
                type=FieldType.Text,
                features=[FieldFeature.Filter],
                lexical_field_name=None,
                filter_field_name="description_filter",
            ),
        ]
        tensor_fields = [
            TensorField(
                name="title",
                embeddings_field_name="marqo__embeddings_title",
                chunk_field_name="marqo__chunks_title",
            ),
        ]
        marqo_index = SemiStructuredMarqoIndex(
            name="test",
            schema_name="test",
            model=Model(name="test"),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type="float",
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version="2.16.0",
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=[],
        )
        return SemiStructuredVespaIndex(marqo_index)

    def test_bm25_field_nonexistent_raises(self):
        """bm25_field_<name> for field not in index must raise InvalidArgumentError."""
        vespa_index = self._create_semi_structured_vespa_index_with_lexical_and_tensor()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        with self.assertRaises(InvalidArgumentError) as ctx:
            vespa_index._validate_custom_score_modifier_fields({f"{prefix}bm25_field_nonexistent"})
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("not in the index", str(ctx.exception))

    def test_bm25_field_not_lexically_searchable_raises(self):
        """bm25_field_<name> for field without lexical search (e.g. description here) must raise."""
        vespa_index = self._create_semi_structured_vespa_index_with_lexical_and_tensor()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        with self.assertRaises(InvalidArgumentError) as ctx:
            vespa_index._validate_custom_score_modifier_fields({f"{prefix}bm25_field_description"})
        self.assertIn("not a lexically searchable field", str(ctx.exception))
        self.assertIn("description", str(ctx.exception))

    def test_closeness_field_nonexistent_raises(self):
        """closeness_retrieval_vector_field_<name> for non-tensor field must raise InvalidArgumentError."""
        vespa_index = self._create_semi_structured_vespa_index_with_lexical_and_tensor()
        prefix = MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
        with self.assertRaises(InvalidArgumentError) as ctx:
            vespa_index._validate_custom_score_modifier_fields(
                {f"{prefix}closeness_retrieval_vector_field_nonexistent"}
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("not a tensor field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
