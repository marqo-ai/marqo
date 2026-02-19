"""Unit tests for semi-structured Vespa schema: custom score rerank query inputs and summary-features."""
import time
import unittest
from typing import List

from marqo.core.models.marqo_index import (
    SemiStructuredMarqoIndex,
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
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema


class TestSemiStructuredSchemaCustomScoreRerank(unittest.TestCase):
    """Semi-structured schema must include custom score rerank query inputs and summary-features (no match-features)."""

    def _create_marqo_index(
        self,
        lexical_field_names: List[str],
        tensor_field_names: List[str],
        marqo_version: str = "2.16.0",
    ) -> SemiStructuredMarqoIndex:
        lexical_fields = [
            Field(
                name=name,
                type=FieldType.Text,
                features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name=f"marqo__lexical_{name}",
                filter_field_name=f"{name}_filter",
            )
            for name in lexical_field_names
        ]
        tensor_fields = [
            TensorField(
                name=name,
                embeddings_field_name=f"marqo__embeddings_{name}",
                chunk_field_name=f"marqo__chunks_{name}",
            )
            for name in tensor_field_names
        ]
        return SemiStructuredMarqoIndex(
            name="test",
            schema_name="test",
            model=Model(name="hf/all-MiniLM-L6-v2"),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type="float",
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version=marqo_version,
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

    def test_schema_has_custom_score_query_inputs(self):
        """Rendered schema must define query inputs for custom score weights (semi-structured 2.16+ only)."""
        marqo_index = self._create_marqo_index(
            lexical_field_names=["title"],
            tensor_field_names=["title"],
            marqo_version="2.16.0",
        )
        schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
        self.assertIn("query(marqo__query_embedding)", schema)
        self.assertIn("query(marqo__custom_score_mult_weights_global)", schema)
        self.assertIn("query(marqo__custom_score_add_weights_global)", schema)

    def test_schema_has_summary_features_for_custom_score(self):
        """Schema must expose ranking_closeness_metric_<field> and bm25(lexical) as summary-features, not match-features."""
        marqo_index = self._create_marqo_index(
            lexical_field_names=["title", "desc"],
            tensor_field_names=["title", "desc"],
            marqo_version="2.16.0",
        )
        schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
        self.assertIn("summary-features", schema)
        self.assertIn("ranking_closeness_metric_title", schema)
        self.assertIn("ranking_closeness_metric_desc", schema)
        self.assertIn("bm25(marqo__lexical_title)", schema)
        self.assertIn("bm25(marqo__lexical_desc)", schema)

    def test_schema_hybrid_custom_searcher_has_custom_score_inputs(self):
        """hybrid_custom_searcher rank profile must include custom score query inputs."""
        marqo_index = self._create_marqo_index(
            lexical_field_names=["title"],
            tensor_field_names=["title"],
            marqo_version="2.16.0",
        )
        schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
        self.assertIn("rank-profile hybrid_custom_searcher", schema)
        self.assertIn("marqo__custom_score_mult_weights_global", schema)
        self.assertIn("marqo__custom_score_add_weights_global", schema)
        self.assertIn("marqo__query_embedding", schema)


if __name__ == "__main__":
    unittest.main()
