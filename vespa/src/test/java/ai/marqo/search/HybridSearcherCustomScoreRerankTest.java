package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for custom score rerank logic in HybridSearcher (Part C of the feature plan):
 * parsing keys, resolving match feature names, extracting scores, and min-max normalization.
 */
class HybridSearcherCustomScoreRerankTest {

    @Nested
    class ParseCustomScoreKeyTest {

        @Test
        void bm25_field_returns_parsed() {
            HybridSearcher.CustomScoreKeyParsed p =
                    HybridSearcher.parseCustomScoreKey("bm25_field_variantTitle");
            assertThat(p).isNotNull();
            assertThat(p.scoreType).isEqualTo("bm25");
            assertThat(p.fieldName).isEqualTo("variantTitle");
            assertThat(p.aggregateType).isNull();
        }

        @Test
        void bm25_aggregates_return_parsed() {
            assertThat(HybridSearcher.parseCustomScoreKey("bm25_sum"))
                    .satisfies(
                            p -> {
                                assertThat(p.scoreType).isEqualTo("bm25");
                                assertThat(p.fieldName).isNull();
                                assertThat(p.aggregateType).isEqualTo("sum");
                            });
            assertThat(HybridSearcher.parseCustomScoreKey("bm25_max"))
                    .satisfies(
                            p -> {
                                assertThat(p.aggregateType).isEqualTo("max");
                            });
            assertThat(HybridSearcher.parseCustomScoreKey("bm25_avg"))
                    .satisfies(
                            p -> {
                                assertThat(p.aggregateType).isEqualTo("avg");
                            });
        }

        @Test
        void closeness_retrieval_vector_field_returns_parsed() {
            HybridSearcher.CustomScoreKeyParsed p =
                    HybridSearcher.parseCustomScoreKey(
                            "closeness_retrieval_vector_field_variantImage");
            assertThat(p).isNotNull();
            assertThat(p.scoreType).isEqualTo("closeness_retrieval_vector");
            assertThat(p.fieldName).isEqualTo("variantImage");
            assertThat(p.aggregateType).isNull();
        }

        @Test
        void closeness_retrieval_vector_aggregates_return_parsed() {
            assertThat(HybridSearcher.parseCustomScoreKey("closeness_retrieval_vector_sum"))
                    .satisfies(
                            p -> {
                                assertThat(p.scoreType).isEqualTo("closeness_retrieval_vector");
                                assertThat(p.fieldName).isNull();
                                assertThat(p.aggregateType).isEqualTo("sum");
                            });
        }

        @Test
        void unsupported_or_invalid_returns_null() {
            assertThat(HybridSearcher.parseCustomScoreKey("closeness_ranking_vector_sum")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey("unknown_type_field_x")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey("")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey("bm25")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey("bm25_")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey("bm25_field_")).isNull();
            assertThat(HybridSearcher.parseCustomScoreKey(null)).isNull();
        }
    }

    @Nested
    class FindBm25MatchFeatureNameTest {

        @Test
        void finds_marqo_lexical_convention() {
            Set<String> keys = new HashSet<>();
            keys.add("bm25(marqo__lexical_title)");
            keys.add("bm25(marqo__lexical_description)");
            assertThat(HybridSearcher.findBm25MatchFeatureName(keys, "title"))
                    .isEqualTo("bm25(marqo__lexical_title)");
        }

        @Test
        void finds_suffix_lexical_convention() {
            Set<String> keys = new HashSet<>();
            keys.add("bm25(title_lexical)");
            keys.add("bm25(description_lexical)");
            assertThat(HybridSearcher.findBm25MatchFeatureName(keys, "title"))
                    .isEqualTo("bm25(title_lexical)");
        }

        @Test
        void returns_null_when_not_found() {
            Set<String> keys = new HashSet<>();
            keys.add("bm25(marqo__lexical_other)");
            assertThat(HybridSearcher.findBm25MatchFeatureName(keys, "title")).isNull();
        }
    }

    @Nested
    class FindClosenessMatchFeatureNameTest {

        @Test
        void finds_marqo_embeddings_convention() {
            Set<String> keys = new HashSet<>();
            keys.add("closeness(field,marqo__embeddings_title)");
            keys.add("closeness(field,marqo__embeddings_description)");
            assertThat(HybridSearcher.findClosenessMatchFeatureName(keys, "title"))
                    .isEqualTo("closeness(field,marqo__embeddings_title)");
        }

        @Test
        void finds_suffix_embeddings_convention() {
            Set<String> keys = new HashSet<>();
            keys.add("closeness(field,title_embeddings)");
            assertThat(HybridSearcher.findClosenessMatchFeatureName(keys, "title"))
                    .isEqualTo("closeness(field,title_embeddings)");
        }

        @Test
        void returns_null_when_not_found() {
            Set<String> keys = new HashSet<>();
            keys.add("closeness(field,marqo__embeddings_other)");
            assertThat(HybridSearcher.findClosenessMatchFeatureName(keys, "title")).isNull();
        }
    }

    @Nested
    class MinMaxNormalizeTest {

        @Test
        void normalizes_to_zero_one() {
            assertThat(HybridSearcher.minMaxNormalize(0.0, 0.0, 10.0)).isEqualTo(0.0);
            assertThat(HybridSearcher.minMaxNormalize(10.0, 0.0, 10.0)).isEqualTo(1.0);
            assertThat(HybridSearcher.minMaxNormalize(5.0, 0.0, 10.0)).isEqualTo(0.5);
        }

        @Test
        void min_equals_max_returns_half() {
            assertThat(HybridSearcher.minMaxNormalize(3.0, 3.0, 3.0)).isEqualTo(1.0);
        }

        @Test
        void clamps_to_zero_one() {
            assertThat(HybridSearcher.minMaxNormalize(-1.0, 0.0, 10.0)).isEqualTo(0.0);
            assertThat(HybridSearcher.minMaxNormalize(11.0, 0.0, 10.0)).isEqualTo(1.0);
        }
    }

    @Nested
    class ExtractCustomScoreForHitTest {

        /** Custom score reranking uses only summary-features; pass summaryFeatures with bm25(marqo__lexical_<field>). */
        @Test
        void extracts_bm25_single_field() {
            FeatureData summaryFeatures = mock(FeatureData.class);
            when(summaryFeatures.getDouble("bm25(marqo__lexical_title)")).thenReturn(2.5);
            Set<String> keys = Set.of();
            HybridSearcher.CustomScoreKeyParsed parsed =
                    HybridSearcher.parseCustomScoreKey("bm25_field_title");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null, "bm25_field_title", parsed, keys, summaryFeatures))
                    .isEqualTo(2.5);
        }

        /** Custom score uses only summary-features; pass summaryFeatures with ranking_closeness_metric_<field>. */
        @Test
        void extracts_closeness_single_field() {
            FeatureData summaryFeatures = mock(FeatureData.class);
            when(summaryFeatures.getDouble("ranking_closeness_metric_title")).thenReturn(0.9);
            Set<String> keys = Set.of();
            HybridSearcher.CustomScoreKeyParsed parsed =
                    HybridSearcher.parseCustomScoreKey("closeness_retrieval_vector_field_title");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null,
                                    "closeness_retrieval_vector_field_title",
                                    parsed,
                                    keys,
                                    summaryFeatures))
                    .isEqualTo(0.9);
        }

        /** BM25 aggregate: sum over all bm25(marqo__lexical_*) in summary-features. */
        @Test
        void aggregates_bm25_sum() {
            FeatureData summaryFeatures = mock(FeatureData.class);
            when(summaryFeatures.getDouble("bm25(marqo__lexical_a)")).thenReturn(1.0);
            when(summaryFeatures.getDouble("bm25(marqo__lexical_b)")).thenReturn(2.0);
            when(summaryFeatures.featureNames())
                    .thenReturn(Set.of("bm25(marqo__lexical_a)", "bm25(marqo__lexical_b)"));
            Set<String> keys = Set.of();
            HybridSearcher.CustomScoreKeyParsed parsed =
                    HybridSearcher.parseCustomScoreKey("bm25_sum");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null, "bm25_sum", parsed, keys, summaryFeatures))
                    .isEqualTo(3.0);
        }

        @Test
        void returns_null_when_summary_features_null() {
            HybridSearcher.CustomScoreKeyParsed parsed =
                    HybridSearcher.parseCustomScoreKey("bm25_field_title");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null, "bm25_field_title", parsed, Set.of(), null))
                    .isNull();
        }
    }

    /**
     * Closeness (e.g. dot product) is min-max normalized to [0,1] with 1 = closest so that
     * unbounded metrics behave like angular/euclidean. These tests cover the normalization helper
     * and the computation of per-key min/max across hits.
     */
    @Nested
    class ClosenessDotProductNormalizationTest {

        @Test
        void shouldMinMaxNormalizeCloseness_true_only_for_dotproduct() {
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("dotproduct")).isTrue();
        }

        @Test
        void shouldMinMaxNormalizeCloseness_false_for_other_metrics() {
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("angular")).isFalse();
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("euclidean")).isFalse();
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("prenormalized-angular"))
                    .isFalse();
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("hamming")).isFalse();
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness("")).isFalse();
            assertThat(HybridSearcher.shouldMinMaxNormalizeCloseness(null)).isFalse();
        }

        @Test
        void normalizeClosenessScoreForRerank_returns_zero_one_when_map_has_key() {
            String key = "closeness_retrieval_vector_field_title";
            Map<String, double[]> map = new HashMap<>();
            map.put(key, new double[] {10.0, 30.0}); // min=10, max=30 (e.g. raw dot product)
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(10.0, key, map))
                    .isEqualTo(0.0);
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(30.0, key, map))
                    .isEqualTo(1.0);
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(20.0, key, map))
                    .isEqualTo(0.5);
        }

        @Test
        void normalizeClosenessScoreForRerank_returns_score_unchanged_when_map_null() {
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(25.0, "any_key", null))
                    .isEqualTo(25.0);
        }

        @Test
        void normalizeClosenessScoreForRerank_returns_score_unchanged_when_key_missing() {
            Map<String, double[]> map = new HashMap<>();
            map.put("other_key", new double[] {0.0, 1.0});
            assertThat(
                            HybridSearcher.normalizeClosenessScoreForRerank(
                                    25.0, "closeness_retrieval_vector_field_title", map))
                    .isEqualTo(25.0);
        }

        @Test
        void normalizeClosenessScoreForRerank_clamps_to_zero_one() {
            String key = "closeness_retrieval_vector_field_title";
            Map<String, double[]> map = new HashMap<>();
            map.put(key, new double[] {10.0, 30.0});
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(5.0, key, map))
                    .isEqualTo(0.0);
            assertThat(HybridSearcher.normalizeClosenessScoreForRerank(35.0, key, map))
                    .isEqualTo(1.0);
        }

        @Test
        void
                computeClosenessMinMaxPerKey_returns_min_max_across_hits_for_dot_product_like_scores() {
            HitGroup hits = new HitGroup();
            for (double value : new double[] {10.0, 20.0, 30.0}) {
                Hit hit = new Hit("doc_" + value, 1.0);
                FeatureData summaryFeatures = mock(FeatureData.class);
                when(summaryFeatures.getDouble("ranking_closeness_metric_title")).thenReturn(value);
                when(summaryFeatures.featureNames())
                        .thenReturn(Set.of("ranking_closeness_metric_title"));
                hit.setField("summaryfeatures", summaryFeatures);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(
                                    TensorAddress.ofLabels(
                                            "closeness_retrieval_vector_field_title"),
                                    1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, double[]> result =
                    searcher.computeClosenessMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_field_title");
            double[] minMax = result.get("closeness_retrieval_vector_field_title");
            assertThat(minMax).hasSize(2);
            assertThat(minMax[0]).isEqualTo(10.0);
            assertThat(minMax[1]).isEqualTo(30.0);
        }

        @Test
        void computeClosenessMinMaxPerKey_ignores_bm25_keys() {
            HitGroup hits = new HitGroup();
            Hit hit = new Hit("doc1", 1.0);
            FeatureData summaryFeatures = mock(FeatureData.class);
            when(summaryFeatures.getDouble("ranking_closeness_metric_title")).thenReturn(0.9);
            when(summaryFeatures.featureNames())
                    .thenReturn(Set.of("ranking_closeness_metric_title"));
            hit.setField("summaryfeatures", summaryFeatures);
            hits.add(hit);
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_field_title"), 1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, double[]> result =
                    searcher.computeClosenessMinMaxPerKey(hits, addWeights, null);
            assertThat(result).isEmpty();
        }
    }
}
