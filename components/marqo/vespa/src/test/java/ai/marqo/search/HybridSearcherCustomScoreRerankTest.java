package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for custom score rerank logic in HybridSearcher (Part C of the feature plan):
 * parsing keys, resolving match feature names, extracting scores, and divide-by-max normalization.
 */
class HybridSearcherCustomScoreRerankTest {

    @Nested
    class ParseCustomScoreKeyTest {

        @Test
        void bm25_field_returns_parsed() {
            HybridSearcher.CustomScoreKey p =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_field_variantTitle");
            assertThat(p).isNotNull();
            assertThat(p.scoreType).isEqualTo("bm25");
            assertThat(p.fieldName).isEqualTo("variantTitle");
            assertThat(p.aggregateType).isNull();
        }

        @Test
        void bm25_aggregates_return_parsed() {
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_sum"))
                    .satisfies(
                            p -> {
                                assertThat(p.scoreType).isEqualTo("bm25");
                                assertThat(p.fieldName).isNull();
                                assertThat(p.aggregateType).isEqualTo("sum");
                            });
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_max"))
                    .satisfies(
                            p -> {
                                assertThat(p.aggregateType).isEqualTo("max");
                            });
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_avg"))
                    .satisfies(
                            p -> {
                                assertThat(p.aggregateType).isEqualTo("avg");
                            });
        }

        @Test
        void closeness_retrieval_vector_field_returns_parsed() {
            HybridSearcher.CustomScoreKey p =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey(
                            "closeness_retrieval_vector_field_variantImage");
            assertThat(p).isNotNull();
            assertThat(p.scoreType).isEqualTo("closeness_retrieval_vector");
            assertThat(p.fieldName).isEqualTo("variantImage");
            assertThat(p.aggregateType).isNull();
        }

        @Test
        void closeness_retrieval_vector_aggregates_return_parsed() {
            assertThat(
                            HybridSearcher.CustomScoreKey.parseCustomScoreKey(
                                    "closeness_retrieval_vector_sum"))
                    .satisfies(
                            p -> {
                                assertThat(p.scoreType).isEqualTo("closeness_retrieval_vector");
                                assertThat(p.fieldName).isNull();
                                assertThat(p.aggregateType).isEqualTo("sum");
                            });
        }

        @Test
        void unsupported_or_invalid_returns_null() {
            assertThat(
                            HybridSearcher.CustomScoreKey.parseCustomScoreKey(
                                    "closeness_ranking_vector_sum"))
                    .isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("unknown_type_field_x"))
                    .isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("")).isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25")).isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_")).isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_field_")).isNull();
            assertThat(HybridSearcher.CustomScoreKey.parseCustomScoreKey(null)).isNull();
        }
    }

    @Nested
    class NormalizeByMaxTest {

        @Test
        void normalizes_by_dividing_by_max() {
            assertThat(HybridSearcher.normalizeByMax(10.0, 10.0)).isEqualTo(1.0);
            assertThat(HybridSearcher.normalizeByMax(5.0, 10.0)).isEqualTo(0.5);
            assertThat(HybridSearcher.normalizeByMax(2.0, 10.0)).isEqualTo(0.2);
        }

        @Test
        void max_zero_returns_one() {
            assertThat(HybridSearcher.normalizeByMax(3.0, 0.0)).isEqualTo(1.0);
        }

        @Test
        void max_negative_returns_one() {
            assertThat(HybridSearcher.normalizeByMax(3.0, -1.0)).isEqualTo(1.0);
        }

        @Test
        void max_nan_returns_one() {
            assertThat(HybridSearcher.normalizeByMax(3.0, Double.NaN)).isEqualTo(1.0);
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
            HybridSearcher.CustomScoreKey parsed =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_field_title");
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
            HybridSearcher.CustomScoreKey parsed =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey(
                            "closeness_retrieval_vector_field_title");
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
            HybridSearcher.CustomScoreKey parsed =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_sum");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null, "bm25_sum", parsed, keys, summaryFeatures))
                    .isEqualTo(3.0);
        }

        @Test
        void returns_null_when_summary_features_null() {
            HybridSearcher.CustomScoreKey parsed =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey("bm25_field_title");
            assertThat(
                            HybridSearcher.extractCustomScoreForHit(
                                    null, "bm25_field_title", parsed, Set.of(), null))
                    .isNull();
        }
    }

    /** Divide-by-max normalization and key stripping: used for both BM25 and closeness. */
    @Nested
    class DivideByMaxNormalizationTest {

        @Test
        void normalizeByMax_returns_proportion_of_max() {
            assertThat(HybridSearcher.normalizeByMax(10.0, 30.0)).isCloseTo(0.333, within(0.001));
            assertThat(HybridSearcher.normalizeByMax(30.0, 30.0)).isEqualTo(1.0);
            assertThat(HybridSearcher.normalizeByMax(20.0, 30.0)).isCloseTo(0.667, within(0.001));
        }

        @Test
        void normalizeByMax_returns_one_when_max_zero() {
            assertThat(HybridSearcher.normalizeByMax(5.0, 0.0)).isEqualTo(1.0);
        }

        @Test
        void computeMaxPerKey_returns_max_for_closeness_single_field() {
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
            Map<String, Double> result = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_field_title");
            assertThat(result.get("closeness_retrieval_vector_field_title")).isEqualTo(30.0);
        }

        @Test
        void computeMaxPerKey_aggregate_closeness_sum_two_fields() {
            HitGroup hits = new HitGroup();
            Hit hit1 = new Hit("doc1", 1.0);
            FeatureData sf1 = mock(FeatureData.class);
            when(sf1.getDouble("ranking_closeness_metric_f1")).thenReturn(0.2);
            when(sf1.getDouble("ranking_closeness_metric_f2")).thenReturn(0.4);
            when(sf1.featureNames())
                    .thenReturn(
                            Set.of("ranking_closeness_metric_f1", "ranking_closeness_metric_f2"));
            hit1.setField("summaryfeatures", sf1);
            hits.add(hit1);
            Hit hit2 = new Hit("doc2", 1.0);
            FeatureData sf2 = mock(FeatureData.class);
            when(sf2.getDouble("ranking_closeness_metric_f1")).thenReturn(0.5);
            when(sf2.getDouble("ranking_closeness_metric_f2")).thenReturn(0.5);
            when(sf2.featureNames())
                    .thenReturn(
                            Set.of("ranking_closeness_metric_f1", "ranking_closeness_metric_f2"));
            hit2.setField("summaryfeatures", sf2);
            hits.add(hit2);
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("closeness_retrieval_vector_sum"), 1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, Double> result = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_sum");
            assertThat(result.get("closeness_retrieval_vector_sum")).isEqualTo(1.0);
        }

        @Test
        void computeMaxPerKey_includes_both_bm25_and_closeness_keys() {
            HitGroup hits = new HitGroup();
            Hit hit = new Hit("doc1", 1.0);
            FeatureData sf = mock(FeatureData.class);
            when(sf.getDouble("bm25(marqo__lexical_title)")).thenReturn(1.0);
            when(sf.getDouble("ranking_closeness_metric_title")).thenReturn(0.9);
            when(sf.featureNames())
                    .thenReturn(
                            Set.of("bm25(marqo__lexical_title)", "ranking_closeness_metric_title"));
            hit.setField("summaryfeatures", sf);
            hits.add(hit);
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_field_title"), 1.0)
                            .cell(
                                    TensorAddress.ofLabels(
                                            "closeness_retrieval_vector_field_title"),
                                    1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, Double> result = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("bm25_field_title");
            assertThat(result).containsKey("closeness_retrieval_vector_field_title");
        }
    }

    /**
     * Divide-by-max normalization is applied after aggregation: for aggregate keys (bm25_sum,
     * closeness_retrieval_vector_sum, etc.) we first compute the aggregate per hit, then compute
     * min/max of that aggregated value across hits, then normalize. So the normalized score is
     * based on the aggregate, not on individual field values.
     */
    @Nested
    class NormalizationAfterAggregationTest {

        @Test
        void computeMaxPerKey_for_bm25_sum_uses_aggregated_value_per_hit() {
            HitGroup hits = new HitGroup();
            // Hit1: bm25_a=1, bm25_b=2 -> sum=3
            Hit hit1 = new Hit("doc1", 1.0);
            FeatureData sf1 = mock(FeatureData.class);
            when(sf1.getDouble("bm25(marqo__lexical_a)")).thenReturn(1.0);
            when(sf1.getDouble("bm25(marqo__lexical_b)")).thenReturn(2.0);
            when(sf1.featureNames())
                    .thenReturn(Set.of("bm25(marqo__lexical_a)", "bm25(marqo__lexical_b)"));
            hit1.setField("summaryfeatures", sf1);
            hits.add(hit1);
            // Hit2: bm25_a=2, bm25_b=4 -> sum=6
            Hit hit2 = new Hit("doc2", 1.0);
            FeatureData sf2 = mock(FeatureData.class);
            when(sf2.getDouble("bm25(marqo__lexical_a)")).thenReturn(2.0);
            when(sf2.getDouble("bm25(marqo__lexical_b)")).thenReturn(4.0);
            when(sf2.featureNames())
                    .thenReturn(Set.of("bm25(marqo__lexical_a)", "bm25(marqo__lexical_b)"));
            hit2.setField("summaryfeatures", sf2);
            hits.add(hit2);
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_sum"), 1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, Double> result = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("bm25_sum");
            assertThat(result.get("bm25_sum")).isEqualTo(6.0);
        }

        @Test
        void computeMaxPerKey_for_closeness_retrieval_vector_sum_uses_aggregated_value_per_hit() {
            HitGroup hits = new HitGroup();
            // Hit1: f1=0.2, f2=0.4 -> sum=0.6
            Hit hit1 = new Hit("doc1", 1.0);
            FeatureData sf1 = mock(FeatureData.class);
            when(sf1.getDouble("ranking_closeness_metric_f1")).thenReturn(0.2);
            when(sf1.getDouble("ranking_closeness_metric_f2")).thenReturn(0.4);
            when(sf1.featureNames())
                    .thenReturn(
                            Set.of("ranking_closeness_metric_f1", "ranking_closeness_metric_f2"));
            hit1.setField("summaryfeatures", sf1);
            hits.add(hit1);
            // Hit2: f1=0.5, f2=0.5 -> sum=1.0
            Hit hit2 = new Hit("doc2", 1.0);
            FeatureData sf2 = mock(FeatureData.class);
            when(sf2.getDouble("ranking_closeness_metric_f1")).thenReturn(0.5);
            when(sf2.getDouble("ranking_closeness_metric_f2")).thenReturn(0.5);
            when(sf2.featureNames())
                    .thenReturn(
                            Set.of("ranking_closeness_metric_f1", "ranking_closeness_metric_f2"));
            hit2.setField("summaryfeatures", sf2);
            hits.add(hit2);
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("closeness_retrieval_vector_sum"), 1.0)
                            .build();
            HybridSearcher searcher = new HybridSearcher();
            Map<String, Double> result = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_sum");
            assertThat(result.get("closeness_retrieval_vector_sum")).isEqualTo(1.0);
        }
    }

    /**
     * Divide-by-max normalization maps raw custom scores to [value/max], with the maximum score
     * across hits mapping to 1. Three hits with positive raw scores 2, 4, 8 are used; we assert
     * max maps to 1.0 and others are proportional.
     */
    @Nested
    class NormalizationOutputTest {

        private static final double[] RAW_SCORES = {2.0, 4.0, 8.0};
        private static final double MAX_RAW = 8.0;

        /** Build hits, compute maxPerKey for the given key, return normalized scores in hit order. */
        private List<Double> computeNormalizedScoresForKey(
                HitGroup hits, String key, Tensor addWeights) {
            HybridSearcher searcher = new HybridSearcher();
            Map<String, Double> maxPerKey = searcher.computeMaxPerKey(hits, addWeights, null);
            assertThat(maxPerKey).containsKey(key);
            assertThat(maxPerKey.get(key)).isEqualTo(MAX_RAW);

            HybridSearcher.CustomScoreKey parsed =
                    HybridSearcher.CustomScoreKey.parseCustomScoreKey(key);
            assertThat(parsed).isNotNull();
            List<Double> normalized = new ArrayList<>();
            for (Hit hit : hits) {
                FeatureData summaryFeatures = (FeatureData) hit.getField("summaryfeatures");
                Double raw =
                        HybridSearcher.extractCustomScoreForHit(
                                null, key, parsed, Set.of(), summaryFeatures);
                assertThat(raw).isNotNull();
                double norm = HybridSearcher.normalizeByMax(raw, maxPerKey.get(key));
                normalized.add(norm);
            }
            return normalized;
        }

        private void assertNormalizationMaxToOneOthersProportional(List<Double> normalized) {
            assertThat(normalized).hasSize(3);
            // Max (8.0) maps to 1.0
            assertThat(normalized.get(2)).isEqualTo(1.0);
            // Others are proportional: 2/8=0.25, 4/8=0.5
            assertThat(normalized.get(0)).isCloseTo(2.0 / MAX_RAW, within(1e-9));
            assertThat(normalized.get(1)).isCloseTo(4.0 / MAX_RAW, within(1e-9));
            // No score is 0 (all raw scores are positive)
            for (Double n : normalized) {
                assertThat(n).isGreaterThan(0.0);
                assertThat(n).isLessThanOrEqualTo(1.0);
            }
        }

        @Test
        void bm25_single_field_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("bm25(marqo__lexical_title)")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("bm25(marqo__lexical_title)"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_field_title"), 1.0)
                            .build();
            List<Double> normalized =
                    computeNormalizedScoresForKey(hits, "bm25_field_title", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void closeness_retrieval_vector_single_field_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("ranking_closeness_metric_title")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("ranking_closeness_metric_title"));
                hit.setField("summaryfeatures", sf);
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
            List<Double> normalized =
                    computeNormalizedScoresForKey(
                            hits, "closeness_retrieval_vector_field_title", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void bm25_sum_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("bm25(marqo__lexical_a)")).thenReturn(raw);
                when(sf.getDouble("bm25(marqo__lexical_b)")).thenReturn(0.0);
                when(sf.featureNames())
                        .thenReturn(Set.of("bm25(marqo__lexical_a)", "bm25(marqo__lexical_b)"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_sum"), 1.0)
                            .build();
            List<Double> normalized = computeNormalizedScoresForKey(hits, "bm25_sum", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void bm25_max_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("bm25(marqo__lexical_title)")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("bm25(marqo__lexical_title)"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_max"), 1.0)
                            .build();
            List<Double> normalized = computeNormalizedScoresForKey(hits, "bm25_max", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void bm25_avg_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("bm25(marqo__lexical_title)")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("bm25(marqo__lexical_title)"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("bm25_avg"), 1.0)
                            .build();
            List<Double> normalized = computeNormalizedScoresForKey(hits, "bm25_avg", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void closeness_retrieval_vector_sum_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("ranking_closeness_metric_f1")).thenReturn(raw);
                when(sf.getDouble("ranking_closeness_metric_f2")).thenReturn(0.0);
                when(sf.featureNames())
                        .thenReturn(
                                Set.of(
                                        "ranking_closeness_metric_f1",
                                        "ranking_closeness_metric_f2"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("closeness_retrieval_vector_sum"), 1.0)
                            .build();
            List<Double> normalized =
                    computeNormalizedScoresForKey(
                            hits, "closeness_retrieval_vector_sum", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void closeness_retrieval_vector_max_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("ranking_closeness_metric_title")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("ranking_closeness_metric_title"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("closeness_retrieval_vector_max"), 1.0)
                            .build();
            List<Double> normalized =
                    computeNormalizedScoresForKey(
                            hits, "closeness_retrieval_vector_max", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }

        @Test
        void closeness_retrieval_vector_avg_normalized_max_to_one() {
            HitGroup hits = new HitGroup();
            for (double raw : RAW_SCORES) {
                Hit hit = new Hit("doc_" + raw, 1.0);
                FeatureData sf = mock(FeatureData.class);
                when(sf.getDouble("ranking_closeness_metric_title")).thenReturn(raw);
                when(sf.featureNames()).thenReturn(Set.of("ranking_closeness_metric_title"));
                hit.setField("summaryfeatures", sf);
                hits.add(hit);
            }
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor addWeights =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("closeness_retrieval_vector_avg"), 1.0)
                            .build();
            List<Double> normalized =
                    computeNormalizedScoresForKey(
                            hits, "closeness_retrieval_vector_avg", addWeights);
            assertNormalizationMaxToOneOthersProportional(normalized);
        }
    }
}
