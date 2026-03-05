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
import java.util.HashSet;
import java.util.List;
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

    /** Min-max normalization and key stripping: used for both BM25 and closeness. */
    @Nested
    class MinMaxNormalizationTest {

        @Test
        void minMaxNormalize_returns_zero_one_between_min_max() {
            assertThat(HybridSearcher.minMaxNormalize(10.0, 10.0, 30.0)).isEqualTo(0.0);
            assertThat(HybridSearcher.minMaxNormalize(30.0, 10.0, 30.0)).isEqualTo(1.0);
            assertThat(HybridSearcher.minMaxNormalize(20.0, 10.0, 30.0)).isEqualTo(0.5);
        }

        @Test
        void minMaxNormalize_clamps_out_of_range() {
            assertThat(HybridSearcher.minMaxNormalize(-1.0, 0.0, 10.0)).isEqualTo(0.0);
            assertThat(HybridSearcher.minMaxNormalize(11.0, 0.0, 10.0)).isEqualTo(1.0);
        }

        @Test
        void minMaxNormalize_returns_one_when_min_equals_max() {
            assertThat(HybridSearcher.minMaxNormalize(5.0, 5.0, 5.0)).isEqualTo(1.0);
        }

        @Test
        void computeMinMaxPerKey_returns_min_max_for_closeness_single_field() {
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
            Map<String, double[]> result = searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_field_title");
            double[] minMax = result.get("closeness_retrieval_vector_field_title");
            assertThat(minMax).hasSize(2);
            assertThat(minMax[0]).isEqualTo(10.0);
            assertThat(minMax[1]).isEqualTo(30.0);
        }

        @Test
        void computeMinMaxPerKey_aggregate_closeness_sum_two_fields() {
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
            Map<String, double[]> result = searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_sum");
            double[] minMax = result.get("closeness_retrieval_vector_sum");
            assertThat(minMax[0]).isCloseTo(0.6, within(1e-9));
            assertThat(minMax[1]).isEqualTo(1.0);
        }

        @Test
        void computeMinMaxPerKey_includes_both_bm25_and_closeness_keys() {
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
            Map<String, double[]> result = searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("bm25_field_title");
            assertThat(result).containsKey("closeness_retrieval_vector_field_title");
        }
    }

    /**
     * Min-max normalization is applied after aggregation: for aggregate keys (bm25_sum,
     * closeness_retrieval_vector_sum, etc.) we first compute the aggregate per hit, then compute
     * min/max of that aggregated value across hits, then normalize. So the normalized score is
     * based on the aggregate, not on individual field values.
     */
    @Nested
    class NormalizationAfterAggregationTest {

        @Test
        void computeBm25MinMaxPerKey_for_bm25_sum_uses_aggregated_value_per_hit() {
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
            Map<String, double[]> result = searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("bm25_sum");
            double[] minMax = result.get("bm25_sum");
            assertThat(minMax).hasSize(2);
            assertThat(minMax[0]).isEqualTo(3.0);
            assertThat(minMax[1]).isEqualTo(6.0);
        }

        @Test
        void
                computeClosenessMinMaxPerKey_for_closeness_retrieval_vector_sum_uses_aggregated_value_per_hit() {
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
            Map<String, double[]> result = searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(result).containsKey("closeness_retrieval_vector_sum");
            double[] minMax = result.get("closeness_retrieval_vector_sum");
            assertThat(minMax).hasSize(2);
            assertThat(minMax[0]).isCloseTo(0.6, within(1e-9));
            assertThat(minMax[1]).isEqualTo(1.0);
        }
    }

    /**
     * Normalization maps raw custom scores (which may be outside [0,1]) to [0,1], with the
     * minimum score across hits mapping to 0 and the maximum to 1. Five hits with raw scores
     * -2, 0.25, 0.5, 0.75, 5 (inside and outside [0,1]) are used; we assert normalized values
     * are in [0,1] and that min->0 and max->1 for all key types.
     */
    @Nested
    class NormalizationOutputZeroToOneTest {

        private static final double[] RAW_SCORES = {-2.0, 0.25, 0.5, 0.75, 5.0};
        private static final double MIN_RAW = -2.0;
        private static final double MAX_RAW = 5.0;

        /** Build 5 hits, compute minMaxPerKey for the given key, return normalized scores in hit order. */
        private List<Double> computeNormalizedScoresForKey(
                HitGroup hits, String key, Tensor addWeights) {
            HybridSearcher searcher = new HybridSearcher();
            Map<String, double[]> minMaxPerKey =
                    searcher.computeMinMaxPerKey(hits, addWeights, null);
            assertThat(minMaxPerKey).containsKey(key);
            double[] minMax = minMaxPerKey.get(key);
            assertThat(minMax).hasSize(2);
            assertThat(minMax[0]).isEqualTo(MIN_RAW);
            assertThat(minMax[1]).isEqualTo(MAX_RAW);

            HybridSearcher.CustomScoreKeyParsed parsed = HybridSearcher.parseCustomScoreKey(key);
            assertThat(parsed).isNotNull();
            List<Double> normalized = new ArrayList<>();
            for (Hit hit : hits) {
                FeatureData summaryFeatures = (FeatureData) hit.getField("summaryfeatures");
                Double raw =
                        HybridSearcher.extractCustomScoreForHit(
                                null, key, parsed, Set.of(), summaryFeatures);
                assertThat(raw).isNotNull();
                double norm = HybridSearcher.minMaxNormalize(raw, minMax[0], minMax[1]);
                normalized.add(norm);
            }
            return normalized;
        }

        private void assertNormalizationMapsMinToZeroMaxToOne(List<Double> normalized) {
            assertThat(normalized).hasSize(5);
            for (Double n : normalized) {
                assertThat(n).isBetween(0.0, 1.0);
            }
            assertThat(normalized.get(0)).isEqualTo(0.0);
            assertThat(normalized.get(4)).isEqualTo(1.0);
            assertThat(normalized.get(1))
                    .isCloseTo((0.25 - MIN_RAW) / (MAX_RAW - MIN_RAW), within(1e-9));
            assertThat(normalized.get(2))
                    .isCloseTo((0.5 - MIN_RAW) / (MAX_RAW - MIN_RAW), within(1e-9));
            assertThat(normalized.get(3))
                    .isCloseTo((0.75 - MIN_RAW) / (MAX_RAW - MIN_RAW), within(1e-9));
        }

        @Test
        void bm25_single_field_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void bm25_sum_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void bm25_max_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void bm25_avg_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void closeness_retrieval_vector_single_field_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void closeness_retrieval_vector_sum_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void closeness_retrieval_vector_max_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }

        @Test
        void closeness_retrieval_vector_avg_normalized_in_zero_one_min_zero_max_one() {
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
            assertNormalizationMapsMinToZeroMaxToOne(normalized);
        }
    }
}
