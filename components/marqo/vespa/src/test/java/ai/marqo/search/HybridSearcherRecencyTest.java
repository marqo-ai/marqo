package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.yahoo.search.Query;
import com.yahoo.search.query.ranking.RankFeatures;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Unit tests for recency scoring functionality in HybridSearcher.
 * Tests the four main recency-related methods:
 * - extractRecencyScore()
 * - applyGlobalScoreModifiers() with recency
 * - postProcessResults() with standalone recency
 * - createSubQuery() recency tensor extraction
 */
class HybridSearcherRecencyTest {
    private HybridSearcher hybridSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    /**
     * Helper method to create a hit with match features including recency score
     */
    private Hit createHitWithRecencyScore(String id, double relevance, Double recencyScore) {
        Hit hit = new Hit(id, relevance);
        if (recencyScore != null) {
            FeatureData matchFeatures = mock(FeatureData.class);
            when(matchFeatures.getDouble("recency_score")).thenReturn(recencyScore);
            hit.setField("matchfeatures", matchFeatures);
        }
        return hit;
    }

    /**
     * Helper method to create a hit with complete match features for global score modifiers
     */
    private Hit createHitWithScoreModifiers(
            String id,
            double relevance,
            double multModifier,
            double addModifier,
            Double recencyScore) {
        Hit hit = new Hit(id, relevance);
        FeatureData matchFeatures = mock(FeatureData.class);
        when(matchFeatures.getDouble("global_mult_modifier")).thenReturn(multModifier);
        when(matchFeatures.getDouble("global_add_modifier")).thenReturn(addModifier);
        if (recencyScore != null) {
            when(matchFeatures.getDouble("recency_score")).thenReturn(recencyScore);
        }
        hit.setField("matchfeatures", matchFeatures);
        return hit;
    }

    /**
     * Helper method to create a query with recency enabled
     */
    private Query createQueryWithRecency(boolean enabled, boolean applyInGlobalPhase) {
        Query query = new Query("search/?query=test");
        query.properties().set("marqo__recency_enabled", enabled);
        query.properties().set("marqo__recency_apply_in_global_ranking_phase", applyInGlobalPhase);
        return query;
    }

    /**
     * Helper method to create a query with additive recency enabled
     */
    private Query createQueryWithAdditiveRecency(
            boolean enabled, boolean applyInGlobalPhase, double addToScoreWeight) {
        Query query = createQueryWithRecency(enabled, applyInGlobalPhase);
        // Set addToScoreWeight in rank features (not properties) - this is where HybridSearcher
        // reads it from
        query.getRanking()
                .getFeatures()
                .put("query(marqo__recency_add_to_score_weight)", addToScoreWeight);
        return query;
    }

    @Nested
    class ExtractRecencyScoreTests {
        @Test
        void shouldExtractRecencyScoreWhenEnabled() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.85));
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 0.8, 0.50));

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            assertThat(result.get(0).getField("marqo__recency_score")).isEqualTo(0.85);
            assertThat(result.get(1).getField("marqo__recency_score")).isEqualTo(0.50);
        }

        @Test
        void shouldSkipExtractionWhenRecencyDisabled() {
            Query query = createQueryWithRecency(false, false);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.85));

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            assertThat(result.get(0).getField("marqo__recency_score")).isNull();
        }

        @Test
        void shouldHandleNullMatchFeatures() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();
            Hit hit = new Hit("index:test/0/doc1", 1.0);
            // No match features set
            hits.add(hit);

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            // Should not crash, just skip this hit
            assertThat(result.get(0).getField("marqo__recency_score")).isNull();
        }

        @Test
        void shouldHandleMissingRecencyScoreInMatchFeatures() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();
            Hit hit = new Hit("index:test/0/doc1", 1.0);
            FeatureData matchFeatures = mock(FeatureData.class);
            // recency_score returns null
            when(matchFeatures.getDouble("recency_score")).thenReturn(null);
            hit.setField("matchfeatures", matchFeatures);
            hits.add(hit);

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            // Should not crash, recency_score should be null
            assertThat(result.get(0).getField("marqo__recency_score")).isNull();
        }

        @Test
        void shouldHandleEmptyHitGroup() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            assertThat(result.size()).isEqualTo(0);
        }

        @Test
        void shouldHandleMultipleHitsWithDifferentScores() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 1.0));
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 0.9, 0.75));
            hits.add(createHitWithRecencyScore("index:test/0/doc3", 0.8, 0.50));
            hits.add(createHitWithRecencyScore("index:test/0/doc4", 0.7, 0.25));
            hits.add(createHitWithRecencyScore("index:test/0/doc5", 0.6, 0.01));

            HitGroup result = hybridSearcher.extractRecencyScore(hits, query, false);

            assertThat(result.size()).isEqualTo(5);
            assertThat(result.get(0).getField("marqo__recency_score")).isEqualTo(1.0);
            assertThat(result.get(1).getField("marqo__recency_score")).isEqualTo(0.75);
            assertThat(result.get(2).getField("marqo__recency_score")).isEqualTo(0.50);
            assertThat(result.get(3).getField("marqo__recency_score")).isEqualTo(0.25);
            assertThat(result.get(4).getField("marqo__recency_score")).isEqualTo(0.01);
        }
    }

    @Nested
    class ApplyGlobalScoreModifiersWithRecencyTests {
        @Test
        void shouldApplyRecencyInGlobalRankingPhase() {
            Query query = createQueryWithRecency(true, true);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 2.0, 0.5, 0.8));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (1.0 * 2.0 + 0.5) * 0.8 = 2.5 * 0.8 = 2.0
            assertThat(result.get(0).getRelevance().getScore()).isEqualTo(2.0);
        }

        @Test
        void shouldDefaultRecencyToOneWhenDisabled() {
            Query query = createQueryWithRecency(true, false);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 2.0, 0.5, 0.5));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (1.0 * 2.0 + 0.5) * 1.0 = 2.5 (recency not applied)
            assertThat(result.get(0).getRelevance().getScore()).isEqualTo(2.5);
        }

        @Test
        void shouldHandleVariousRecencyScores() {
            Query query = createQueryWithRecency(true, true);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 1.0, 0.0, 1.0));
            hits.add(createHitWithScoreModifiers("index:test/0/doc2", 1.0, 1.0, 0.0, 0.5));
            hits.add(createHitWithScoreModifiers("index:test/0/doc3", 1.0, 1.0, 0.0, 0.1));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // All should have mult=1.0, add=0.0, so original score = 1.0
            // doc1: 1.0 * 1.0 = 1.0
            // doc2: 1.0 * 0.5 = 0.5
            // doc3: 1.0 * 0.1 = 0.1
            assertThat(result.get(0).getRelevance().getScore()).isEqualTo(1.0);
            assertThat(result.get(1).getRelevance().getScore()).isEqualTo(0.5);
            assertThat(result.get(2).getRelevance().getScore()).isEqualTo(0.1);
        }

        @Test
        void shouldThrowExceptionForMissingMatchFeatures() {
            Query query = createQueryWithRecency(true, true);
            HitGroup hits = new HitGroup();
            Hit hit = new Hit("index:test/0/doc1", 1.0);
            // No match features set
            hits.add(hit);

            assertThrows(
                    RuntimeException.class,
                    () -> hybridSearcher.applyGlobalScoreModifiers(hits, query, false));
        }

        @Test
        void shouldHandleComplexScoreCalculation() {
            Query query = createQueryWithRecency(true, true);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 0.75, 3.0, 1.0, 0.6));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (0.75 * 3.0 + 1.0) * 0.6 = (2.25 + 1.0) * 0.6 = 3.25 * 0.6 = 1.95
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(1.95, within(0.0001));
        }

        @Test
        void shouldApplyAdditiveRecencyInGlobalRankingPhase() {
            Query query = createQueryWithAdditiveRecency(true, true, 0.5);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 2.0, 0.5, 0.8));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (1.0 * 2.0 + 0.5) + (0.8 * 0.5) = 2.5 + 0.4 = 2.9
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(2.9, within(0.0001));
        }

        @Test
        void shouldUseDefaultMultiplicativeModeWhenAddToScoreWeightIsZero() {
            Query query = createQueryWithAdditiveRecency(true, true, 0.0);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 2.0, 0.5, 0.8));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (1.0 * 2.0 + 0.5) * 0.8 = 2.5 * 0.8 = 2.0 (multiplicative mode)
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(2.0, within(0.0001));
        }

        @Test
        void shouldHandleVariousAdditiveRecencyWeights() {
            // Test different addToScoreWeight values
            double[] weights = {0.1, 0.5, 1.0, 10.0};
            double baseScore = 1.0;
            double multModifier = 1.0;
            double addModifier = 0.0;
            double recencyScore = 0.8;

            for (double weight : weights) {
                Query query = createQueryWithAdditiveRecency(true, true, weight);
                HitGroup hits = new HitGroup();
                hits.add(
                        createHitWithScoreModifiers(
                                "index:test/0/doc1",
                                baseScore,
                                multModifier,
                                addModifier,
                                recencyScore));

                HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

                // Expected: (baseScore * multModifier + addModifier) + (recencyScore * weight)
                double expected =
                        (baseScore * multModifier + addModifier) + (recencyScore * weight);
                assertThat(result.get(0).getRelevance().getScore())
                        .as("Failed for weight=" + weight)
                        .isCloseTo(expected, within(0.0001));
            }
        }

        @Test
        void shouldHandleAdditiveRecencyWithComplexScoreModifiers() {
            Query query = createQueryWithAdditiveRecency(true, true, 2.0);
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 0.75, 3.0, 1.0, 0.6));

            HitGroup result = hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            // Expected: (0.75 * 3.0 + 1.0) + (0.6 * 2.0) = 3.25 + 1.2 = 4.45
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(4.45, within(0.0001));
        }
    }

    @Nested
    class PostProcessResultsWithRecencyTests {
        @Test
        void shouldApplyStandaloneRecencyWithoutGlobalWeights() {
            Query query = createQueryWithRecency(true, true);
            // Set empty global weights to trigger standalone recency path
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.8));
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 0.9, 0.6));

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // Should apply recency directly: score * recencyScore
            assertThat(result.get(0).getRelevance().getScore()).isEqualTo(0.8);
            assertThat(result.get(1).getRelevance().getScore()).isEqualTo(0.54);
        }

        @Test
        void shouldNotApplyRecencyWhenDisabled() {
            Query query = createQueryWithRecency(true, false);
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.5));

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // Score should remain unchanged
            assertThat(result.get(0).getRelevance().getScore()).isEqualTo(1.0);
        }

        @Test
        void shouldRerankAfterApplyingRecency() {
            Query query = createQueryWithRecency(true, true);
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            // Add hits in one order
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 0.5, 1.0)); // 0.5 * 1.0 = 0.5
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 1.0, 0.3)); // 1.0 * 0.3 = 0.3
            hits.add(createHitWithRecencyScore("index:test/0/doc3", 0.6, 0.8)); // 0.6 * 0.8 = 0.48

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // After sorting, doc1 should be first (0.5), then doc3 (0.48), then doc2 (0.3)
            assertThat(result.get(0).getId().toString()).contains("doc1");
            assertThat(result.get(1).getId().toString()).contains("doc3");
            assertThat(result.get(2).getId().toString()).contains("doc2");
        }

        @Test
        void shouldRespectRerankDepthGlobal() {
            // Test that rerankDepthGlobal limits which hits get recency applied
            // No weight tensors, so standalone recency is applied
            Query query = createQueryWithRecency(true, true);

            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.9));
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 0.9, 0.8));
            hits.add(createHitWithRecencyScore("index:test/0/doc3", 0.8, 0.7));

            // Only rerank first 2 hits
            HitGroup result = hybridSearcher.postProcessResults(hits, query, 2, 10, 0, false);

            // First 2 hits get recency applied (standalone recency: score * recencyScore)
            // doc1: 1.0 * 0.9 = 0.9
            // doc2: 0.9 * 0.8 = 0.72
            // doc3: stays at 0.8 (no recency applied because rerankDepthGlobal=2)
            // After sorting by score: doc1 (0.9), doc3 (0.8), doc2 (0.72)
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(0.9, within(0.0001));
            assertThat(result.get(0).getId().toString()).contains("doc1");
            assertThat(result.get(1).getRelevance().getScore()).isCloseTo(0.8, within(0.0001));
            assertThat(result.get(1).getId().toString()).contains("doc3");
            assertThat(result.get(2).getRelevance().getScore()).isCloseTo(0.72, within(0.0001));
            assertThat(result.get(2).getId().toString()).contains("doc2");
        }

        @Test
        void shouldHandlePaginationWithRecency() {
            Query query = createQueryWithRecency(true, true);
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            for (int i = 0; i < 10; i++) {
                hits.add(createHitWithRecencyScore("index:test/0/doc" + i, 1.0 - (i * 0.05), 0.8));
            }

            // Request limit=5, offset=0
            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 5, 0, false);

            assertThat(result.size()).isEqualTo(5);
        }

        @Test
        void shouldApplyAdditiveRecencyWithoutGlobalWeights() {
            Query query = createQueryWithAdditiveRecency(true, true, 0.5);
            // Set empty global weights to trigger standalone recency path
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.8));
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 0.9, 0.6));

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // Additive mode: score + (recencyScore * weight)
            // doc1: 1.0 + (0.8 * 0.5) = 1.0 + 0.4 = 1.4
            // doc2: 0.9 + (0.6 * 0.5) = 0.9 + 0.3 = 1.2
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(1.4, within(0.0001));
            assertThat(result.get(1).getRelevance().getScore()).isCloseTo(1.2, within(0.0001));
        }

        @Test
        void shouldUseMultiplicativeModeWhenAddToScoreWeightIsZeroStandalone() {
            Query query = createQueryWithAdditiveRecency(true, true, 0.0);
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 1.0, 0.8));

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // Multiplicative mode (default): score * recencyScore
            // doc1: 1.0 * 0.8 = 0.8
            assertThat(result.get(0).getRelevance().getScore()).isCloseTo(0.8, within(0.0001));
        }

        @Test
        void shouldRerankCorrectlyAfterAdditiveRecency() {
            Query query = createQueryWithAdditiveRecency(true, true, 1.0);
            RankFeatures rankFeatures = query.getRanking().getFeatures();
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyTensor = Tensor.Builder.of(tensorType).build();
            rankFeatures.put("query(marqo__mult_weights_global)", emptyTensor);
            rankFeatures.put("query(marqo__add_weights_global)", emptyTensor);

            HitGroup hits = new HitGroup();
            // Add hits that will change order after additive recency
            hits.add(createHitWithRecencyScore("index:test/0/doc1", 0.5, 1.0)); // 0.5 + 1.0 = 1.5
            hits.add(createHitWithRecencyScore("index:test/0/doc2", 1.0, 0.3)); // 1.0 + 0.3 = 1.3
            hits.add(createHitWithRecencyScore("index:test/0/doc3", 0.6, 0.8)); // 0.6 + 0.8 = 1.4

            HitGroup result = hybridSearcher.postProcessResults(hits, query, null, 10, 0, false);

            // After sorting: doc1 (1.5), doc3 (1.4), doc2 (1.3)
            assertThat(result.get(0).getId().toString()).contains("doc1");
            assertThat(result.get(1).getId().toString()).contains("doc3");
            assertThat(result.get(2).getId().toString()).contains("doc2");
        }
    }

    @Nested
    class CreateSubQueryRecencyTensorTests {
        @Test
        void shouldExtractRecencyTimestampKeyTensor() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__yql.lexical", "lexical yql");
            query.properties().set("marqo__ranking.lexical.lexical", "bm25");

            // Create recency timestamp key tensor
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor recencyTensor =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("created_at"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_timestamp_key)", recencyTensor);

            // Create fields to rank tensor
            Tensor fieldsToRank =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", fieldsToRank);

            Query subQuery = hybridSearcher.createSubQuery(query, "lexical", "lexical", false);

            // Verify recency tensor was extracted and added to rank features
            RankFeatures features = subQuery.getRanking().getFeatures();
            assertThat(features.getDouble("query(created_at)")).hasValue(1.0);
        }

        @Test
        void shouldHandleNullRecencyTimestampKey() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__yql.lexical", "lexical yql");
            query.properties().set("marqo__ranking.lexical.lexical", "bm25");

            // No recency tensor set

            // Create fields to rank tensor
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor fieldsToRank =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", fieldsToRank);

            // Should not crash
            Query subQuery = hybridSearcher.createSubQuery(query, "lexical", "lexical", false);

            assertThat(subQuery).isNotNull();
        }

        @Test
        void shouldHandleEmptyRecencyTensor() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__yql.lexical", "lexical yql");
            query.properties().set("marqo__ranking.lexical.lexical", "bm25");

            // Create empty recency tensor
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor emptyRecencyTensor = Tensor.Builder.of(tensorType).build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_timestamp_key)", emptyRecencyTensor);

            // Create fields to rank tensor
            Tensor fieldsToRank =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", fieldsToRank);

            Query subQuery = hybridSearcher.createSubQuery(query, "lexical", "lexical", false);

            // Should handle gracefully
            assertThat(subQuery).isNotNull();
        }

        @Test
        void shouldAddMultipleRecencyTensorCells() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__yql.lexical", "lexical yql");
            query.properties().set("marqo__ranking.lexical.lexical", "bm25");

            // Create recency timestamp key tensor with multiple cells
            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor recencyTensor =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("created_at"), 1.0)
                            .cell(TensorAddress.ofLabels("updated_at"), 0.5)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_timestamp_key)", recencyTensor);

            // Create fields to rank tensor
            Tensor fieldsToRank =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", fieldsToRank);

            Query subQuery = hybridSearcher.createSubQuery(query, "lexical", "lexical", false);

            // Verify both cells were added
            RankFeatures features = subQuery.getRanking().getFeatures();
            assertThat(features.getDouble("query(created_at)")).hasValue(1.0);
            assertThat(features.getDouble("query(updated_at)")).hasValue(0.5);
        }

        static Stream<Arguments> applyToSubqueriesCases() {
            return Stream.of(
                    // tensor disabled when apply_to_tensor=false
                    Arguments.of(
                            "tensor",
                            "embedding_similarity",
                            "marqo__tensor_text_field_1",
                            false,
                            true,
                            true),
                    // lexical disabled when apply_to_lexical=false
                    Arguments.of(
                            "lexical", "bm25", "marqo__lexical_text_field_1", true, false, true),
                    // tensor NOT disabled when apply_to_tensor=true
                    Arguments.of(
                            "tensor",
                            "embedding_similarity",
                            "marqo__tensor_text_field_1",
                            true,
                            false,
                            false),
                    // lexical NOT disabled when apply_to_lexical=true
                    Arguments.of(
                            "lexical", "bm25", "marqo__lexical_text_field_1", false, true, false),
                    // defaults (not set) — recency enabled
                    Arguments.of(
                            "tensor",
                            "embedding_similarity",
                            "marqo__tensor_text_field_1",
                            null,
                            null,
                            false));
        }

        @ParameterizedTest
        @MethodSource("applyToSubqueriesCases")
        void shouldRespectApplyToSubqueryFlags(
                String subqueryType,
                String rankProfile,
                String fieldKey,
                Boolean applyToTensor,
                Boolean applyToLexical,
                boolean expectDisabled) {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__yql." + subqueryType, subqueryType + " yql");
            query.properties()
                    .set("marqo__ranking." + subqueryType + "." + subqueryType, rankProfile);
            if (applyToTensor != null) {
                query.properties().set("marqo__recency_apply_to_tensor", applyToTensor);
            }
            if (applyToLexical != null) {
                query.properties().set("marqo__recency_apply_to_lexical", applyToLexical);
            }

            TensorType tensorType = new TensorType.Builder().mapped("p").build();
            Tensor recencyTensor =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("created_at"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_timestamp_key)", recencyTensor);

            Tensor fieldsToRank =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels(fieldKey), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_" + subqueryType + ")", fieldsToRank);

            Query subQuery =
                    hybridSearcher.createSubQuery(query, subqueryType, subqueryType, false);

            RankFeatures features = subQuery.getRanking().getFeatures();
            if (expectDisabled) {
                assertThat(features.getDouble("query(marqo__recency_should_apply_score)"))
                        .hasValue(0.0);
            } else {
                assertThat(features.getDouble("query(marqo__recency_should_apply_score)"))
                        .isEmpty();
            }
        }
    }

    @Nested
    class PreRerankScoreExposureTests {
        @Test
        void shouldNotSetPreRerankScoreWhenPropertyIsFalse() {
            // Default: marqo__expose_pre_rerank_score is not set → false
            Query query = new Query("search/?query=test");
            HitGroup hits = new HitGroup();
            hits.add(createHitWithScoreModifiers("index:test/0/doc1", 1.0, 2.0, 0.5, null));

            hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            assertThat(hits.get(0).getField("marqo__pre_rerank_score")).isNull();
        }

        @Test
        void shouldSetPreRerankScoreWhenPropertyIsTrue() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__expose_pre_rerank_score", true);
            double originalRelevance = 1.0;
            HitGroup hits = new HitGroup();
            hits.add(
                    createHitWithScoreModifiers(
                            "index:test/0/doc1", originalRelevance, 2.0, 0.5, null));

            hybridSearcher.applyGlobalScoreModifiers(hits, query, false);

            assertThat(hits.get(0).getField("marqo__pre_rerank_score"))
                    .isEqualTo(originalRelevance);
        }
    }
}
