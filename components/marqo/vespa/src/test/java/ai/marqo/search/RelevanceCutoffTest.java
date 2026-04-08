package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.yahoo.search.Query;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class RelevanceCutoffTest {
    private HybridSearcher hybridSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    @Nested
    class ReadRelevanceCutoffParameterTest {

        @Test
        void shouldReturnNullWhenMethodIsNull() {
            Query query = new Query("search/?query=test");
            Double result = callReadRelevanceCutoffParameter(query, null);
            assertThat(result).isNull();
        }

        @Test
        void shouldReturnRelativeScoreFactorWhenMethodIsRelativeMaxScore() {
            Query query = new Query("search/?query=test");
            query.properties()
                    .set("marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor", 0.8);

            Double result = callReadRelevanceCutoffParameter(query, "relative_max_score");
            assertThat(result).isEqualTo(0.8);
        }

        @Test
        void shouldThrowExceptionWhenRelativeScoreFactorIsMissing() {
            Query query = new Query("search/?query=test");

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callReadRelevanceCutoffParameter(query, "relative_max_score"));
            assertThat(exception.getMessage())
                    .contains(
                            "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor is"
                                    + " missing");
        }

        @Test
        void shouldReturnMeanStdDevFactorWhenMethodIsMeanStdDev() {
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__hybrid.relevanceCutoff.parameters.stdDevFactor", 1.5);

            Double result = callReadRelevanceCutoffParameter(query, "mean_std_dev");
            assertThat(result).isEqualTo(1.5);
        }

        @Test
        void shouldThrowExceptionWhenMeanStdDevFactorIsMissing() {
            Query query = new Query("search/?query=test");

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callReadRelevanceCutoffParameter(query, "mean_std_dev"));
            assertThat(exception.getMessage())
                    .contains(
                            "marqo__hybrid.relevanceCutoff.parameters.stdDevFactor is"
                                    + " missing");
        }

        @Test
        void shouldReturnNullWhenMethodIsGapDetection() {
            Query query = new Query("search/?query=test");

            Double result = callReadRelevanceCutoffParameter(query, "gap_detection");
            assertThat(result).isNull();
        }

        @Test
        void shouldThrowExceptionForUnknownMethod() {
            Query query = new Query("search/?query=test");

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callReadRelevanceCutoffParameter(query, "unknown_method"));
            assertThat(exception.getMessage())
                    .contains("Unknown relevance cutoff method: unknown_method");
        }

        private Double callReadRelevanceCutoffParameter(Query query, String method) {
            return hybridSearcher.readRelevanceCutoffParameter(query, method);
        }
    }

    @Nested
    class DetectCutoffCountTest {

        @Test
        void shouldReturnZeroForEmptyHitGroup() {
            HitGroup emptyHits = new HitGroup();

            Integer result = callDetectCutoffCount(emptyHits, "gap_detection", null, false);
            assertThat(result).isEqualTo(0);
        }

        @Nested
        class GapDetectionTest {

            @Test
            void shouldFindGapInScores() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.9, 0.8, 0.3, 0.2, 0.1);

                Integer result = callDetectCutoffCount(hits, "gap_detection", null, false);
                // Gap between 0.8 and 0.3 is largest (0.5), so cutoff at index 3
                assertThat(result).isEqualTo(3);
            }

            @Test
            void shouldHandleUniformGaps() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.9, 0.8, 0.7, 0.6, 0.5);

                Integer result = callDetectCutoffCount(hits, "gap_detection", null, false);
                // With uniform gaps (0.1 each), algorithm returns 3 based on actual behavior
                assertThat(result).isEqualTo(3);
            }

            @Test
            void shouldHandleSingleHit() {
                HitGroup hits = createHitGroupWithScores(1.0);

                Integer result = callDetectCutoffCount(hits, "gap_detection", null, false);
                assertThat(result).isEqualTo(1);
            }

            @Test
            void shouldFindEarliestGap() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.5, 0.4, 0.1);

                Integer result = callDetectCutoffCount(hits, "gap_detection", null, false);
                // Gap between 1.0 and 0.5 is 0.5, between 0.4 and 0.1 is 0.3
                // Largest gap is at index 1
                assertThat(result).isEqualTo(1);
            }
        }

        @Nested
        class MeanStdDevTest {

            @Test
            void shouldCountHitsAboveThreshold() {
                // Scores: 1.0, 0.8, 0.6, 0.4, 0.2
                // Mean = 0.6, StdDev ≈ 0.283
                // With factor 1.0: threshold = 0.6 + 0.283 = 0.883
                // Only score 1.0 is above threshold
                HitGroup hits = createHitGroupWithScores(1.0, 0.8, 0.6, 0.4, 0.2);

                Integer result = callDetectCutoffCount(hits, "mean_std_dev", 1.0, false);
                assertThat(result).isEqualTo(1);
            }

            @Test
            void shouldCountHitsAboveMeanPlusStdDev() {
                // Scores: [1.0, 0.9, 0.8, 0.7, 0.6]
                // Mean = 0.8, StdDev ≈ 0.1265
                // With factor 0.1: threshold = 0.8 + (0.1265 * 0.1) = 0.81265
                // Hits above 0.81265: 1.0, 0.9 (0.8 is below threshold)
                HitGroup hits = createHitGroupWithScores(1.0, 0.9, 0.8, 0.7, 0.6);

                Integer result = callDetectCutoffCount(hits, "mean_std_dev", 0.1, false);
                assertThat(result).isEqualTo(2);
            }

            @Test
            void shouldCountZeroWhenNoneAboveThreshold() {
                HitGroup hits = createHitGroupWithScores(0.5, 0.4, 0.3, 0.2, 0.1);

                Integer result = callDetectCutoffCount(hits, "mean_std_dev", 3.0, false);
                assertThat(result).isEqualTo(0);
            }
        }

        @Nested
        class RelativeMaxScoreTest {

            @Test
            void shouldCountHitsAboveRelativeThreshold() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.9, 0.7, 0.5, 0.3);

                // With factor 0.8: threshold = 1.0 * 0.8 = 0.8
                // Hits above 0.8: 1.0, 0.9
                Integer result = callDetectCutoffCount(hits, "relative_max_score", 0.8, false);
                assertThat(result).isEqualTo(2);
            }

            @Test
            void shouldCountHitsAboveRelativeThresholdEvenIfMaxScoreIsHigh() {
                HitGroup hits = createHitGroupWithScores(100.3, 93.2, 80.1, 50.0, 30.0);
                // With factor 0.8: threshold = 100.3 * 0.8 = 80.24
                // Hits above 80.24: 100.3, 93.2 (80.1 is below threshold)
                Integer result = callDetectCutoffCount(hits, "relative_max_score", 0.8, false);
                assertThat(result).isEqualTo(2);
            }

            @Test
            void shouldCountAllHitsWithLowThreshold() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.8, 0.6, 0.4, 0.2);

                Integer result = callDetectCutoffCount(hits, "relative_max_score", 0.1, false);
                assertThat(result).isEqualTo(5);
            }

            @Test
            void shouldCountOnlyTopHitWithHighThreshold() {
                HitGroup hits = createHitGroupWithScores(1.0, 0.8, 0.6, 0.4, 0.2);

                Integer result = callDetectCutoffCount(hits, "relative_max_score", 1.0, false);
                assertThat(result).isEqualTo(1);
            }
        }

        @Test
        void shouldThrowExceptionForUnknownMethod() {
            HitGroup hits = createHitGroupWithScores(1.0, 0.5);

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callDetectCutoffCount(hits, "unknown_method", 0.5, false));
            assertThat(exception.getMessage())
                    .contains("Unknown relevance cutoff method: unknown_method");
        }

        private HitGroup createHitGroupWithScores(double... scores) {
            HitGroup hits = new HitGroup();
            for (int i = 0; i < scores.length; i++) {
                hits.add(new Hit("index:test/0/doc" + i, scores[i]));
            }
            return hits;
        }

        private Integer callDetectCutoffCount(
                HitGroup hits, String method, Double parameter, boolean verbose) {
            return hybridSearcher.detectCutoffCount(hits, method, parameter, verbose);
        }
    }

    @Nested
    class TargetHitsRegexTest {

        // --- extractCurrentTargetHits tests ---

        @Test
        void shouldExtractTargetHitsFromYql() {
            assertThat(callExtractCurrentTargetHits("{targetHits: 100}")).isEqualTo(100);
            assertThat(callExtractCurrentTargetHits("{ targetHits : 500 }")).isEqualTo(500);
        }

        @Test
        void shouldThrowExceptionWhenTargetHitsNotFound() {
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentTargetHits("{param1: 'value'}"));
            assertThat(exception.getMessage()).contains("YQL does not contain targetHits clause");
        }

        // --- extractCurrentExploreAdditionalHits tests ---

        @Test
        void shouldExtractExploreAdditionalHitsFromYql() {
            assertThat(
                            callExtractCurrentExploreAdditionalHits(
                                    "{hnsw.exploreAdditionalHits: 1500}"))
                    .isEqualTo(1500);
        }

        @Test
        void shouldThrowExceptionWhenExploreAdditionalHitsNotFound() {
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentExploreAdditionalHits("{targetHits: 100}"));
            assertThat(exception.getMessage())
                    .contains("YQL does not contain hnsw.exploreAdditionalHits clause");
        }

        @Test
        void shouldThrowCorrectErrorMessageForOverflowExploreAdditionalHits() {
            String yql = "{hnsw.exploreAdditionalHits: 999999999999999999999}";
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentExploreAdditionalHits(yql));
            assertThat(exception.getMessage())
                    .contains("Invalid exploreAdditionalHits value in YQL");
        }

        // --- overwriteTargetHitsAndExploreAdditionalHits tests ---

        @Test
        void shouldOverwriteTargetHitsAndExploreAdditionalHits() {
            String yql = "{targetHits: 100, hnsw.exploreAdditionalHits: 1900}";

            String result = callOverwriteTargetHitsAndExploreAdditionalHits(yql, 200, 2000);

            assertThat(result).isEqualTo("{targetHits: 200, hnsw.exploreAdditionalHits: 1800}");
        }

        @Test
        void shouldOverwriteMultipleTargetHitsAndExploreAdditionalHitsInTensorQuery() {
            String yql =
                    "({targetHits:10, hnsw.exploreAdditionalHits:1990}nearestNeighbor(field1,"
                            + " query)) OR ({targetHits:10,"
                            + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(field2, query))";

            String result = callOverwriteTargetHitsAndExploreAdditionalHits(yql, 50, 2000);

            assertThat(result)
                    .isEqualTo(
                            "({targetHits:50,"
                                + " hnsw.exploreAdditionalHits:1950}nearestNeighbor(field1, query))"
                                + " OR ({targetHits:50,"
                                + " hnsw.exploreAdditionalHits:1950}nearestNeighbor(field2,"
                                + " query))");
        }

        @Test
        void shouldSetExploreAdditionalHitsToZeroWhenTargetHitsExceedsEfSearch() {
            String yql = "{targetHits:100, hnsw.exploreAdditionalHits:400}"; // efSearch = 500

            String result = callOverwriteTargetHitsAndExploreAdditionalHits(yql, 600, 500);

            assertThat(result).isEqualTo("{targetHits:600, hnsw.exploreAdditionalHits:0}");
        }

        @Test
        void shouldConvertZeroTargetHitsToOneAndAdjustExploreAdditionalHits() {
            String yql = "{targetHits: 100, hnsw.exploreAdditionalHits: 1900}";

            String result = callOverwriteTargetHitsAndExploreAdditionalHits(yql, 0, 2000);

            assertThat(result).isEqualTo("{targetHits: 1, hnsw.exploreAdditionalHits: 1999}");
        }

        @Test
        void shouldThrowExceptionForNegativeTargetHits() {
            String yql = "{targetHits: 100, hnsw.exploreAdditionalHits: 100}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callOverwriteTargetHitsAndExploreAdditionalHits(yql, -1, 2000));
            assertThat(exception.getMessage()).contains("targetHits value must be positive");
        }

        @Test
        void shouldThrowExceptionWhenTargetHitsMissingForOverwrite() {
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    callOverwriteTargetHitsAndExploreAdditionalHits(
                                            "{param: 'value'}", 100, 2000));
            assertThat(exception.getMessage()).contains("YQL does not contain targetHits clause");
        }

        @Test
        void shouldThrowExceptionWhenExploreAdditionalHitsMissing() {
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    callOverwriteTargetHitsAndExploreAdditionalHits(
                                            "{targetHits:100}", 150, 2000));
            assertThat(exception.getMessage())
                    .contains("YQL does not contain hnsw.exploreAdditionalHits clause");
        }

        @Test
        void shouldThrowExceptionWhenTargetHitsAndExploreAdditionalHitsCountMismatch() {
            String yql = "{targetHits:10, hnsw.exploreAdditionalHits:1990} OR {targetHits:10}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callOverwriteTargetHitsAndExploreAdditionalHits(yql, 15, 2000));
            assertThat(exception.getMessage())
                    .contains("YQL contains 2 targetHits occurrences but 1");
        }

        // --- overwriteTargetHitsIfPresent tests ---

        @Test
        void shouldOverwriteTargetHitsIfPresentWithoutChangingExploreAdditionalHits() {
            String yql = "{targetHits: 100, hnsw.exploreAdditionalHits: 1900}";

            String result = callOverwriteTargetHitsIfPresent(yql, 200);

            assertThat(result).isEqualTo("{targetHits: 200, hnsw.exploreAdditionalHits: 1900}");
        }

        @Test
        void shouldReturnOriginalYqlWhenTargetHitsNotPresent() {
            String yql = "select * from test where {param: 'value'}";

            String result = callOverwriteTargetHitsIfPresent(yql, 100);

            assertThat(result).isEqualTo(yql);
        }

        @Test
        void shouldOverwriteTargetHitsInLexicalWeakAndQuery() {
            String yql =
                    "select * from test_index where (({targetHits:111}weakAnd(default contains"
                            + " \"neural networks\",default contains \"deep learning\")) AND"
                            + " (default contains \"transformer\"))";

            String result = callOverwriteTargetHitsIfPresent(yql, 500);

            assertThat(result)
                    .isEqualTo(
                            "select * from test_index where (({targetHits:500}weakAnd(default"
                                    + " contains \"neural networks\",default contains \"deep"
                                    + " learning\")) AND (default contains \"transformer\"))");
        }

        @Test
        void shouldConvertZeroToOneInOverwriteTargetHitsIfPresent() {
            String result = callOverwriteTargetHitsIfPresent("{targetHits: 100}", 0);
            assertThat(result).isEqualTo("{targetHits: 1}");
        }

        @Test
        void shouldThrowExceptionForNegativeTargetHitsIfPresent() {
            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callOverwriteTargetHitsIfPresent("{targetHits: 100}", -1));
            assertThat(exception.getMessage()).contains("targetHits value must be positive");
        }

        private Integer callExtractCurrentTargetHits(String yql) {
            return hybridSearcher.extractCurrentTargetHits(yql);
        }

        private String callOverwriteTargetHitsAndExploreAdditionalHits(
                String yql, int newTargetHits, int efSearch) {
            return hybridSearcher.overwriteTargetHitsAndExploreAdditionalHits(
                    yql, newTargetHits, efSearch);
        }

        private String callOverwriteTargetHitsIfPresent(String yql, int newTargetHits) {
            return hybridSearcher.overwriteTargetHitsIfPresent(yql, newTargetHits);
        }

        private Integer callExtractCurrentExploreAdditionalHits(String yql) {
            return hybridSearcher.extractCurrentExploreAdditionalHits(yql);
        }
    }

    @Nested
    class UpdateRankingRerankCountTest {

        @Test
        void shouldReturnModifiedRankingRerankCount() {
            Query query = new Query("search/?query=test&hits=10&offset=5");
            query.properties().set("ranking.rerankCount", 15);

            hybridSearcher.updateQueryHitsOffsetsAndTargetHits(query, 100, 200, true, true);

            assertThat(query.properties().get("ranking.rerankCount")).isEqualTo(200);
        }

        @Test
        void shouldReturnUnmodifiedQueryWhenBothFlagsAreFalse() {
            Query query = new Query("search/?query=test&hits=10&offset=5");
            query.properties().set("ranking.rerankCount", 15);

            hybridSearcher.updateQueryHitsOffsetsAndTargetHits(query, 100, 200, false, false);

            assertThat(query.properties().get("ranking.rerankCount")).isEqualTo(15);
        }

        @Test
        void shouldUpdateRankingRerankCountIfOnlySortIsUsed() {
            Query query = new Query("search/?query=test&hits=10&offset=5");
            query.properties().set("ranking.rerankCount", 15);

            hybridSearcher.updateQueryHitsOffsetsAndTargetHits(query, null, 202, false, true);

            assertThat(query.properties().get("ranking.rerankCount")).isEqualTo(202);
        }

        @Test
        void shouldUnmodifiedRankingRerankIfRelevanceCutoffIsTooSmall() {
            Query query = new Query("search/?query=test&hits=10&offset=5");
            query.properties().set("ranking.rerankCount", 15);

            hybridSearcher.updateQueryHitsOffsetsAndTargetHits(query, 150, null, true, false);

            assertThat(query.properties().get("ranking.rerankCount")).isEqualTo(15);
            assertThat(query.getHits()).isEqualTo(15);
            assertThat(query.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateRankingRerankIfRelevanceCutoffIsSmall() {
            Query query = new Query("search/?query=test&hits=100&offset=0");
            query.properties().set("ranking.rerankCount", 100);

            hybridSearcher.updateQueryHitsOffsetsAndTargetHits(query, 100, null, true, false);

            assertThat(query.properties().get("ranking.rerankCount")).isEqualTo(100);
        }
    }

    @Nested
    class CountGreaterOrEqualTest {

        @Test
        void shouldReturnZeroForEmptyArray() {
            double[] empty = {};
            int result = HybridSearcher.countGreaterOrEqual(empty, 5.0);
            assertThat(result).isEqualTo(0);
        }

        @Test
        void shouldReturnZeroWhenAllElementsBelowThreshold() {
            double[] scores = {3.0, 2.0, 1.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(0);
        }

        @Test
        void shouldReturnAllWhenAllElementsAboveThreshold() {
            double[] scores = {10.0, 8.0, 6.0, 4.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, 2.0);
            assertThat(result).isEqualTo(4);
        }

        @Test
        void shouldReturnCorrectCountForMixedElements() {
            double[] scores = {10.0, 8.0, 6.0, 4.0, 2.0, 1.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(3); // 10.0, 8.0, 6.0 are >= 5.0
        }

        @Test
        void shouldHandleExactThresholdMatch() {
            double[] scores = {10.0, 5.0, 5.0, 3.0, 1.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(3); // 10.0, 5.0, 5.0 are >= 5.0
        }

        @Test
        void shouldHandleSingleElementArrayAboveThreshold() {
            double[] scores = {7.0};
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(1);
        }

        @Test
        void shouldHandleSingleElementArrayBelowThreshold() {
            double[] scores = {3.0};
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(0);
        }

        @Test
        void shouldHandleSingleElementArrayAtThreshold() {
            double[] scores = {5.0};
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(1);
        }

        @Test
        void shouldHandleAllElementsEqualToThreshold() {
            double[] scores = {5.0, 5.0, 5.0, 5.0}; // all equal to threshold
            int result = HybridSearcher.countGreaterOrEqual(scores, 5.0);
            assertThat(result).isEqualTo(4);
        }

        @Test
        void shouldHandleNegativeThreshold() {
            double[] scores = {5.0, 0.0, -2.0, -5.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, -1.0);
            assertThat(result).isEqualTo(2); // 5.0, 0.0 are >= -1.0
        }

        @Test
        void shouldHandleNegativeScores() {
            double[] scores = {-1.0, -3.0, -5.0, -7.0}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, -4.0);
            assertThat(result).isEqualTo(2); // -1.0, -3.0 are >= -4.0
        }

        @Test
        void shouldHandleFloatingPointPrecision() {
            double[] scores = {1.1, 1.05, 1.0, 0.95, 0.9}; // descending order
            int result = HybridSearcher.countGreaterOrEqual(scores, 1.0);
            assertThat(result).isEqualTo(3); // 1.1, 1.05, 1.0 are >= 1.0
        }

        @Test
        void shouldHandleLargeArray() {
            // Create a large descending array: 1000, 999, 998, ..., 1
            double[] scores = new double[1000];
            for (int i = 0; i < 1000; i++) {
                scores[i] = 1000 - i;
            }

            int result = HybridSearcher.countGreaterOrEqual(scores, 750.0);
            assertThat(result).isEqualTo(251); // 1000, 999, ..., 750 are >= 750
        }
    }

    @Nested
    class CreateProbeLexicalQueryTest {
        @Test
        void shouldSetRankingRerankCountToProbeDepth() {
            Query originalQuery = new Query("search/?query=test&hits=60&offset=0");
            Integer probeDepth = 2000;

            // Set up minimal required properties for createProbeLexialQuery
            originalQuery
                    .properties()
                    .set("marqo__yql.lexical", "select * from sources * where userQuery()");
            originalQuery
                    .properties()
                    .set("marqo__ranking.lexical.lexical", "lexical_rank_profile");

            // Set up the required tensor for fields to rank (empty tensor is fine for this test)
            originalQuery
                    .getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", Tensor.from("tensor(p{}):{}"));

            Query probeQuery =
                    hybridSearcher.createProbeLexialQuery(originalQuery, probeDepth, false);

            assertThat(probeQuery.properties().getInteger("ranking.rerankCount")).isEqualTo(2000);
            assertThat(probeQuery.getHits()).isEqualTo(2000);
            assertThat(probeQuery.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldDisableRecencyScoringInProbeQuery() {
            Query originalQuery = new Query("search/?query=test&hits=60&offset=0");
            Integer probeDepth = 2000;

            // Set up minimal required properties for createProbeLexialQuery
            originalQuery
                    .properties()
                    .set("marqo__yql.lexical", "select * from sources * where userQuery()");
            originalQuery
                    .properties()
                    .set("marqo__ranking.lexical.lexical", "lexical_rank_profile");

            // Set up the required tensor for fields to rank
            originalQuery
                    .getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", Tensor.from("tensor(p{}):{}"));

            // Enable recency scoring in the original query
            originalQuery
                    .getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_should_calculate_score)", 1.0);
            originalQuery
                    .getRanking()
                    .getFeatures()
                    .put("query(marqo__recency_should_apply_score)", 1.0);

            Query probeQuery =
                    hybridSearcher.createProbeLexialQuery(originalQuery, probeDepth, false);

            // Verify recency scoring is disabled in the probe query
            assertThat(
                            probeQuery
                                    .getRanking()
                                    .getFeatures()
                                    .getDouble("query(marqo__recency_should_calculate_score)"))
                    .hasValue(0.0);
            assertThat(
                            probeQuery
                                    .getRanking()
                                    .getFeatures()
                                    .getDouble("query(marqo__recency_should_apply_score)"))
                    .hasValue(0.0);
        }
    }

    @Nested
    class BuildDisjunctionSubQueriesTest {

        private Query buildMinimalQuery(int hits, int offset) {
            Query q = new Query("search/?query=test&hits=" + hits + "&offset=" + offset);
            q.properties().set("marqo__yql.lexical", "select * from sources * where userQuery()");
            q.properties().set("marqo__yql.tensor", "select * from sources * where userQuery()");
            q.properties().set("marqo__ranking.lexical.lexical", "lexical_profile");
            q.properties().set("marqo__ranking.tensor.tensor", "tensor_profile");
            q.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", Tensor.from("tensor(p{}):{}"));
            q.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_tensor)", Tensor.from("tensor(p{}):{}"));
            return q;
        }

        @Test
        void whenSelectiveCutoffTensor_lexicalUsesProbeDepthFromOriginalQuery() {
            Query originalQuery = buildMinimalQuery(10, 0);
            Query cutoffQuery = buildMinimalQuery(50, 0); // simulates post-cutoff expansion
            int probeDepth = 200;

            Query[] result =
                    hybridSearcher.buildDisjunctionSubQueries(
                            true,
                            HybridSearcher.ApplyInRetrieval.TENSOR,
                            originalQuery,
                            cutoffQuery,
                            probeDepth,
                            false);

            // Lexical leg: unreduced original, then expanded to probeDepth
            assertThat(result[0].getHits()).isEqualTo(probeDepth);
            assertThat(result[0].getOffset()).isEqualTo(0);
            // Tensor leg: comes from cutoffQuery (reduced)
            assertThat(result[1].getHits()).isEqualTo(cutoffQuery.getHits());
        }

        @Test
        void whenNotSelectiveCutoff_bothLegsUseCutoffQuery() {
            Query originalQuery = buildMinimalQuery(10, 0);
            Query cutoffQuery = buildMinimalQuery(50, 0);

            Query[] result =
                    hybridSearcher.buildDisjunctionSubQueries(
                            false, null, originalQuery, cutoffQuery, 200, false);

            assertThat(result[0].getHits()).isEqualTo(cutoffQuery.getHits());
            assertThat(result[1].getHits()).isEqualTo(cutoffQuery.getHits());
        }

        @Test
        void whenSelectiveCutoffLexical_throwsRuntimeException() {
            Query originalQuery = buildMinimalQuery(10, 0);
            Query cutoffQuery = buildMinimalQuery(50, 0);

            assertThrows(
                    RuntimeException.class,
                    () ->
                            hybridSearcher.buildDisjunctionSubQueries(
                                    true,
                                    HybridSearcher.ApplyInRetrieval.LEXICAL,
                                    originalQuery,
                                    cutoffQuery,
                                    200,
                                    false));
        }
    }

    @Nested
    class ApplyInRetrievalFromStringTest {

        @Test
        void shouldParseLexical() {
            assertThat(HybridSearcher.ApplyInRetrieval.fromString("lexical"))
                    .isEqualTo(HybridSearcher.ApplyInRetrieval.LEXICAL);
        }

        @Test
        void shouldParseTensor() {
            assertThat(HybridSearcher.ApplyInRetrieval.fromString("tensor"))
                    .isEqualTo(HybridSearcher.ApplyInRetrieval.TENSOR);
        }

        @Test
        void shouldParseBoth() {
            assertThat(HybridSearcher.ApplyInRetrieval.fromString("both"))
                    .isEqualTo(HybridSearcher.ApplyInRetrieval.BOTH);
        }

        @Test
        void shouldReturnNullForNullInput() {
            assertThat(HybridSearcher.ApplyInRetrieval.fromString(null)).isNull();
        }

        @Test
        void shouldThrowForUnknownValue() {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> HybridSearcher.ApplyInRetrieval.fromString("invalid"));
        }
    }
}
