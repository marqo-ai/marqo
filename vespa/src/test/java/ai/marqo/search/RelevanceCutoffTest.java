package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.yahoo.search.Query;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class RelevanceCutoffTest {
    private HybridSearcher hybridSearcher;
    private Searcher downstreamSearcher;

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

        @Test
        void shouldExtractTargetHitsFromValidYql() {
            String yql = "select * from sources * where {targetHits: 100}";

            Integer result = callExtractCurrentTargetHits(yql);
            assertThat(result).isEqualTo(100);
        }

        @Test
        void shouldExtractTargetHitsWithWhitespace() {
            String yql = "select * from sources * where { targetHits : 500 }";

            Integer result = callExtractCurrentTargetHits(yql);
            assertThat(result).isEqualTo(500);
        }

        @Test
        void shouldExtractTargetHitsFromComplexYql() {
            String yql =
                    "select * from sources * where {param1: 'value', targetHits: 250, param2:"
                            + " true}";

            Integer result = callExtractCurrentTargetHits(yql);
            assertThat(result).isEqualTo(250);
        }

        @Test
        void shouldThrowExceptionWhenTargetHitsNotFound() {
            String yql = "select * from sources * where {param1: 'value', param2: 100}";

            RuntimeException exception =
                    assertThrows(RuntimeException.class, () -> callExtractCurrentTargetHits(yql));
            assertThat(exception.getMessage()).contains("YQL does not contain targetHits clause");
        }

        @Test
        void shouldThrowExceptionForInvalidTargetHitsValue() {
            String yql = "select * from sources * where {targetHits: invalid}";

            RuntimeException exception =
                    assertThrows(RuntimeException.class, () -> callExtractCurrentTargetHits(yql));
            // The regex doesn't match "invalid" as a number, so it throws "YQL does not contain
            // targetHits clause"
            assertThat(exception.getMessage()).contains("YQL does not contain targetHits clause");
        }

        @Test
        void shouldOverwriteTargetHitsInYql() {
            String originalYql =
                    "select * from sources * where {targetHits: 100, hnsw.exploreAdditionalHits:"
                            + " 1900}";

            String result = callOverwriteTargetHits(originalYql, 200, 2000);

            assertThat(result)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 200,"
                                    + " hnsw.exploreAdditionalHits: 1800}");
        }

        @Test
        void shouldOverwriteTargetHitsWithWhitespace() {
            String originalYql =
                    "select * from sources * where { targetHits : 150, hnsw.exploreAdditionalHits :"
                            + " 1850 }";

            String result = callOverwriteTargetHits(originalYql, 300, 2000);
            assertThat(result)
                    .isEqualTo(
                            "select * from sources * where { targetHits : 300,"
                                    + " hnsw.exploreAdditionalHits : 1700 }");
        }

        @Test
        void shouldOverwriteTargetHitsInComplexYql() {
            String originalYql =
                    "select * from sources * where {param1: 'value', targetHits: 75,"
                            + " hnsw.exploreAdditionalHits: 1925, param2: true}";

            String result = callOverwriteTargetHits(originalYql, 125, 2000);
            assertThat(result)
                    .isEqualTo(
                            "select * from sources * where {param1: 'value', targetHits: 125,"
                                    + " hnsw.exploreAdditionalHits: 1875, param2: true}");
        }

        @Test
        void shouldThrowExceptionWhenOverwritingNonExistentTargetHits() {
            String yql = "select * from sources * where {param1: 'value'}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class, () -> callOverwriteTargetHits(yql, 100, 2000));
            assertThat(exception.getMessage()).contains("YQL does not contain targetHits clause");
        }

        @Test
        void shouldThrowExceptionForNegativeTargetHits() {
            String yql =
                    "select * from sources * where {targetHits: 100, hnsw.exploreAdditionalHits:"
                            + " 100}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class, () -> callOverwriteTargetHits(yql, -1, 2000));
            assertThat(exception.getMessage()).contains("targetHits value must be positive");
        }

        @Test
        void shouldConvertZeroTargetHitsToOne() {
            String originalYql =
                    "select * from sources * where {targetHits: 100, hnsw.exploreAdditionalHits:"
                            + " 1900}";

            String result = callOverwriteTargetHits(originalYql, 0, 2000);
            assertThat(result)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 1,"
                                    + " hnsw.exploreAdditionalHits: 1999}");
        }

        @Test
        void shouldHandleComplexYqlWithMultipleTargetHitsAndHnswParameters() {
            // Test with complex YQL containing multiple targetHits and hnsw.exploreAdditionalHits
            String originalYql =
                    "({targetHits:10, approximate:True,"
                        + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_title,"
                        + " marqo__query_embedding)) OR ({targetHits:10, approximate:True,"
                        + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_content,"
                        + " marqo__query_embedding))";

            String result = callOverwriteTargetHits(originalYql, 15, 2000);

            assertThat(result)
                    .isEqualTo(
                            "({targetHits:15, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1985}nearestNeighbor(marqo__embeddings_title,"
                                + " marqo__query_embedding)) OR ({targetHits:15, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1985}nearestNeighbor(marqo__embeddings_content,"
                                + " marqo__query_embedding))");
        }

        @Test
        void shouldUpdateHnswExploreAdditionalHitsWithDifferentValues() {
            // Test various newTargetHits values to verify the 2000-newTargetHits formula
            String originalYql = "{targetHits:50, hnsw.exploreAdditionalHits:1950}";

            // Test with newTargetHits = 100, should result in hnsw.exploreAdditionalHits = 1900
            String result1 = callOverwriteTargetHits(originalYql, 100, 2000);
            assertThat(result1).isEqualTo("{targetHits:100, hnsw.exploreAdditionalHits:1900}");

            // Test with newTargetHits = 1, should result in hnsw.exploreAdditionalHits = 1999
            String result2 = callOverwriteTargetHits(originalYql, 1, 2000);
            assertThat(result2).isEqualTo("{targetHits:1, hnsw.exploreAdditionalHits:1999}");
        }

        @Test
        void shouldHandleEdgeCaseWhenTargetHitsEqualsTwo() {
            // Test boundary condition where newTargetHits = 2000
            String originalYql = "{targetHits:10, hnsw.exploreAdditionalHits:1990}";

            String result = callOverwriteTargetHits(originalYql, 2000, 2000);
            assertThat(result).isEqualTo("{targetHits:2000, hnsw.exploreAdditionalHits:0}");
        }

        @Test
        void shouldHandleHnswExploreAdditionalHitsWithWhitespace() {
            // Test hnsw.exploreAdditionalHits with various whitespace patterns
            String originalYql = "{targetHits: 25, hnsw.exploreAdditionalHits : 1975}";

            String result = callOverwriteTargetHits(originalYql, 50, 2000);
            assertThat(result).isEqualTo("{targetHits: 50, hnsw.exploreAdditionalHits : 1950}");
        }

        @Test
        void shouldThrowExceptionWhenHnswExploreAdditionalHitsIsMissing() {
            // Test that YQL without hnsw.exploreAdditionalHits throws an error
            String originalYql = "{targetHits:100, approximate:True}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callOverwriteTargetHits(originalYql, 150, 2000));
            assertThat(exception.getMessage())
                    .contains(
                            "YQL does not contain hnsw.exploreAdditionalHits clause, but targetHits"
                                    + " is present. Both parameters must be present together.");
        }

        @Test
        void shouldThrowExceptionWhenTargetHitsAndHnswCountMismatch() {
            // Test with mismatched counts: 2 targetHits but 1 hnsw.exploreAdditionalHits
            String originalYql =
                    "{targetHits:10, hnsw.exploreAdditionalHits:1990} OR {targetHits:10}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callOverwriteTargetHits(originalYql, 15, 2000));
            assertThat(exception.getMessage())
                    .contains(
                            "YQL contains 2 targetHits occurrences but 1 hnsw.exploreAdditionalHits"
                                    + " occurrences");
        }

        @Test
        void shouldExtractExploreAdditionalHitsFromValidYql() {
            String yql = "select * from sources * where {hnsw.exploreAdditionalHits: 1500}";

            Integer result = callExtractCurrentExploreAdditionalHits(yql);
            assertThat(result).isEqualTo(1500);
        }

        @Test
        void shouldExtractExploreAdditionalHitsWithWhitespace() {
            String yql = "select * from sources * where { hnsw.exploreAdditionalHits : 1750 }";

            Integer result = callExtractCurrentExploreAdditionalHits(yql);
            assertThat(result).isEqualTo(1750);
        }

        @Test
        void shouldExtractExploreAdditionalHitsFromComplexYql() {
            String yql =
                    "select * from sources * where {param1: 'value', targetHits: 250,"
                            + " hnsw.exploreAdditionalHits: 1750, param2: true}";

            Integer result = callExtractCurrentExploreAdditionalHits(yql);
            assertThat(result).isEqualTo(1750);
        }

        @Test
        void shouldThrowExceptionWhenExploreAdditionalHitsNotFound() {
            String yql = "select * from sources * where {targetHits: 100, param1: 'value'}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentExploreAdditionalHits(yql));
            assertThat(exception.getMessage())
                    .contains("YQL does not contain hnsw.exploreAdditionalHits clause");
        }

        @Test
        void shouldThrowExceptionForInvalidExploreAdditionalHitsValue() {
            String yql = "select * from sources * where {hnsw.exploreAdditionalHits: invalid}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentExploreAdditionalHits(yql));
            // The regex doesn't match "invalid" as a number, so it throws "YQL does not contain
            // clause"
            assertThat(exception.getMessage())
                    .contains("YQL does not contain hnsw.exploreAdditionalHits clause");
        }

        @Test
        void shouldThrowCorrectErrorMessageForInvalidExploreAdditionalHitsNumber() {
            // Test that the error message correctly mentions "exploreAdditionalHits" not
            // "targetHits"
            // This tests the fix for the copy-paste error in the error message
            String yql =
                    "select * from sources * where {hnsw.exploreAdditionalHits:"
                            + " 999999999999999999999}";

            RuntimeException exception =
                    assertThrows(
                            RuntimeException.class,
                            () -> callExtractCurrentExploreAdditionalHits(yql));
            // The number is too large to parse as Integer, should throw NumberFormatException
            // The error message should mention "exploreAdditionalHits" not "targetHits"
            assertThat(exception.getMessage())
                    .contains("Invalid exploreAdditionalHits value in YQL");
            assertThat(exception.getMessage()).doesNotContain("Invalid targetHits value");
        }

        @Test
        void shouldExtractFirstExploreAdditionalHitsWhenMultipleOccurrences() {
            // Test with multiple hnsw.exploreAdditionalHits - should return the first one
            String yql =
                    "select * from sources * where {hnsw.exploreAdditionalHits: 1500} and"
                            + " {hnsw.exploreAdditionalHits: 1800}";

            Integer result = callExtractCurrentExploreAdditionalHits(yql);
            assertThat(result).isEqualTo(1500); // Should return the first occurrence
        }

        @Test
        void shouldTestEfSearchLogicWithDifferentValues() {
            // Test the efSearch calculation: efSearch = targetHits + exploreAdditionalHits
            // When newTargetHits changes, newExploreAdditionalHits = efSearch - newTargetHits

            // Example: targetHits=50, exploreAdditionalHits=1950, so efSearch=2000
            // When newTargetHits=100, newExploreAdditionalHits should be 2000-100=1900
            String originalYql = "{targetHits:50, hnsw.exploreAdditionalHits:1950}";

            String result = callOverwriteTargetHits(originalYql, 100, 2000);
            assertThat(result).isEqualTo("{targetHits:100, hnsw.exploreAdditionalHits:1900}");
        }

        @Test
        void shouldMaintainEfSearchConstantAcrossUpdates() {
            // Test that efSearch (targetHits + exploreAdditionalHits) remains constant
            String originalYql =
                    "{targetHits:300, hnsw.exploreAdditionalHits:1200}"; // efSearch = 1500

            String result = callOverwriteTargetHits(originalYql, 500, 1500);
            assertThat(result).isEqualTo("{targetHits:500, hnsw.exploreAdditionalHits:1000}");

            // Verify the total remains 1500
            Integer newTargetHits = callExtractCurrentTargetHits(result);
            Integer newExploreAdditionalHits = callExtractCurrentExploreAdditionalHits(result);
            assertThat(newTargetHits + newExploreAdditionalHits).isEqualTo(1500);
        }

        @Test
        void shouldHandleEfSearchWithZeroTargetHitsConversion() {
            // When targetHits=0 gets converted to 1, efSearch logic should still work
            String originalYql =
                    "{targetHits:100, hnsw.exploreAdditionalHits:1900}"; // efSearch = 2000

            String result = callOverwriteTargetHits(originalYql, 0, 2000);
            assertThat(result)
                    .isEqualTo(
                            "{targetHits:1, hnsw.exploreAdditionalHits:1999}"); // 0 converted to 1
        }

        @Test
        void shouldValidateEfSearchCalculationAcrossMultipleScenarios() {
            // Test comprehensive efSearch validation with various input combinations

            // Scenario 1: Small efSearch value
            String yql1 = "{targetHits:50, hnsw.exploreAdditionalHits:50}"; // efSearch = 100
            String result1 = callOverwriteTargetHits(yql1, 30, 100);
            assertThat(result1).isEqualTo("{targetHits:30, hnsw.exploreAdditionalHits:70}");

            // Scenario 2: Large efSearch value
            String yql2 = "{targetHits:500, hnsw.exploreAdditionalHits:4500}"; // efSearch = 5000
            String result2 = callOverwriteTargetHits(yql2, 1000, 5000);
            assertThat(result2).isEqualTo("{targetHits:1000, hnsw.exploreAdditionalHits:4000}");

            // Scenario 3: Edge case where newTargetHits equals efSearch
            String yql3 = "{targetHits:100, hnsw.exploreAdditionalHits:900}"; // efSearch = 1000
            String result3 = callOverwriteTargetHits(yql3, 1000, 1000);
            assertThat(result3).isEqualTo("{targetHits:1000, hnsw.exploreAdditionalHits:0}");
        }

        @Test
        void shouldSetExploreAdditionalHitsToZeroWhenNewTargetHitsExceedsEfSearch() {
            // Test that when newTargetHits > efSearch, exploreAdditionalHits is set to 0
            String originalYql =
                    "{targetHits:100, hnsw.exploreAdditionalHits:400}"; // efSearch = 500

            // Case 1: newTargetHits slightly larger than efSearch
            String result1 = callOverwriteTargetHits(originalYql, 600, 500);
            assertThat(result1).isEqualTo("{targetHits:600, hnsw.exploreAdditionalHits:0}");

            // Case 2: newTargetHits much larger than efSearch
            String result2 = callOverwriteTargetHits(originalYql, 1500, 500);
            assertThat(result2).isEqualTo("{targetHits:1500, hnsw.exploreAdditionalHits:0}");

            // Case 3: newTargetHits way larger than efSearch
            String result3 = callOverwriteTargetHits(originalYql, 10000, 500);
            assertThat(result3).isEqualTo("{targetHits:10000, hnsw.exploreAdditionalHits:0}");
        }

        private Integer callExtractCurrentTargetHits(String yql) {
            return hybridSearcher.extractCurrentTargetHits(yql);
        }

        private String callOverwriteTargetHits(String yql, int newTargetHits, int efSearch) {
            return hybridSearcher.overwriteTargetHits(yql, newTargetHits, efSearch);
        }

        private Integer callExtractCurrentExploreAdditionalHits(String yql) {
            return hybridSearcher.extractCurrentExploreAdditionalHits(yql);
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
}
