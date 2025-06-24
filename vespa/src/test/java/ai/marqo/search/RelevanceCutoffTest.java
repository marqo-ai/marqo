package ai.marqo.search;

import com.yahoo.search.Query;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
            // Use reflection to call private method
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
                            () ->
                                    callReadRelevanceCutoffParameter(
                                            query, "relative_max_score"));
            assertThat(exception.getMessage())
                    .contains(
                            "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor is"
                                    + " missing");
        }

        @Test
        void shouldReturnMeanStdDevFactorWhenMethodIsMeanStdDev() {
            Query query = new Query("search/?query=test");
            query.properties()
                    .set("marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor", 1.5);

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
                            "marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor is"
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
            try {
                java.lang.reflect.Method readMethod =
                        HybridSearcher.class.getDeclaredMethod(
                                "readRelevanceCutoffParameter", Query.class, String.class);
                readMethod.setAccessible(true);
                return (Double) readMethod.invoke(hybridSearcher, query, method);
            } catch (Exception e) {
                if (e.getCause() instanceof RuntimeException) {
                    throw (RuntimeException) e.getCause();
                }
                throw new RuntimeException(e);
            }
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
            try {
                java.lang.reflect.Method detectMethod =
                        HybridSearcher.class.getDeclaredMethod(
                                "detectCutoffCount",
                                HitGroup.class,
                                String.class,
                                Double.class,
                                boolean.class);
                detectMethod.setAccessible(true);
                return (Integer)
                        detectMethod.invoke(hybridSearcher, hits, method, parameter, verbose);
            } catch (Exception e) {
                if (e.getCause() instanceof RuntimeException) {
                    throw (RuntimeException) e.getCause();
                }
                throw new RuntimeException(e);
            }
        }
    }
}