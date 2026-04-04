package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.yahoo.search.Query;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class TestFacetsWithRelevanceCutoffAndSortBy {
    private HybridSearcher hybridSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    @Nested
    class InjectMaxHitsIntoFacetsGroupingTest {

        @Test
        void shouldInsertMaxWhenNotPresent() {
            // No max() in grouping — wrap inner content with max(N) all(...)
            String input =
                    "select * from schema where (query) limit 0 | all(group(color)"
                            + " each(output(count())))";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result)
                    .isEqualTo(
                            "select * from schema where (query) limit 0"
                                    + " | all(max(5) all(group(color) each(output(count()))))");
        }

        @Test
        void shouldReplaceMaxWhenNewValueIsSmaller() {
            String input =
                    "select * from schema where (query) limit 0 | all( max(100) all(group(color)"
                            + " each(output(count()))))";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 3, false);
            assertThat(result)
                    .isEqualTo(
                            "select * from schema where (query) limit 0 | all( max(3)"
                                    + " all(group(color) each(output(count()))))");
        }

        @Test
        void shouldSkipWhenExistingMaxIsSmallerOrEqual() {
            String input =
                    "select * from schema where (query) limit 0 | all( max(5) all(group(color)"
                            + " each(output(count()))))";
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 10, false))
                    .isEqualTo(input);
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false))
                    .isEqualTo(input);
        }

        @Test
        void shouldReturnUnchangedWhenNoPipeAll() {
            String input = "select * from schema where (query)";
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false))
                    .isEqualTo(input);
        }

        @Test
        void shouldReturnUnchangedWhenGroupingDoesNotStartWithAll() {
            String input = "select * from schema where (query) limit 0 | each(output(count()))";
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false))
                    .isEqualTo(input);
        }

        @Test
        void shouldHandleNullAndEmptyInput() {
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(null, 5, false))
                    .isEqualTo("");
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping("", 5, false)).isEqualTo("");
        }

        @Test
        void shouldHandleRealWorldFacetsYqlWithMultipleFieldsFromVespaQuery() {
            // Real-world facets YQL: max(10) present with N=5 < 10, should replace
            String input =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "default contains \"intelligence\" OR default contains \"world\" OR "
                            + "default contains \"vocabulary\" OR default contains \"millions\" OR "
                            + "default contains \"day\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all( max(10) all(group(marqo__short_string_fields"
                            + "{\"color\"}) max(100) order(-count()) each(output(count()))) "
                            + "all(group(marqo__short_string_fields{\"brand\"}) max(100) "
                            + "order(-count()) each(output(count()))) )";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result)
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"universe\" OR default contains \"ocean\" OR default contains"
                                + " \"intelligence\" OR default contains \"world\" OR default"
                                + " contains \"vocabulary\" OR default contains \"millions\" OR"
                                + " default contains \"day\" OR ({targetHits:10, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all( max(5)"
                                + " all(group(marqo__short_string_fields{\"color\"}) max(100)"
                                + " order(-count()) each(output(count())))"
                                + " all(group(marqo__short_string_fields{\"brand\"}) max(100)"
                                + " order(-count()) each(output(count()))) )");
        }

        @Test
        void shouldHandleRealWorldTotalHitsYqlFromVespaQuery() {
            // totalHits YQL: no max() present, should wrap with max(5) all(...)
            String input =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "default contains \"intelligence\" OR default contains \"world\" OR "
                            + "default contains \"vocabulary\" OR default contains \"millions\" OR "
                            + "default contains \"day\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all(group(1.1) each(output(count())))";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result)
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"universe\" OR default contains \"ocean\" OR default contains"
                                + " \"intelligence\" OR default contains \"world\" OR default"
                                + " contains \"vocabulary\" OR default contains \"millions\" OR"
                                + " default contains \"day\" OR ({targetHits:10, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all(max(5) all(group(1.1)"
                                + " each(output(count()))))");
        }
    }

    @Nested
    class UpdateQueryWithAffectFacetsTest {

        @Test
        void shouldAdjustFacetsYqlTargetHitsAndInjectMax() {
            String delimiter = "\n---MARQO-YQL-QUERY-DELIMITER---\n";
            String totalHitsYql =
                    "select * from marqo__index where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "({targetHits:2, approximate:True, hnsw.exploreAdditionalHits:1998}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all(group(1.1) each(output(count())))";
            String facetsYql =
                    "select * from marqo__index where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "({targetHits:2, approximate:True, hnsw.exploreAdditionalHits:1998}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all( max(10) all(group(marqo__short_string_fields"
                            + "{\"color\"}) max(100) order(-count()) each(output(count()))) )";

            Query query = new Query();
            query.setHits(2);
            query.setOffset(0);
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from marqo__index where "
                                    + "({targetHits:2, approximate:True,"
                                    + " hnsw.exploreAdditionalHits:1998}"
                                    + "nearestNeighbor(marqo__embeddings_text,"
                                    + " marqo__query_embedding))");
            query.properties().set("marqo__yql.facets", totalHitsYql + delimiter + facetsYql);

            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 5, null, true, false, true, false, false);

            assertThat(result.getHits()).isEqualTo(2);

            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);
            assertThat(updatedQueries).hasSize(2);

            // Verify entire results
            assertThat(updatedQueries[0])
                    .isEqualTo(
                            "select * from marqo__index where (default contains \"universe\" OR"
                                + " default contains \"ocean\" OR ({targetHits:2, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1998}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all(max(2) all(group(1.1)"
                                + " each(output(count()))))");
            assertThat(updatedQueries[1])
                    .isEqualTo(
                            "select * from marqo__index where (default contains \"universe\" OR"
                                + " default contains \"ocean\" OR ({targetHits:2, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1998}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all( max(2)"
                                + " all(group(marqo__short_string_fields{\"color\"}) max(100)"
                                + " order(-count()) each(output(count()))) )");
        }

        @Test
        void shouldNotModifyFacetsYqlWhenAffectFacetsIsFalse() {
            String delimiter = "\n---MARQO-YQL-QUERY-DELIMITER---\n";
            String totalHitsYql =
                    "select * from marqo__index where (query OR "
                            + "({targetHits:2, approximate:True, hnsw.exploreAdditionalHits:1998}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all(group(1.1) each(output(count())))";
            String facetsYql =
                    "select * from marqo__index where (query OR "
                            + "({targetHits:2, approximate:True, hnsw.exploreAdditionalHits:1998}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all( max(10) all(group(color) max(100)"
                            + " each(output(count()))) )";
            String combinedFacetsYql = totalHitsYql + delimiter + facetsYql;

            Query query = new Query();
            query.setHits(2);
            query.setOffset(0);
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from marqo__index where "
                                    + "({targetHits:2, approximate:True,"
                                    + " hnsw.exploreAdditionalHits:1998}"
                                    + "nearestNeighbor(marqo__embeddings_text,"
                                    + " marqo__query_embedding))");
            query.properties().set("marqo__yql.facets", combinedFacetsYql);

            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 5, null, true, false, false, false, false);

            assertThat(result.properties().getString("marqo__yql.facets"))
                    .isEqualTo(combinedFacetsYql);
        }

        @Test
        void shouldHandleRealVespaQueryWithSortByAndAffectFacetsAndTwoFacetFields() {
            String delimiter = "\n---MARQO-YQL-QUERY-DELIMITER---\n";
            String totalHitsYql =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "default contains \"intelligence\" OR default contains \"world\" OR "
                            + "default contains \"vocabulary\" OR default contains \"millions\" OR "
                            + "default contains \"day\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all(group(1.1) each(output(count())))";
            String facetsYql =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"universe\" OR default contains \"ocean\" OR "
                            + "default contains \"intelligence\" OR default contains \"world\" OR "
                            + "default contains \"vocabulary\" OR default contains \"millions\" OR "
                            + "default contains \"day\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all( max(10) "
                            + "all(group(marqo__short_string_fields{\"color\"}) max(100) "
                            + "order(-count()) each(output(count()))) "
                            + "all(group(marqo__short_string_fields{\"brand\"}) max(100) "
                            + "order(-count()) each(output(count()))) )";

            Query query = new Query();
            query.setHits(10);
            query.setOffset(0);
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from marqo__facets_01rc_01match where "
                                    + "({targetHits:10, approximate:True,"
                                    + " hnsw.exploreAdditionalHits:1990}"
                                    + "nearestNeighbor(marqo__embeddings_text,"
                                    + " marqo__query_embedding))");
            query.properties().set("marqo__yql.facets", totalHitsYql + delimiter + facetsYql);

            // newHits = max(relevantCandidates=5, sortByMinSortCandidates=10) = 10
            // targetHits unchanged (already 10), facets max(10) unchanged (10 >= 10)
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 5, 10, true, true, true, false, false);

            assertThat(result.getHits()).isEqualTo(10);

            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);
            assertThat(updatedQueries).hasSize(2);

            // Verify entire results
            assertThat(updatedQueries[0])
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"universe\" OR default contains \"ocean\" OR default contains"
                                + " \"intelligence\" OR default contains \"world\" OR default"
                                + " contains \"vocabulary\" OR default contains \"millions\" OR"
                                + " default contains \"day\" OR ({targetHits:10, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all(max(10) all(group(1.1)"
                                + " each(output(count()))))");
            assertThat(updatedQueries[1])
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"universe\" OR default contains \"ocean\" OR default contains"
                                + " \"intelligence\" OR default contains \"world\" OR default"
                                + " contains \"vocabulary\" OR default contains \"millions\" OR"
                                + " default contains \"day\" OR ({targetHits:10, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all( max(10)"
                                + " all(group(marqo__short_string_fields{\"color\"}) max(100)"
                                + " order(-count()) each(output(count())))"
                                + " all(group(marqo__short_string_fields{\"brand\"}) max(100)"
                                + " order(-count()) each(output(count()))) )");
        }

        @Test
        void shouldUpdateTargetHitsInFacetsWhenSortByIncreasesHits() {
            String delimiter = "\n---MARQO-YQL-QUERY-DELIMITER---\n";
            String totalHitsYql =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"test\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all(group(1.1) each(output(count())))";
            String facetsYql =
                    "select * from marqo__facets_01rc_01match where ("
                            + "default contains \"test\" OR "
                            + "({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}"
                            + "nearestNeighbor(marqo__embeddings_text, marqo__query_embedding))"
                            + ") limit 0 | all( max(10) "
                            + "all(group(marqo__short_string_fields{\"color\"}) max(100) "
                            + "order(-count()) each(output(count()))) )";

            Query query = new Query();
            query.setHits(10);
            query.setOffset(0);
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from marqo__facets_01rc_01match where "
                                    + "({targetHits:10, approximate:True,"
                                    + " hnsw.exploreAdditionalHits:1990}"
                                    + "nearestNeighbor(marqo__embeddings_text,"
                                    + " marqo__query_embedding))");
            query.properties().set("marqo__yql.facets", totalHitsYql + delimiter + facetsYql);

            // newHits=20 exceeds original targetHits:10
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 20, 15, true, true, true, false, false);

            assertThat(result.getHits()).isEqualTo(20);

            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);

            // Verify entire results — targetHits updated to 20, max(20) inserted in totalHits,
            // facets max(10) unchanged since 20 >= 10
            assertThat(updatedQueries[0])
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"test\" OR ({targetHits:20, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1980}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all(max(20) all(group(1.1)"
                                + " each(output(count()))))");
            assertThat(updatedQueries[1])
                    .isEqualTo(
                            "select * from marqo__facets_01rc_01match where (default contains"
                                + " \"test\" OR ({targetHits:20, approximate:True,"
                                + " hnsw.exploreAdditionalHits:1980}nearestNeighbor(marqo__embeddings_text,"
                                + " marqo__query_embedding))) limit 0 | all( max(10)"
                                + " all(group(marqo__short_string_fields{\"color\"}) max(100)"
                                + " order(-count()) each(output(count()))) )");
        }
    }
}
