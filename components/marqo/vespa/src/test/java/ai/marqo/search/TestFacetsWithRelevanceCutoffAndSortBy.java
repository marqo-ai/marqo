package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
        void shouldWrapSimpleGroupingWithMaxAndAll() {
            // all(group(...) each(...)) -> all(max(5) all(group(...) each(...)))
            String input =
                    "select * from schema where (query) limit 0 | all(group(color)"
                            + " each(output(count())))";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result).endsWith("| all(max(5) all(group(color) each(output(count()))))");
        }

        @Test
        void shouldReturnUnchangedWhenNoPipeAll() {
            String input = "select * from schema where (query)";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result).isEqualTo(input);
        }

        @Test
        void shouldReturnUnchangedWhenGroupingDoesNotStartWithAll() {
            String input = "select * from schema where (query) limit 0 | each(output(count()))";
            String result = hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 5, false);
            assertThat(result).isEqualTo(input);
        }

        @Test
        void shouldHandleNullAndEmptyInput() {
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(null, 5, false))
                    .isEqualTo("");
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping("", 5, false)).isEqualTo("");
        }

        @Test
        void shouldThrowWhenMaxHitsIsZeroOrNegative() {
            String input =
                    "select * from schema where (query) limit 0 | all(group(color)"
                            + " each(output(count())))";
            assertThatThrownBy(
                            () -> hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, 0, false))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("maxHits must be >= 1");
            assertThatThrownBy(
                            () -> hybridSearcher.injectMaxHitsIntoFacetsGrouping(input, -3, false))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("maxHits must be >= 1");
        }

        @Test
        void shouldHandleVariousWhitespaceInGroupingPart() {
            // Multiple spaces after all(
            String multiSpace =
                    "select * from schema where (query) limit 0 | all(   group(color)"
                            + " each(output(count())))";
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(multiSpace, 7, false))
                    .endsWith("| all(max(7) all(group(color) each(output(count()))))");

            // Trailing whitespace after closing paren
            String trailing =
                    "select * from schema where (query) limit 0 | all(group(color)"
                            + " each(output(count())))  ";
            assertThat(hybridSearcher.injectMaxHitsIntoFacetsGrouping(trailing, 4, false))
                    .contains("all(max(4) all(group(color) each(output(count()))))");
        }

        @Test
        void shouldHandleRealWorldFacetsYqlWithMultipleFieldsFromVespaQuery() {
            // Real-world facets YQL from an actual Vespa query with color and brand facet fields
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
            // Our injected max wraps the existing structure
            assertThat(result).contains("all(max(5) all(max(10)");
            // Existing user maxDepth=10 preserved
            assertThat(result).contains("max(10)");
            // Both facet fields preserved
            assertThat(result).contains("marqo__short_string_fields{\"color\"}");
            assertThat(result).contains("marqo__short_string_fields{\"brand\"}");
            // Per-field max(100) preserved (two occurrences)
            int max100Count = 0;
            int searchIdx = 0;
            while ((searchIdx = result.indexOf("max(100)", searchIdx)) != -1) {
                max100Count++;
                searchIdx++;
            }
            assertThat(max100Count).isEqualTo(2);
        }

        @Test
        void shouldHandleRealWorldTotalHitsYqlFromVespaQuery() {
            // Real-world totalHits YQL from an actual Vespa query
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
            assertThat(result).contains("all(max(5) all(group(1.1) each(output(count()))))");
        }
    }

    @Nested
    class UpdateQueryWithAffectFacetsTest {

        @Test
        void shouldAdjustFacetsYqlTargetHitsAndInjectMax() {
            // Simulate a real disjunction query with facets, relevanceCutoff affectFacets=true,
            // relevantCandidates=5, no sort. The facets YQL contains two delimiter-separated
            // queries: totalHits and a string facet with maxDepth=10.
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
            String combinedFacetsYql = totalHitsYql + delimiter + facetsYql;

            // Set up query with hits=2, offset=0 and tensor YQL with targetHits:2
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

            // relevantCandidates=5, no sort, affectFacets=true
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 5, null, true, false, true, false);

            // newHits = min(relevantCandidates=5, limit+offset=2) = 2
            assertThat(result.getHits()).isEqualTo(2);

            // Facets YQL should use newHits=2 for max and newTensorTargetHits=2 for targetHits
            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);
            assertThat(updatedQueries).hasSize(2);

            // totalHits query: max(2) wraps the grouping via all(max(2) all(...))
            assertThat(updatedQueries[0]).contains("all(max(2) all(");
            assertThat(updatedQueries[0]).contains("group(1.1)");

            // facets query: max(2) wraps existing max(10), per-field max(100) untouched
            assertThat(updatedQueries[1]).contains("all(max(2) all(");
            assertThat(updatedQueries[1]).contains("max(10)");
            assertThat(updatedQueries[1]).contains("max(100)");
            assertThat(updatedQueries[1]).contains("group(marqo__short_string_fields");
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

            // affectFacets=false
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 5, null, true, false, false, false);

            // Facets YQL should be unchanged
            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            assertThat(updatedFacetsYql).isEqualTo(combinedFacetsYql);
        }

        @Test
        void shouldHandleRealVespaQueryWithSortByAndAffectFacetsAndTwoFacetFields() {
            // Exact replica of the real Vespa query structure:
            // - hits=10, offset=0, targetHits:10, exploreAdditionalHits:1990
            // - sortBy enabled, affectFacets=true, relevantCandidates=5
            // - Two facet fields (color, brand) with maxDepth=10, per-field max(100)
            // - Plus totalHits grouping
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
            String combinedFacetsYql = totalHitsYql + delimiter + facetsYql;

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
            query.properties().set("marqo__yql.facets", combinedFacetsYql);

            // Both sortBy and relevanceCutoff enabled, affectFacets=true
            // relevantCandidates=5, sortByMinSortCandidates=10
            int relevantCandidates = 5;
            int sortByMinSortCandidates = 10;
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query,
                            relevantCandidates,
                            sortByMinSortCandidates,
                            true,
                            true,
                            true,
                            false);

            // newHits = max(relevantCandidates=5, sortByMinSortCandidates=10) = 10
            assertThat(result.getHits()).isEqualTo(10);

            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);
            assertThat(updatedQueries).hasSize(2);

            // totalHits query: max(10) injected, grouping preserved
            assertThat(updatedQueries[0]).contains("all(max(10) all(");
            assertThat(updatedQueries[0]).contains("group(1.1)");

            // facets query: max(10) injected wrapping existing max(10) maxDepth
            assertThat(updatedQueries[1]).contains("all(max(10) all(max(10)");
            // Both facet fields preserved
            assertThat(updatedQueries[1]).contains("marqo__short_string_fields{\"color\"}");
            assertThat(updatedQueries[1]).contains("marqo__short_string_fields{\"brand\"}");
            // Per-field max(100) preserved (two occurrences)
            int max100Count = 0;
            int searchIdx = 0;
            while ((searchIdx = updatedQueries[1].indexOf("max(100)", searchIdx)) != -1) {
                max100Count++;
                searchIdx++;
            }
            assertThat(max100Count).isEqualTo(2);
        }

        @Test
        void shouldUpdateTargetHitsInFacetsWhenSortByIncreasesHits() {
            // When sortBy + relevanceCutoff together increase newHits beyond original targetHits,
            // the facets YQL targetHits should also be updated.
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
            String combinedFacetsYql = totalHitsYql + delimiter + facetsYql;

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
            query.properties().set("marqo__yql.facets", combinedFacetsYql);

            // relevantCandidates=20, sortByMinSortCandidates=15 -> newHits=20
            // This exceeds original targetHits:10, so targetHits in facets YQL must be updated too
            Query result =
                    hybridSearcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 20, 15, true, true, true, false);

            assertThat(result.getHits()).isEqualTo(20);

            String updatedFacetsYql = result.properties().getString("marqo__yql.facets");
            String[] updatedQueries = updatedFacetsYql.split(delimiter);

            // targetHits in facets YQL should be updated from 10 to 20
            assertThat(updatedQueries[0]).contains("targetHits:20");
            assertThat(updatedQueries[1]).contains("targetHits:20");

            // max(20) should be injected
            assertThat(updatedQueries[0]).contains("max(20)");
            assertThat(updatedQueries[1]).contains("max(20)");
        }
    }
}
