package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.yahoo.search.Query;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for HybridSearcher.postProcessResults method.
 * Tests the new pagination logic controlled by needToTrimPreviousPages parameter.
 */
@DisplayName("HybridSearcher PostProcessResults Tests")
class HybridSearcherPostProcessResultsTest {

    private HybridSearcher hybridSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    /**
     * Helper method to create a HitGroup with specified number of hits.
     * Each hit has a relevance score of 1.0/(index+1) to ensure predictable sorting.
     */
    private HitGroup createTestHitGroup(int size) {
        HitGroup hitGroup = new HitGroup();
        for (int i = 0; i < size; i++) {
            Hit hit = new Hit("hit_" + i, 1.0 / (i + 1));
            hitGroup.add(hit);
        }
        return hitGroup;
    }

    /**
     * Helper method to create a query with no global modifiers.
     */
    private Query createTestQuery() {
        Query query = new Query();
        // Ensure no global modifiers are present
        // TODO add test coverage for global score modifier later
        query.getRanking()
                .getFeatures()
                .put("query(marqo__mult_weights_global)", Tensor.from("tensor(p{}):{}"));
        query.getRanking()
                .getFeatures()
                .put("query(marqo__add_weights_global)", Tensor.from("tensor(p{}):{}"));
        return query;
    }

    @Test
    @DisplayName("Test 1: needToTrimPreviousPages=true adjusts rerankDepthGlobal by adding offset")
    void testPostProcessResults_NeedToTrimTrue_AdjustsRerankDepth() {
        // Arrange
        HitGroup inputHits = createTestHitGroup(20); // 20 hits
        Query query = createTestQuery();

        int limit = 10;
        int offset = 5;
        Integer rerankDepthGlobal = 10; // Should become 15 (10 + 5)
        boolean needToTrimPreviousPages = true;
        boolean verbose = false;

        // Act
        HitGroup result =
                hybridSearcher.postProcessResults(
                        inputHits,
                        query,
                        rerankDepthGlobal,
                        limit,
                        offset,
                        needToTrimPreviousPages,
                        verbose);

        // Assert
        // When needToTrimPreviousPages=true, rerankDepthGlobal becomes 15 (10+5)
        // So hits 0-14 should be reranked (sorted by relevance)
        // Final result should have 10 hits starting from offset 5
        assertThat(result.size()).isEqualTo(10);

        // Verify we get hits from positions 5-14 (after reranking adjustment)
        // Since all hits have descending relevance scores, the first hit should be hit_0
        assertThat(result.get(0).getId().toString()).isEqualTo("hit_5");
    }

    @Test
    @DisplayName("Test 2: needToTrimPreviousPages=false keeps original rerankDepthGlobal")
    void testPostProcessResults_NeedToTrimFalse_KeepsOriginalRerankDepth() {
        // Arrange
        HitGroup inputHits = createTestHitGroup(20); // 20 hits
        Query query = createTestQuery();

        int limit = 10;
        int offset = 5;
        Integer rerankDepthGlobal = 10; // Should stay 10
        boolean needToTrimPreviousPages = false;
        boolean verbose = false;

        // Act
        HitGroup result =
                hybridSearcher.postProcessResults(
                        inputHits,
                        query,
                        rerankDepthGlobal,
                        limit,
                        offset,
                        needToTrimPreviousPages,
                        verbose);

        // Assert
        // When needToTrimPreviousPages=false, rerankDepthGlobal stays 10
        // So only hits 0-9 are reranked
        // Final result should have 10 hits starting from position 0 (ignoring offset)
        assertThat(result.size()).isEqualTo(10);

        // Should get the first 10 hits after reranking (starts from 0, ignores offset)
        assertThat(result.get(0).getId().toString()).isEqualTo("hit_0");
    }

    @Test
    @DisplayName(
            "Test 3: needToTrimPreviousPages=true with null rerankDepthGlobal reranks all hits")
    void testPostProcessResults_NeedToTrimTrue_WithNullRerankDepth() {
        // Arrange
        HitGroup inputHits = createTestHitGroup(15); // 15 hits
        Query query = createTestQuery();

        int limit = 10;
        int offset = 5;
        Integer rerankDepthGlobal = null; // Should rerank all hits regardless of offset
        boolean needToTrimPreviousPages = true;
        boolean verbose = false;

        // Act
        HitGroup result =
                hybridSearcher.postProcessResults(
                        inputHits,
                        query,
                        rerankDepthGlobal,
                        limit,
                        offset,
                        needToTrimPreviousPages,
                        verbose);

        // Assert
        // When rerankDepthGlobal is null, all hits are reranked regardless of
        // needToTrimPreviousPages
        // Final result should have 10 hits starting from offset 5
        assertThat(result.size()).isEqualTo(10);

        // Verify we get hits from positions 5-14 after reranking
        assertThat(result.get(0).getId().toString()).isEqualTo("hit_5");
    }

    @Test
    @DisplayName("Test 4: needToTrimPreviousPages=true uses trim(offset, limit)")
    void testPostProcessResults_NeedToTrimTrue_TrimsWithOffset() {
        // Arrange
        HitGroup inputHits = createTestHitGroup(20); // 20 hits
        Query query = createTestQuery();

        int limit = 5;
        int offset = 3;
        Integer rerankDepthGlobal = 20; // Rerank all
        boolean needToTrimPreviousPages = true;
        boolean verbose = false;

        // Act
        HitGroup result =
                hybridSearcher.postProcessResults(
                        inputHits,
                        query,
                        rerankDepthGlobal,
                        limit,
                        offset,
                        needToTrimPreviousPages,
                        verbose);

        // Assert
        // Should return 5 hits starting from offset 3 (positions 3,4,5,6,7)
        assertThat(result.size()).isEqualTo(5);

        // After reranking, hits are sorted by relevance (hit_0 has highest score)
        // So result should contain hits starting from position 3
        assertThat(result.get(0).getId().toString()).isEqualTo("hit_3");
        assertThat(result.get(1).getId().toString()).isEqualTo("hit_4");
        assertThat(result.get(4).getId().toString()).isEqualTo("hit_7");
    }

    @Test
    @DisplayName("Test 5: needToTrimPreviousPages=false uses trim(0, limit)")
    void testPostProcessResults_NeedToTrimFalse_TrimsFromZero() {
        // Arrange
        HitGroup inputHits = createTestHitGroup(20); // 20 hits
        Query query = createTestQuery();

        int limit = 5;
        int offset = 3; // Should be ignored
        Integer rerankDepthGlobal = 20; // Rerank all
        boolean needToTrimPreviousPages = false;
        boolean verbose = false;

        // Act
        HitGroup result =
                hybridSearcher.postProcessResults(
                        inputHits,
                        query,
                        rerankDepthGlobal,
                        limit,
                        offset,
                        needToTrimPreviousPages,
                        verbose);

        // Assert
        // Should return 5 hits starting from position 0 (ignoring offset)
        assertThat(result.size()).isEqualTo(5);

        // After reranking, should get the top 5 hits (positions 0,1,2,3,4)
        assertThat(result.get(0).getId().toString()).isEqualTo("hit_0");
        assertThat(result.get(1).getId().toString()).isEqualTo("hit_1");
        assertThat(result.get(4).getId().toString()).isEqualTo("hit_4");
    }
}
