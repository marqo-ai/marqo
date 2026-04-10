package ai.marqo.search;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Unit tests for HybridSearcher collapse fields functionality.
 * Tests the RRF deduplication logic when collapse fields are enabled.
 */
@DisplayName("HybridSearcher Collapse Fields Tests")
class HybridSearcherCollapseFieldsTest {

    private HybridSearcher hybridSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    // 1. Core Collapse Field Deduplication Tests
    @ParameterizedTest
    @ValueSource(doubles = {0.0, 0.25, 0.5, 0.75, 1.0})
    @DisplayName("RRF with collapse fields: alpha parameter variations")
    void testRRFWithCollapseFields_AlphaVariations(double alpha) {
        // Same rank, higher alpha wins, tensor wins when alpha is 0.5
        // Test different alpha values with collapse fields
        Hit tensorHit = createHitWithCollapseHash("tensor_doc", 1.0, 123.0);
        Hit lexicalHit = createHitWithCollapseHash("lexical_doc", 0.8, 123.0);

        HitGroup tensorHits = createHitGroup(tensorHit);
        HitGroup lexicalHits = createHitGroup(lexicalHit);

        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, alpha, false, true);

        assertEquals(1, result.size());

        // Calculate expected RRF scores
        double tensorRRF = alpha * (1.0 / (1 + 60));
        double lexicalRRF = (1.0 - alpha) * (1.0 / (1 + 60));

        if (tensorRRF >= lexicalRRF) {
            assertEquals("tensor_doc", extractDocIdFromResult(result.get(0)));
            assertEquals(tensorRRF, result.get(0).getRelevance().getScore(), 0.00001);
        } else {
            assertEquals("lexical_doc", extractDocIdFromResult(result.get(0)));
            assertEquals(lexicalRRF, result.get(0).getRelevance().getScore(), 0.00001);
        }
    }

    @Test
    @DisplayName("RRF with collapse fields: multiple collapse groups")
    void testRRFWithCollapseFields_MultipleCollapseGroups() {
        // Create hits with different collapse field values
        // Different rank, higher one wins
        HitGroup tensorHits =
                createHitGroup(
                        createHitWithCollapseHash("tensor_doc_1", 0.9, 100.0),
                        createHitWithCollapseHash("tensor_doc_2", 0.8, 200.0));

        HitGroup lexicalHits =
                createHitGroup(
                        createHitWithCollapseHash("lexical_doc_2", 0.7, 200.0),
                        createHitWithCollapseHash("lexical_doc_1", 0.6, 100.0));

        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);

        // tensor_doc_1 wins over lexical_doc_1 (same hash)
        // lexical_doc_2 wins over tensor_doc_2 (same hash)
        assertEquals(2, result.size());
        assertTrue(containsDocId(result, "tensor_doc_1"));
        assertTrue(containsDocId(result, "lexical_doc_2"));
    }

    @Test
    @DisplayName("RRF with collapse fields: no collapse groups overlap")
    void testRRFWithCollapseFields_NoOverlap() {
        // Create hits with completely different collapse field values
        HitGroup tensorHits =
                createHitGroup(
                        createHitWithCollapseHash("tensor_doc_1", 0.9, 100.0),
                        createHitWithCollapseHash("tensor_doc_2", 0.8, 200.0));

        HitGroup lexicalHits =
                createHitGroup(
                        createHitWithCollapseHash("lexical_doc_1", 0.7, 300.0),
                        createHitWithCollapseHash("lexical_doc_2", 0.6, 400.0));

        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);

        // Should have all 4 hits since no collapse field values overlap
        assertEquals(4, result.size());
        assertTrue(containsDocId(result, "tensor_doc_1"));
        assertTrue(containsDocId(result, "tensor_doc_2"));
        assertTrue(containsDocId(result, "lexical_doc_1"));
        assertTrue(containsDocId(result, "lexical_doc_2"));
    }

    @Test
    @DisplayName("RRF with collapse fields: same document ID with different collapse hash")
    void testRRFWithCollapseFields_SameDocumentId() {
        // Test when same document appears in both tensor and lexical with collapse enabled
        Hit tensorHit = createHitWithCollapseHash("same_doc", 0.8, 123.0);
        Hit lexicalHit = createHitWithCollapseHash("same_doc", 0.6, 123.0);

        HitGroup tensorHits = createHitGroup(tensorHit);
        HitGroup lexicalHits = createHitGroup(lexicalHit);

        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);

        // Should merge scores since it's the same document (handled by existing logic)
        assertEquals(1, result.size());
        Hit mergedHit = result.get(0);
        assertEquals("same_doc", extractDocIdFromResult(mergedHit));
        // Should have combined RRF score from both tensor and lexical
        assertNotNull(mergedHit.getField("marqo__raw_tensor_score"));
        assertNotNull(mergedHit.getField("marqo__raw_lexical_score"));
    }

    // 2. Edge Cases and Error Handling Tests

    @Test
    @DisplayName("RRF with collapse fields: missing collapse field hash")
    void testRRFWithCollapseFields_MissingCollapseFieldHash() {
        // Test behavior when collapse_field_hash is null or missing
        Hit hitWithoutHash = createHitWithoutCollapseHash("doc_without_hash", 0.8);
        Hit normalHit = createHitWithCollapseHash("normal_doc", 0.7, 123.0);

        HitGroup tensorHits = createHitGroup(hitWithoutHash);
        HitGroup lexicalHits = createHitGroup(normalHit);

        // Should handle gracefully without throwing exceptions
        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);

        // Both hits should be present since they can't be compared for collapse
        assertEquals(2, result.size());
        assertTrue(containsDocId(result, "doc_without_hash"));
        assertTrue(containsDocId(result, "normal_doc"));
    }

    @Test
    @DisplayName("RRF with collapse fields: null match features")
    void testRRFWithCollapseFields_NullMatchFeatures() {
        Hit hitWithNullFeatures = createHitWithNullMatchFeatures("doc_without_features", 0.8);
        Hit normalHit = createHitWithCollapseHash("normal_doc", 0.7, 123.0);

        HitGroup tensorHits = createHitGroup(hitWithNullFeatures);
        HitGroup lexicalHits = createHitGroup(normalHit);

        // Should handle null match features gracefully
        assertDoesNotThrow(
                () -> {
                    HitGroup result =
                            hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);
                    assertEquals(2, result.size());
                });
    }

    @Test
    @DisplayName("RRF with collapse fields: empty result sets")
    void testRRFWithCollapseFields_EmptyResultSets() {
        HitGroup emptyTensorHits = new HitGroup();
        HitGroup emptyLexicalHits = new HitGroup();

        HitGroup result =
                hybridSearcher.rrf(emptyTensorHits, emptyLexicalHits, 60, 0.5, false, true);

        assertEquals(0, result.size());
    }

    @Test
    @DisplayName("RRF with collapse fields: one empty result set")
    void testRRFWithCollapseFields_OneEmptyResultSet() {
        Hit tensorHit = createHitWithCollapseHash("tensor_doc", 0.8, 123.0);
        HitGroup tensorHits = createHitGroup(tensorHit);
        HitGroup emptyLexicalHits = new HitGroup();

        HitGroup result = hybridSearcher.rrf(tensorHits, emptyLexicalHits, 60, 0.5, false, true);

        assertEquals(1, result.size());
        assertEquals("tensor_doc", extractDocIdFromResult(result.get(0)));
    }

    // 4. Backwards Compatibility Tests

    @Test
    @DisplayName("RRF without collapse fields: backwards compatible behavior")
    void testRRFWithoutCollapseFields_BackwardsCompatible() {
        // Test that setting collapse=false preserves original behavior
        HitGroup tensorHits =
                createHitGroup(
                        createHitWithCollapseHash("tensor_doc_1", 0.9, 123.0),
                        createHitWithCollapseHash("tensor_doc_2", 0.8, 456.0));

        HitGroup lexicalHits =
                createHitGroup(
                        createHitWithCollapseHash(
                                "lexical_doc_1", 0.7, 123.0), // Same collapse hash as tensor_doc_1
                        createHitWithCollapseHash("lexical_doc_2", 0.6, 789.0));

        HitGroup resultWithoutCollapse =
                hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, false);
        HitGroup resultWithCollapse =
                hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, true);

        // Without collapse: should have all 4 hits
        assertEquals(4, resultWithoutCollapse.size());

        // With collapse: should have 3 hits (tensor_doc_1 wins over lexical_doc_1)
        assertEquals(3, resultWithCollapse.size());

        // Verify that without collapse, original behavior is preserved
        assertTrue(containsDocId(resultWithoutCollapse, "tensor_doc_1"));
        assertTrue(containsDocId(resultWithoutCollapse, "tensor_doc_2"));
        assertTrue(containsDocId(resultWithoutCollapse, "lexical_doc_1"));
        assertTrue(containsDocId(resultWithoutCollapse, "lexical_doc_2"));
    }

    @Test
    @DisplayName("RRF without collapse fields: same document merging still works")
    void testRRFWithoutCollapseFields_SameDocumentMerging() {
        // Test that same document ID merging still works when collapse=false
        Hit tensorHit = createHitWithCollapseHash("same_doc", 0.8, 123.0);
        Hit lexicalHit = createHitWithCollapseHash("same_doc", 0.6, 123.0);

        HitGroup tensorHits = createHitGroup(tensorHit);
        HitGroup lexicalHits = createHitGroup(lexicalHit);

        HitGroup result = hybridSearcher.rrf(tensorHits, lexicalHits, 60, 0.5, false, false);

        // Should merge the same document
        assertEquals(1, result.size());
        Hit mergedHit = result.get(0);
        assertEquals("same_doc", extractDocIdFromResult(mergedHit));
        assertNotNull(mergedHit.getField("marqo__raw_tensor_score"));
        assertNotNull(mergedHit.getField("marqo__raw_lexical_score"));
    }

    // 4. Test extract collapse_field_hash match features
    @Test
    @DisplayName("Extract collapse field hash: valid match features")
    void testExtractCollapseFieldHash_ValidMatchFeatures() {
        Hit hit = createHitWithCollapseHash("test_doc", 0.8, 123.45);

        Double collapseHash = hybridSearcher.extractCollapseFieldHash(hit);

        assertNotNull(collapseHash);
        assertEquals(123.45, collapseHash, 0.0001);
    }

    @Test
    @DisplayName("Extract collapse field hash: null match features")
    void testExtractCollapseFieldHash_NullMatchFeatures() {
        Hit hit = createHitWithNullMatchFeatures("test_doc", 0.8);

        Double result = hybridSearcher.extractCollapseFieldHash(hit);
        assertNull(result);
    }

    @Test
    @DisplayName("Extract collapse field hash: missing collapse field")
    void testExtractCollapseFieldHash_MissingCollapseField() {
        Hit hit = createHitWithoutCollapseHash("test_doc", 0.8);

        Double collapseHash = hybridSearcher.extractCollapseFieldHash(hit);

        // Should return null when collapse_field_hash is not present
        assertNull(collapseHash);
    }

    // Helper Methods

    /**
     * Creates a Hit with collapse field hash in match features.
     */
    private Hit createHitWithCollapseHash(String docId, double score, double collapseHash) {
        Hit hit = new Hit("index:test/0/" + docId);
        hit.setRelevance(score);
        FeatureData matchFeatures = mock(FeatureData.class);
        when(matchFeatures.getDouble("collapse_field_hash")).thenReturn(collapseHash);
        hit.setField("matchfeatures", matchFeatures);
        return hit;
    }

    /**
     * Creates a Hit without collapse field hash in match features.
     */
    private Hit createHitWithoutCollapseHash(String docId, double score) {
        Hit hit = new Hit("index:test/0/" + docId);
        hit.setRelevance(score);
        FeatureData matchFeatures = mock(FeatureData.class);
        when(matchFeatures.getDouble("collapse_field_hash")).thenReturn(null);
        hit.setField("matchfeatures", matchFeatures);
        return hit;
    }

    /**
     * Creates a Hit with null match features.
     */
    private Hit createHitWithNullMatchFeatures(String docId, double score) {
        Hit hit = new Hit("index:test/0/" + docId);
        hit.setRelevance(score);
        return hit;
    }

    /**
     * Creates a HitGroup from variable number of hits.
     */
    private HitGroup createHitGroup(Hit... hits) {
        HitGroup hitGroup = new HitGroup();
        for (Hit hit : hits) {
            hitGroup.add(hit);
        }
        return hitGroup;
    }

    /**
     * Checks if a HitGroup contains a document with the given ID.
     */
    private boolean containsDocId(HitGroup hitGroup, String docId) {
        return hitGroup.asList().stream()
                .anyMatch(
                        hit ->
                                docId.equals(
                                        HybridSearcher.extractDocIdFromHitId(
                                                hit.getId().toString())));
    }

    /**
     * Extracts document ID from the first hit in result.
     */
    private String extractDocIdFromResult(Hit hit) {
        return HybridSearcher.extractDocIdFromHitId(hit.getId().toString());
    }
}
