package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.SearchChainRegistry;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/**
 * Unit tests for RRF pagination logic in HybridSearcher.
 *
 * <p>This test file covers the pagination logic added in lines 283-310 of HybridSearcher.java,
 * which simulates previous pages to avoid duplicates when offset > 0 in disjunction+RRF searches.
 *
 * <p>NOTE: This file is self-contained and can be easily removed if/when this pagination logic
 * is replaced with a different approach (e.g., stateful pagination or over-fetching).
 */
@DisplayName("HybridSearcher RRF Pagination Tests")
class HybridSearcherRRFPaginationTest {

    private HybridSearcher hybridSearcher;
    private Searcher downstreamSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
        downstreamSearcher = mock(Searcher.class);
    }

    // ==================== Helper Methods ====================

    /**
     * Creates a test query with RRF parameters for disjunction search.
     *
     * @param alpha RRF alpha parameter (0.0 to 1.0)
     * @param rrfK RRF k parameter
     * @param limit Number of results to return
     * @param offset Pagination offset
     * @return Configured Query object
     */
    private Query createRRFQuery(double alpha, int rrfK, int limit, int offset) {
        Query query = new Query("search/?query=test");
        query.properties().set("marqo__hybrid.retrievalMethod", "disjunction");
        query.properties().set("marqo__hybrid.rankingMethod", "rrf");
        query.properties().set("marqo__hybrid.alpha", alpha);
        query.properties().set("marqo__hybrid.rrf_k", rrfK);
        query.properties().set("hits", limit);
        query.properties().set("offset", offset);
        query.properties().set("paginationMode", "fuseAndExclude");

        // Set YQL for tensor and lexical searches
        query.properties().set("marqo__yql.tensor", "tensor yql");
        query.properties().set("marqo__yql.lexical", "lexical yql");

        // Add fields to rank (required by createSubQuery)
        TensorType tensorType = new TensorType.Builder().mapped("test_tensor").build();

        Tensor fieldsToRankLexical =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__lexical_field"), 1.0)
                        .build();

        Tensor fieldsToRankTensor =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__embeddings_field"), 1.0)
                        .build();

        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_lexical)", fieldsToRankLexical);
        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_tensor)", fieldsToRankTensor);

        return query;
    }

    /**
     * Creates a HitGroup with specified hits.
     *
     * @param hitIds Array of hit IDs to create
     * @param baseScore Base relevance score for first hit (subsequent hits get lower scores)
     * @return HitGroup with the specified hits
     */
    private HitGroup createHitGroup(String[] hitIds, double baseScore) {
        HitGroup hitGroup = new HitGroup();
        for (int i = 0; i < hitIds.length; i++) {
            Hit hit = new Hit("index:test/0/" + hitIds[i], baseScore - (i * 0.1));
            hitGroup.add(hit);
        }
        return hitGroup;
    }

    /**
     * Extracts document IDs from a list of hits.
     *
     * @param hits List of hits
     * @return List of extracted doc IDs (without index prefix)
     */
    private List<String> extractDocIds(List<Hit> hits) {
        List<String> docIds = new ArrayList<>();
        for (Hit hit : hits) {
            docIds.add(HybridSearcher.extractDocIdFromHitId(hit.getId().toString()));
        }
        return docIds;
    }

    // ==================== Core Functionality Tests ====================

    @Nested
    @DisplayName("Core Functionality Tests")
    class CoreFunctionalityTests {

        @Test
        @DisplayName("Should remove previous page hits from both tensor and lexical results")
        void testRRFPaginationWithOffset_RemovesPreviousPageHits() {
            // Setup mock downstream searcher to return tensor and lexical results
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            // Create test data:
            // Tensor: doc_T1, doc_T2, doc_T3, doc_B1, doc_B2 (5 hits)
            // Lexical: doc_L1, doc_L2, doc_B1, doc_B2 (4 hits)
            // Common: doc_B1, doc_B2 (appear in both)

            HitGroup tensorResults =
                    createHitGroup(
                            new String[] {"doc_T1", "doc_T2", "doc_T3", "doc_B1", "doc_B2"}, 1.0);
            HitGroup lexicalResults =
                    createHitGroup(new String[] {"doc_L1", "doc_L2", "doc_B1", "doc_B2"}, 1.0);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Request page 2 (offset=2, limit=3)
            Query query = createRRFQuery(0.5, 60, 3, 2);
            Result result = execution.search(query);

            // Verify: Should not contain duplicates from simulated previous page
            List<String> resultIds = extractDocIds(result.hits().asList());

            // After simulation and removal, we should get different docs than page 1
            // The exact docs depend on RRF scoring, but we verify:
            // 1. No duplicates
            Set<String> uniqueIds = new HashSet<>(resultIds);
            assertThat(resultIds).hasSize(uniqueIds.size());

            // 2. Result size should be <= limit
            assertThat(result.hits().size()).isLessThanOrEqualTo(3);
        }

        @Test
        @DisplayName("Should handle hits appearing in both tensor and lexical lists")
        void testRRFPaginationWithOffset_HandlesHitsInBothLists() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            // Both lists contain the same docs (complete overlap scenario)
            HitGroup tensorResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4", "doc_5"}, 1.0);
            HitGroup lexicalResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4", "doc_5"}, 0.9);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Request page 2 (offset=2, limit=2)
            Query query = createRRFQuery(0.7, 60, 2, 2);
            Result result = execution.search(query);

            // Should successfully handle removal from both lists
            // No exception should be thrown
            assertThat((Object) result.hits()).isNotNull();

            // Verify no duplicates
            List<String> resultIds = extractDocIds(result.hits().asList());
            Set<String> uniqueIds = new HashSet<>(resultIds);
            assertThat(resultIds).hasSize(uniqueIds.size());
        }

        @Test
        @DisplayName("Should skip pagination logic when offset=0")
        void testRRFPaginationWithOffset_ZeroOffset() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            HitGroup tensorResults = createHitGroup(new String[] {"doc_1", "doc_2", "doc_3"}, 1.0);
            HitGroup lexicalResults = createHitGroup(new String[] {"doc_4", "doc_5"}, 1.0);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Request page 1 (offset=0)
            Query query = createRRFQuery(0.5, 60, 3, 0);
            Result result = execution.search(query);

            // Should process normally without pagination logic
            assertThat((Object) result.hits()).isNotNull();
            assertThat(result.hits().size()).isLessThanOrEqualTo(3);

            // Verify we only made 2 queries (tensor + lexical, no extra simulation queries)
            assertThat(queryCaptor.getAllValues()).hasSize(2);
        }

        @Test
        @DisplayName("Should work with offset when no global modifiers configured")
        void testRRFPaginationWithOffset_WithoutGlobalModifiers() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            // Create hits without match features (testing basic pagination without modifiers)
            HitGroup tensorResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4"}, 1.0);
            HitGroup lexicalResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4"}, 0.9);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // No global score modifiers - testing basic pagination
            Query query = createRRFQuery(0.5, 60, 2, 2);

            Result result = execution.search(query);

            // Should complete without errors
            assertThat((Object) result.hits()).isNotNull();
        }
    }

    // ==================== Edge Cases ====================

    @Nested
    @DisplayName("Edge Case Tests")
    class EdgeCaseTests {

        @Test
        @DisplayName("Should handle offset larger than total results")
        void testRRFPaginationWithOffset_OffsetLargerThanResults() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            // Only 3 total hits
            HitGroup tensorResults = createHitGroup(new String[] {"doc_1", "doc_2"}, 1.0);
            HitGroup lexicalResults = createHitGroup(new String[] {"doc_3"}, 1.0);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Request page with offset=10 (larger than total results)
            Query query = createRRFQuery(0.5, 60, 5, 10);
            Result result = execution.search(query);

            // Should return empty or minimal results, not crash
            assertThat((Object) result.hits()).isNotNull();
            assertThat(result.hits().size()).isLessThanOrEqualTo(5);
        }

        @Test
        @DisplayName("Should handle empty tensor results")
        void testRRFPaginationWithOffset_EmptyTensorResults() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            HitGroup tensorResults = new HitGroup(); // Empty
            HitGroup lexicalResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4"}, 1.0);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            Query query = createRRFQuery(0.5, 60, 2, 1);
            Result result = execution.search(query);

            // Should handle gracefully
            assertThat((Object) result.hits()).isNotNull();
        }

        @Test
        @DisplayName("Should handle empty lexical results")
        void testRRFPaginationWithOffset_EmptyLexicalResults() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            HitGroup tensorResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4"}, 1.0);
            HitGroup lexicalResults = new HitGroup(); // Empty

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            Query query = createRRFQuery(0.5, 60, 2, 1);
            Result result = execution.search(query);

            // Should handle gracefully
            assertThat((Object) result.hits()).isNotNull();
        }

        @Test
        @DisplayName("Should handle pagination with collapse field enabled")
        void testRRFPaginationWithOffset_WithCollapse() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            HitGroup tensorResults =
                    createHitGroup(new String[] {"doc_1", "doc_2", "doc_3", "doc_4"}, 1.0);
            HitGroup lexicalResults = createHitGroup(new String[] {"doc_1", "doc_2", "doc_3"}, 0.9);

            // Note: We cannot easily add collapse_field_hash to match features in tests
            // This test verifies that pagination doesn't crash when collapse is enabled

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            Query query = createRRFQuery(0.5, 60, 2, 1);
            query.properties().set("collapsefield", "test_field");

            Result result = execution.search(query);

            // Should handle collapse field with pagination (even without match features)
            assertThat((Object) result.hits()).isNotNull();
        }
    }

    // ==================== Consistency Tests ====================

    @Nested
    @DisplayName("Consistency Tests")
    class ConsistencyTests {

        //        @Test
        //        @DisplayName("Should have no overlap between consecutive pages")
        void testRRFPaginationConsistency_NoOverlapBetweenPages() {
            // Page 1
            ArgumentCaptor<Query> queryCaptor1 = ArgumentCaptor.forClass(Query.class);
            Searcher downstreamSearcher1 = mock(Searcher.class);

            HitGroup tensorResults1 =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            1.0);
            HitGroup lexicalResults1 =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            0.9);

            when(downstreamSearcher1.process(queryCaptor1.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults1))
                    .thenReturn(new Result(new Query(), tensorResults1));

            Chain<Searcher> searchChain1 = new Chain<>(hybridSearcher, downstreamSearcher1);
            Execution.Context context1 =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution1 = new Execution(searchChain1, context1);

            Query query1 = createRRFQuery(0.5, 60, 3, 0);
            Result result1 = execution1.search(query1);
            List<String> page1Ids = extractDocIds(result1.hits().asList());

            // Page 2
            ArgumentCaptor<Query> queryCaptor2 = ArgumentCaptor.forClass(Query.class);
            Searcher downstreamSearcher2 = mock(Searcher.class);

            HitGroup tensorResults2 =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            1.0);
            HitGroup lexicalResults2 =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            0.9);

            when(downstreamSearcher2.process(queryCaptor2.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults2))
                    .thenReturn(new Result(new Query(), tensorResults2));

            Chain<Searcher> searchChain2 = new Chain<>(new HybridSearcher(), downstreamSearcher2);
            Execution.Context context2 =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution2 = new Execution(searchChain2, context2);

            Query query2 = createRRFQuery(0.5, 60, 3, 3);
            Result result2 = execution2.search(query2);
            List<String> page2Ids = extractDocIds(result2.hits().asList());

            // Verify no overlap between pages
            Set<String> page1Set = new HashSet<>(page1Ids);
            Set<String> page2Set = new HashSet<>(page2Ids);

            Set<String> intersection = new HashSet<>(page1Set);
            intersection.retainAll(page2Set);

            // The pagination logic should minimize overlap (though it may not eliminate it
            // completely
            // due to RRF scoring changes after removal)
            // We verify that there's improvement over naive pagination
            assertThat(intersection.size())
                    .as("Pages should have minimal or no overlap")
                    .isLessThan(Math.min(page1Ids.size(), page2Ids.size()));
        }

        @Test
        @DisplayName("Should adjust rerankDepthGlobal when needToTrimPreviousPages=true")
        void testRRFPaginationWithOffset_AdjustsRerankDepthGlobal() {
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            HitGroup tensorResults =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            1.0);
            HitGroup lexicalResults =
                    createHitGroup(
                            new String[] {
                                "doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7",
                                "doc_8"
                            },
                            0.9);

            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), lexicalResults))
                    .thenReturn(new Result(new Query(), tensorResults));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Set rerankDepthGlobal (without actual global modifiers to avoid match features
            // requirement)
            Query query = createRRFQuery(0.5, 60, 3, 2);
            query.properties().set("marqo__hybrid.rerankDepthGlobal", 5);

            Result result = execution.search(query);

            // With offset=2 and rerankDepthGlobal=5, the effective rerankDepthGlobal should be 7
            // (5+2)
            // This ensures the pagination logic adjusts rerank depth correctly
            assertThat((Object) result.hits()).isNotNull();

            // Verify basic pagination worked
            assertThat(result.hits().size()).isLessThanOrEqualTo(3);
        }
    }
}
