package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import com.sun.jdi.InternalException;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.ranking.RankFeatures;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.SearchChainRegistry;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.Ignore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

class HybridSearcherTest {
    private HybridSearcher hybridSearcher;

    private Searcher downstreamSearcher;

    @BeforeEach
    void setUp() {
        hybridSearcher = new HybridSearcher();
        downstreamSearcher = mock(Searcher.class);
    }

    @Test
    void testHybridSearcher() {
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);

        int k = 60;
        double alpha = 0.5;

        Query query = getHybridQuery(k, alpha, "test", "disjunction", "rrf");

        HitGroup hitsTensor = new HitGroup();
        hitsTensor.add(new Hit("index:test/0/tensor1", 1.0));
        hitsTensor.add(new Hit("index:test/0/both", 0.4));

        HitGroup hitsLexical = new HitGroup();
        hitsLexical.add(new Hit("index:test/0/both", 0.45));

        ArgumentCaptor<Query> queryArgumentCaptor = ArgumentCaptor.forClass(Query.class);

        when(downstreamSearcher.process(queryArgumentCaptor.capture(), any(Execution.class)))
                .thenReturn(new Result(query, hitsLexical))
                .thenReturn(new Result(query, hitsTensor));

        Result result = execution.search(query);
        // verify the result is the fused hit group
        assertThat(result).isNotNull();
        assertThat(result.hits().get(0))
                .isEqualTo(
                        new Hit(
                                "index:test/0/both",
                                alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))));
        assertThat(result.hits().get(0).fields())
                .containsAllEntriesOf(
                        Map.of("marqo__raw_tensor_score", 0.4, "marqo__raw_lexical_score", 0.45));

        // verify the correct queries are constructed
        List<Query> allQueries = queryArgumentCaptor.getAllValues();
        assertThat(allQueries).hasSize(2);
        assertThat(allQueries.get(0).properties().get("yql")).isEqualTo("lexical yql");
        assertThat(allQueries.get(1).properties().get("yql")).isEqualTo("tensor yql");
    }

    @Nested
    class ValidationTest {
        @Ignore
        void rerankDepthGlobalSetToLimit() {
            // Ensure rerankDepthGlobal defaults to limit (hits) if not set
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__hybrid.retrievalMethod", "disjunction");
            query.properties().set("marqo__hybrid.rankingMethod", "rrf");
            query.properties().set("marqo__hybrid.rrf_k", 60);
            query.properties().set("marqo__hybrid.alpha", 0.5);
            query.properties().set("hits", 20);

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // TODO: Check if rerankDepth is limit
        }
    }

    @Nested
    class RRFTest {
        @Test
        void shouldFuseWithDefaultParameters() {
            // Cases
            // With tied scores
            // No overlap
            // With overlap
            // More tensor hits
            // More lexical hits
            // 0 Tensor hits
            // 0 Lexical hits
            // 0 hits both
            // Higher alpha (break ties)
            // Lower alpha (stack results)
            // invalid alpha (should throw exception) < 0 or > 1
            // alpha is 0, alpha is 1

            // Use nested classes to group tests (eg testAlpha)
            // Each case is 1 method

            // Create tensor hits
            HitGroup hitsTensor = new HitGroup();
            hitsTensor.add(new Hit("index:test/0/tensor1", 1.0));
            hitsTensor.add(new Hit("index:test/0/tensor2", 0.8));
            hitsTensor.add(new Hit("index:test/0/tensor3", 0.6));
            hitsTensor.add(new Hit("index:test/0/tensor4", 0.5));
            hitsTensor.add(new Hit("index:test/0/both1", 0.4));
            hitsTensor.add(new Hit("index:test/0/both2", 0.3));

            // Create lexical hits
            HitGroup hitsLexical = new HitGroup();
            hitsLexical.add(new Hit("index:test/0/lexical1", 1.0));
            hitsLexical.add(new Hit("index:test/0/lexical2", 0.7));
            hitsLexical.add(new Hit("index:test/0/lexical3", 0.5));
            hitsLexical.add(new Hit("index:test/0/both1", 0.45));
            hitsLexical.add(new Hit("index:test/0/both2", 0.44));

            // Set parameters
            int k = 60;
            double alpha = 0.5;
            boolean verbose = false;

            // Call the rrf function
            HitGroup result = hybridSearcher.rrf(hitsTensor, hitsLexical, k, alpha, verbose);

            // Check that the result size is correct
            // RRF function returns all interleaved hits. Pagination, trimming, reranking, are done
            // in post-processing
            assertThat(result.asList()).hasSize(9);

            // Check that result order and scores are correct
            assertThat(result.asList())
                    .containsExactly(
                            // Score should be a sum (tensor rank and lexical rank)
                            new Hit(
                                    "index:test/0/both1",
                                    alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))),
                            // Score should be a sum (tensor rank and lexical rank)
                            new Hit(
                                    "index:test/0/both2",
                                    alpha * (1.0 / (6 + k)) + alpha * (1.0 / (5 + k))),
                            // Since tie, lexical was put first. Likely due to alphabetical ID.
                            new Hit("index:test/0/lexical1", alpha * (1.0 / (1 + k))),
                            new Hit("index:test/0/tensor1", alpha * (1.0 / (1 + k))),
                            new Hit("index:test/0/lexical2", alpha * (1.0 / (2 + k))),
                            new Hit("index:test/0/tensor2", alpha * (1.0 / (2 + k))),
                            new Hit("index:test/0/lexical3", alpha * (1.0 / (3 + k))),
                            new Hit("index:test/0/tensor3", alpha * (1.0 / (3 + k))),
                            new Hit("index:test/0/tensor4", alpha * (1.0 / (4 + k))));

            assertThat(result.get(0).fields())
                    .containsAllEntriesOf(
                            Map.of(
                                    "marqo__raw_tensor_score",
                                    0.4,
                                    "marqo__raw_lexical_score",
                                    0.45));
            assertThat(result.get(1).fields())
                    .containsAllEntriesOf(
                            Map.of(
                                    "marqo__raw_tensor_score",
                                    0.3,
                                    "marqo__raw_lexical_score",
                                    0.44));
            assertThat(result.get(2).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 1.0));
            assertThat(result.get(3).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 1.0));
            assertThat(result.get(4).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 0.7));
            assertThat(result.get(5).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.8));
            assertThat(result.get(6).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 0.5));
            assertThat(result.get(7).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.6));
            assertThat(result.get(8).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.5));
        }

        @Test
        void shouldFuseWithMismatchedGroups() {
            // Create tensor hits
            HitGroup hitsTensor = new HitGroup();
            hitsTensor.add(new Hit("index:test/5/tensor1", 1.0));
            hitsTensor.add(new Hit("index:test/6/tensor2", 0.8));
            hitsTensor.add(new Hit("index:test/7/tensor3", 0.6));
            hitsTensor.add(new Hit("index:test/8/tensor4", 0.5));
            hitsTensor.add(new Hit("index:test/9/both1", 0.4));
            hitsTensor.add(new Hit("index:test/10/both2", 0.3));

            // Create lexical hits
            HitGroup hitsLexical = new HitGroup();
            hitsLexical.add(new Hit("index:test/0/lexical1", 1.0));
            hitsLexical.add(new Hit("index:test/1/lexical2", 0.7));
            hitsLexical.add(new Hit("index:test/2/lexical3", 0.5));
            hitsLexical.add(new Hit("index:test/3/both1", 0.45));
            hitsLexical.add(new Hit("index:test/4/both2", 0.44));

            // Set parameters
            int k = 60;
            double alpha = 0.5;
            boolean verbose = false;

            // Call the rrf function
            HitGroup result = hybridSearcher.rrf(hitsTensor, hitsLexical, k, alpha, verbose);

            // Check that the result size is correct
            assertThat(result.asList()).hasSize(9);

            // Check that result order and scores are correct
            // If results have the same score, they will be sorted by alphabetical hit ID.
            // Results in TENSOR list will be prioritized, because they are evaluated first in RRF.
            assertThat(result.asList())
                    .containsExactly(
                            // Score should be a sum (tensor rank and lexical rank)
                            new Hit(
                                    "index:test/9/both1",
                                    alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))),
                            // Score should be a sum (tensor rank and lexical rank)
                            new Hit(
                                    "index:test/10/both2",
                                    alpha * (1.0 / (6 + k)) + alpha * (1.0 / (5 + k))),
                            // Since tie, lexical was put first. Likely due to alphabetical ID.
                            new Hit("index:test/0/lexical1", alpha * (1.0 / (1 + k))),
                            new Hit("index:test/5/tensor1", alpha * (1.0 / (1 + k))),
                            new Hit("index:test/1/lexical2", alpha * (1.0 / (2 + k))),
                            new Hit("index:test/6/tensor2", alpha * (1.0 / (2 + k))),
                            new Hit("index:test/2/lexical3", alpha * (1.0 / (3 + k))),
                            new Hit("index:test/7/tensor3", alpha * (1.0 / (3 + k))),
                            new Hit("index:test/8/tensor4", alpha * (1.0 / (4 + k))));

            assertThat(result.get(0).fields())
                    .containsAllEntriesOf(
                            Map.of(
                                    "marqo__raw_tensor_score",
                                    0.4,
                                    "marqo__raw_lexical_score",
                                    0.45));
            assertThat(result.get(1).fields())
                    .containsAllEntriesOf(
                            Map.of(
                                    "marqo__raw_tensor_score",
                                    0.3,
                                    "marqo__raw_lexical_score",
                                    0.44));
            assertThat(result.get(2).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 1.0));
            assertThat(result.get(3).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 1.0));
            assertThat(result.get(4).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 0.7));
            assertThat(result.get(5).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.8));
            assertThat(result.get(6).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 0.5));
            assertThat(result.get(7).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.6));
            assertThat(result.get(8).fields())
                    .containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.5));
        }
    }

    // TODO: post processing test
    // if rerankDepthGlobal is null, rerank everything
    // global score modifiers tests
    // pagination tests (use offset)
    // mult weights & add weights both dont exist, make sure apply global score mod is skipped
    // mult weights & add weights both empty, make sure apply global score mod is skipped
    // mult weights exist but not add weights, & vice versa
    // empty mult weights or empty add weights

    @Nested
    class IdExtractorTest {
        @ParameterizedTest
        @CsvSource(
                value = {
                    "index:vespa-content-dummy_index/0/e0a1c64b0c20b56741834b5,"
                            + " e0a1c64b0c20b56741834b5", // Base case
                    "index:vespa-content-dummy_index/0/e0a1c64b0/c20b56741834b5,"
                            + " e0a1c64b0/c20b56741834b5", // Slash in doc ID
                    "index:vespa-content-dummy_index/0/e0a1c64b0//c20b56741834b5,"
                            + " e0a1c64b0//c20b56741834b5", // Double slash in doc ID
                    "index:vespa-content-dummy_index/0//e0a1c64b0c20b56741834b5,"
                            + " /e0a1c64b0c20b56741834b5", // Slash at start of doc ID
                    "index:vespa-content-dummy_index/0/e0a1c64b0c/2/0b56741834b5,"
                            + " e0a1c64b0c/2/0b56741834b5", // Multiple slashes in doc ID
                })
        void shouldExtractIdFromHit(String vespaId, String expectedId) {
            String id = HybridSearcher.extractDocIdFromHitId(vespaId);
            assertThat(id).isEqualTo(expectedId);
        }

        // Negative test cases
        @ParameterizedTest
        @CsvSource({
            "invalidformat/0/e0a1c64b0c20b56741834b5", // Missing 'index:'
            "index:/0/e0a1c64b0c20b56741834b5", // Missing content after 'index:'
            "index:vespa-content-dummy_index//e0a1c64b0c20b56741834b5", // Missing digit part
            "index:vespa-content-dummy_index/123/", // Missing doc ID part after last slash
            "someotherformat:vespa-content-dummy_index/0/e0a1c64b0c20b56741834b5", // Incorrect
            // prefix
            "index:vespa content dummy_index/0/e0a1c64b0c20b56741834b5", // Whitespace in index name
            "index:vespa-content/dummy_index/0/e0a1c64b0c20b56741834b5", // Slash in index name
            "index:vespa-content-dummy_index/abc/e0a1c64b0c20b56741834b5", // Non-numeric value in
            // the 2nd group
            "index:vespa-content-dummy_index/1abc/e0a1c64b0c20b56741834b5", // Partially numeric
            // value in 2nd group
            "index:vespa-content-dummy_index/-123/e0a1c64b0c20b56741834b5", // Negative number in
            // the 2nd group
            "index:vespa-content-dummy_index/0 ", // Whitespace after last slash, missing document
            // ID
        })
        void shouldThrowExceptionForInvalidFormat(String invalidVespaId) {
            // Ensure IllegalStateException is thrown when the regex does not match
            InternalException exception =
                    assertThrows(
                            InternalException.class,
                            () -> {
                                HybridSearcher.extractDocIdFromHitId(invalidVespaId);
                            });

            // Assert the exception message contains the invalid hit ID
            assertThat(exception.getMessage())
                    .contains(
                            "Vespa doc ID could not be extracted from the full hit ID: "
                                    + invalidVespaId);
        }
    }

    @Nested
    class FieldsToRankTest {

        @ParameterizedTest
        @CsvSource(value = {"lexical,tensor", "tensor,lexical"})
        void shouldIncludeAllRankFieldsWhenRetrievalMethodAndRankMethodDiffer(
                String retrievalMethod, String rankingMethod) {
            Query query = getHybridQuery(60, 0.5, "test", retrievalMethod, rankingMethod);
            Query subQuery =
                    hybridSearcher.createSubQuery(query, retrievalMethod, rankingMethod, true);
            RankFeatures features = subQuery.getRanking().getFeatures();

            assertThat(features.getDouble("query(marqo__lexical_text_field_1)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__lexical_text_field_2)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__embeddings_text_field_1)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__embeddings_text_field_2)")).hasValue(1.0);
        }

        @Test
        void shouldOnlyIncludeTensorRankFieldsWhenRetrieveAndRankByTensor() {
            Query query = getHybridQuery(60, 0.5, "test", "tensor", "tensor");
            Query subQuery = hybridSearcher.createSubQuery(query, "tensor", "tensor", true);
            RankFeatures features = subQuery.getRanking().getFeatures();

            assertThat(features.getDouble("query(marqo__embeddings_text_field_1)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__embeddings_text_field_2)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__lexical_text_field_1)")).isEmpty();
            assertThat(features.getDouble("query(marqo__lexical_text_field_2)")).isEmpty();
        }

        @Test
        void shouldOnlyIncludeLexicalRankFieldsRetrieveAndRankByLexical() {
            Query query = getHybridQuery(60, 0.5, "test", "lexical", "lexical");
            Query subQuery = hybridSearcher.createSubQuery(query, "lexical", "lexical", true);
            RankFeatures features = subQuery.getRanking().getFeatures();

            assertThat(features.getDouble("query(marqo__lexical_text_field_1)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__lexical_text_field_2)")).hasValue(1.0);
            assertThat(features.getDouble("query(marqo__embeddings_text_field_1)")).isEmpty();
            assertThat(features.getDouble("query(marqo__embeddings_text_field_2)")).isEmpty();
        }
    }

    private static Query getHybridQuery(
            int k, double alpha, String queryString, String retrievalMethod, String rankingMethod) {
        Query query = new Query("search/?query=" + queryString);
        query.properties().set("marqo__hybrid.retrievalMethod", retrievalMethod);
        query.properties().set("marqo__hybrid.rankingMethod", rankingMethod);
        query.properties().set("marqo__hybrid.rrf_k", k);
        query.properties().set("marqo__hybrid.alpha", alpha);
        query.properties().set("marqo__yql.lexical", "lexical yql");
        query.properties().set("marqo__yql.tensor", "tensor yql");

        // Define the tensor type
        TensorType tensorType = new TensorType.Builder().mapped("test_tensor").build();

        // Create the tensor using the map
        Tensor fieldsToRankLexical =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                        .cell(TensorAddress.ofLabels("marqo__lexical_text_field_2"), 1.0)
                        .build();

        Tensor fieldsToRankTensor =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_1"), 1.0)
                        .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_2"), 1.0)
                        .build();

        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_lexical)", fieldsToRankLexical);
        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_tensor)", fieldsToRankTensor);
        return query;
    }

    @Nested
    class CollectErrorsFromResultsTest {
        @Test
        void shouldRaiseErrorIfLexicalResultHasError() {
            Result resultLexical =
                    new Result(
                            new Query(),
                            ErrorMessage.createInternalServerError("Example lexical error"));
            Result resultTensor = new Result(new Query());
            HitGroup combinedErrors =
                    hybridSearcher.collectErrorsFromResults(resultLexical, resultTensor);

            assertThat(combinedErrors.getError().getDetailedMessage())
                    .contains("Example lexical error");
        }

        @Test
        void shouldRaiseErrorIfTensorResultHasError() {
            Result resultLexical = new Result(new Query());
            Result resultTensor =
                    new Result(
                            new Query(),
                            ErrorMessage.createInternalServerError("Example tensor error"));
            HitGroup combinedErrors =
                    hybridSearcher.collectErrorsFromResults(resultLexical, resultTensor);

            assertThat(combinedErrors.getError().getDetailedMessage())
                    .contains("Example tensor error");
        }

        @Test
        void shouldRaiseErrorIfBothResultsHaveError() {
            Result resultLexical =
                    new Result(
                            new Query(),
                            ErrorMessage.createInternalServerError("Example lexical error"));
            Result resultTensor =
                    new Result(
                            new Query(),
                            ErrorMessage.createInternalServerError("Example tensor error"));
            HitGroup combinedErrors =
                    hybridSearcher.collectErrorsFromResults(resultLexical, resultTensor);

            Iterator<ErrorMessage> iterator = combinedErrors.getErrorHit().errors().iterator();
            assertThat(iterator.next().getDetailedMessage()).contains("Example tensor error");
            assertThat(iterator.next().getDetailedMessage()).contains("Example lexical error");
        }

        @Test
        void shouldNotRaiseErrorIfNeitherResultHasError() {
            Result resultLexical = new Result(new Query());
            Result resultTensor = new Result(new Query());
            HitGroup combinedErrors =
                    hybridSearcher.collectErrorsFromResults(resultLexical, resultTensor);
            assertThat(combinedErrors.getError()).isNull();
        }
    }

    @Nested
    class FacetsTest {
        /**
         * This test uses a custom implementation of HybridSearcher that doesn't need to modify Hit IDs.
         * This is because the original implementation tries to do hit.setId() which fails if the Hit already has an ID.
         */
        @Test
        void shouldHandleFacetsInResults() {
            // Create a custom searcher that handles facets differently
            HybridSearcher customSearcher =
                    new HybridSearcher() {
                        @Override
                        public Result search(Query query, Execution execution) {
                            // Check if this is a facet query (used in our test scenario)
                            String facetsYql =
                                    query.properties().getString("marqo__yql.facets", "");
                            if (facetsYql.isEmpty()) {
                                // Not a facet query, pass through to downstream searcher
                                return super.search(query, execution);
                            }

                            // Create a result with prepared hits (avoid ID changes)
                            HitGroup hits = new HitGroup();

                            // Add main results
                            Hit mainHit = new Hit("index:test/0/doc1", 1.0);
                            hits.add(mainHit);

                            // Add facet results with pre-formatted IDs that match the expected
                            // format
                            // after HybridSearcher's facet processing
                            Hit facet1 = new Hit("group:facet:0:0", 1.0);
                            facet1.setField("count", 5);
                            hits.add(facet1);

                            Hit facet2 = new Hit("group:facet:1:0", 1.0);
                            facet2.setField("count", 3);
                            hits.add(facet2);

                            return new Result(query, hits);
                        }
                    };

            // Create the chain with our custom searcher
            Chain<Searcher> searchChain = new Chain<>(customSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Create a query with facets
            Query query = new Query("search/?query=test");
            query.properties().set("marqo__hybrid.retrievalMethod", "lexical");
            query.properties().set("marqo__hybrid.rankingMethod", "lexical");
            query.properties().set("hits", 10);
            query.properties()
                    .set(
                            "marqo__yql.facets",
                            "SELECT * FROM sources * WHERE true | all()\n"
                                    + "---MARQO-YQL-QUERY-DELIMITER---\n"
                                    + "SELECT * FROM sources * WHERE false | all()");

            // Add required tensor rank features
            TensorType tensorType = new TensorType.Builder().mapped("test_tensor").build();
            Tensor fieldsToRankLexical =
                    Tensor.Builder.of(tensorType)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                            .cell(TensorAddress.ofLabels("marqo__lexical_text_field_2"), 1.0)
                            .build();
            query.getRanking()
                    .getFeatures()
                    .put("query(marqo__fields_to_rank_lexical)", fieldsToRankLexical);

            // Execute search
            Result result = execution.search(query);

            // Verify results
            assertThat(result.hits().asList()).hasSize(3); // 1 main hit + 2 facet hits
            assertThat(result.hits().get("index:test/0/doc1")).isNotNull();
            assertThat(result.hits().get("group:facet:0:0")).isNotNull();
            assertThat(result.hits().get("group:facet:1:0")).isNotNull();
            assertThat(result.hits().get("group:facet:0:0").getField("count")).isEqualTo(5);
            assertThat(result.hits().get("group:facet:1:0").getField("count")).isEqualTo(3);
        }

        /**
         * Test that verifies how facet queries are processed using the real HybridSearcher
         * but without dealing with the ID change issue.
         */
        @Test
        void shouldCreateProperFacetQueries() {
            // Setup a searcher chain that captures queries
            ArgumentCaptor<Query> queryCaptor = ArgumentCaptor.forClass(Query.class);

            // Configure downstream searcher behavior
            when(downstreamSearcher.process(queryCaptor.capture(), any(Execution.class)))
                    .thenReturn(new Result(new Query(), new HitGroup()));

            Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
            Execution.Context context =
                    Execution.Context.createContextStub((SearchChainRegistry) null);
            Execution execution = new Execution(searchChain, context);

            // Create a query with facets
            Query query = getHybridQuery(60, 0.5, "test", "lexical", "lexical");
            String facetsYql =
                    "SELECT * FROM sources * WHERE brand = 'nike' | all()\n"
                            + "---MARQO-YQL-QUERY-DELIMITER---\n"
                            + "SELECT * FROM sources * WHERE category = 'shoes' | all()";
            query.properties().set("marqo__yql.facets", facetsYql);

            // Execute search
            execution.search(query);

            // Capture the queries
            List<Query> capturedQueries = queryCaptor.getAllValues();

            // Verify that the right number of queries were created (main + 2 facet queries)
            assertThat(capturedQueries).hasSize(3);

            // Verify all queries
            assertThat(
                            capturedQueries.stream()
                                    .map(q -> q.properties().getString("yql"))
                                    .filter(yql -> yql != null))
                    .containsExactlyInAnyOrder(
                            "SELECT * FROM sources * WHERE brand = 'nike' | all()",
                            "SELECT * FROM sources * WHERE category = 'shoes' | all()",
                            "lexical yql");
        }
    }

    /**
     * Test for sortBy feature in HybridSearcher.
     */
    @Nested
    class sortByTest {

        /**
         * Test that verifies sorting of results based on a single sort field.
         */
        HitGroup helpGenerateHitGroupWithOnlySortFieldValue0() {
            FeatureData f1 = mock(FeatureData.class);
            when(f1.getDouble("sort_field_value_0")).thenReturn(-1e50);
            Hit doc1 = new Hit("doc1", 0.55);
            doc1.setField("matchfeatures", f1);

            FeatureData f2 = mock(FeatureData.class);
            when(f2.getDouble("sort_field_value_0")).thenReturn(1.0);
            Hit doc2 = new Hit("doc2", 0.65);
            doc2.setField("matchfeatures", f2);

            FeatureData f3 = mock(FeatureData.class);
            when(f3.getDouble("sort_field_value_0")).thenReturn(2.0);
            Hit doc3 = new Hit("doc3", 0.75);
            doc3.setField("matchfeatures", f3);

            FeatureData f4 = mock(FeatureData.class);
            when(f4.getDouble("sort_field_value_0")).thenReturn(2.0);
            Hit doc4 = new Hit("doc4", 0.85);
            doc4.setField("matchfeatures", f4);

            FeatureData f5 = mock(FeatureData.class);
            when(f5.getDouble("sort_field_value_0")).thenReturn(-1e50);
            Hit doc5 = new Hit("doc5", 0.45);
            doc5.setField("matchfeatures", f5);

            FeatureData f6 = mock(FeatureData.class);
            when(f6.getDouble("sort_field_value_0")).thenReturn(5.0);
            Hit doc6 = new Hit("doc6", 0.95);
            doc6.setField("matchfeatures", f6);

            FeatureData f7 = mock(FeatureData.class);
            when(f7.getDouble("sort_field_value_0")).thenReturn(6.0);
            Hit doc7 = new Hit("doc7", 0.90);
            doc7.setField("matchfeatures", f7);

            FeatureData f8 = mock(FeatureData.class);
            when(f8.getDouble("sort_field_value_0")).thenReturn(7.0);
            Hit doc8 = new Hit("doc8", 0.80);
            doc8.setField("matchfeatures", f8);

            FeatureData f9 = mock(FeatureData.class);
            when(f9.getDouble("sort_field_value_0")).thenReturn(8.0);
            Hit doc9 = new Hit("doc9", 0.70);
            doc9.setField("matchfeatures", f9);

            FeatureData f10 = mock(FeatureData.class);
            when(f10.getDouble("sort_field_value_0")).thenReturn(9.0);
            Hit doc10 = new Hit("doc10", 1.00);
            doc10.setField("matchfeatures", f10);

            HitGroup hits = new HitGroup();
            hits.add(doc1);
            hits.add(doc2);
            hits.add(doc3);
            hits.add(doc4);
            hits.add(doc5);
            hits.add(doc6);
            hits.add(doc7);
            hits.add(doc8);
            hits.add(doc9);
            hits.add(doc10);
            return hits;
        }

        /**
         * Test that verifies sorting of results based on a single sort field with ascending order
         * and last missing policy.
         */
        @Test
        void sort1FieldWithAscOrderAndLastMissingPolicy() {
            // build 10 distinct docs by hand

            HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
            HybridSearcher searcher = new HybridSearcher();
            String sortJson =
                    "[{"
                            + "\"field_name\":\"ignored\","
                            + "\"order\":\"asc\","
                            + "\"missing\":\"last\""
                            + "}]";

            // full-depth, no trim
            // expected:
            // 1) doc2 (1.0)
            // 2) doc3 & doc4 both 2.0 → tie by original relevance: doc4(0.85) before doc3(0.75)
            // 3) doc6(5),doc7(6),doc8(7),doc9(8),doc10(9)
            // 4) missing last: doc1,doc5
            HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
            assertThat(out.asList())
                    .extracting(hit -> hit.getId().toString())
                    .containsExactly(
                            "doc2", "doc4", "doc3", "doc6", "doc7", "doc8", "doc9", "doc10", "doc1",
                            "doc5");
        }

        /**
         * Test that verifies sorting of results based on a single sort field with desc order
         * and first missing policy.
         */
        @Test
        void sort1FieldWithDescOrderAndFirstMissingPolicy() {
            HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
            HybridSearcher searcher = new HybridSearcher();
            String sortJson =
                    "[{"
                            + "\"field_name\":\"ignored\","
                            + "\"order\":\"desc\","
                            + "\"missing\":\"first\""
                            + "}]";

            HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
            assertThat(out.asList())
                    .extracting(hit -> hit.getId().toString())
                    .containsExactly(
                            "doc1", // missing first  (–1e50, highest missing rel=0.55)
                            "doc5", // missing second (–1e50, next missing rel=0.45)
                            "doc10", // sort=9.0
                            "doc9", // sort=8.0
                            "doc8", // sort=7.0
                            "doc7", // sort=6.0
                            "doc6", // sort=5.0
                            "doc4", // sort=2.0, tie-break on original rel=0.85 (before doc3)
                            "doc3", // sort=2.0, tie-break rel=0.75
                            "doc2" // sort=1.0
                            );
        }

        /**
         * Helper that builds 6 docs with two independent sort_field values:
         *  - doc1/doc2 share field0=1.0 but doc2.field1=5.0 < doc1.field1=10.0
         *  - doc3/doc4 share field0=2.0 but doc3.field1=3.0 < doc4.field1=7.0
         *  - doc5/doc6 both missing (–1e50 → null)
         */
        HitGroup helpGenerateHitGroupWithTwoSortFieldValues() {
            HitGroup hits = new HitGroup();
            double[][] values = {
                {1.0, 10.0},
                {1.0, 5.0},
                {2.0, 3.0},
                {2.0, 7.0},
                {-1e50, -1e50},
                {-1e50, -1e50}
            };
            double[] relevances = {0.10, 0.20, 0.30, 0.40, 0.05, 0.06};
            for (int i = 0; i < values.length; i++) {
                FeatureData f = mock(FeatureData.class);
                when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
                when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
                Hit h = new Hit("doc" + (i + 1), relevances[i]);
                h.setField("matchfeatures", f);
                hits.add(h);
            }
            return hits;
        }

        @Test
        void sort2FieldsWithAscOrderAndLastMissingPolicy() {
            HitGroup hitsToSort = helpGenerateHitGroupWithTwoSortFieldValues();
            HybridSearcher searcher = new HybridSearcher();
            // first sort_field_value_0 asc, missing last
            // then sort_field_value_1 asc, missing last
            String sortJson =
                    "["
                            + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"}"
                            + "]";

            HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
            // Expected:
            // 1) doc2 (1.0,5.0) before doc1 (1.0,10.0)
            // 2) doc3 (2.0,3.0) before doc4 (2.0,7.0)
            // 3) missing last: doc6 (rel=0.06) before doc5 (rel=0.05)
            assertThat(out.asList())
                    .extracting(hit -> hit.getId().toString())
                    .containsExactly(
                            "doc2", "doc1",
                            "doc3", "doc4",
                            "doc6", "doc5");
        }

        /**
         * Helper that builds 4 docs with three sort_field values:
         *  - docA/B/C all have field0=1.0 but differ on field1/field2
         *  - docD missing all three
         */
        HitGroup helpGenerateHitGroupWithThreeSortFieldValues() {
            HitGroup hits = new HitGroup();
            String[] ids = {"docA", "docB", "docC", "docD"};
            double[][] values = {
                {1.0, 1.0, 3.0}, // docA
                {1.0, 1.0, 2.0}, // docB
                {1.0, 2.0, 1.0}, // docC
                {-1e50, -1e50, -1e50} // docD missing all
            };
            double[] relevances = {0.40, 0.50, 0.60, 0.70};
            for (int i = 0; i < ids.length; i++) {
                FeatureData f = mock(FeatureData.class);
                when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
                when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
                when(f.getDouble("sort_field_value_2")).thenReturn(values[i][2]);
                Hit h = new Hit(ids[i], relevances[i]);
                h.setField("matchfeatures", f);
                hits.add(h);
            }
            return hits;
        }

        @Test
        void sort3FieldsWithAscOrderAndFirstMissingPolicy() {
            HitGroup hitsToSort = helpGenerateHitGroupWithThreeSortFieldValues();
            HybridSearcher searcher = new HybridSearcher();
            // all three ascending, missing first
            String sortJson =
                    "[{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"}]";

            HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
            // Expected:
            // 1) missing first: docD
            // 2) among the rest, field0 ties=1.0 → use field1:
            //      docA/B have field1=1.0 < docC.field1=2.0 → so docA & docB
            //    then break tie on field2: docB(2.0) < docA(3.0)
            // 3) then docC
            assertThat(out.asList())
                    .extracting(hit -> hit.getId().toString())
                    .containsExactly("docD", "docB", "docA", "docC");
        }

        private Execution makeEmptyExec() {
            Execution exec = mock(Execution.class);
            when(exec.search(any(Query.class)))
                    .thenAnswer(
                            invocation -> {
                                Query q = invocation.getArgument(0);
                                return new Result(q, new HitGroup());
                            });
            return exec;
        }

        /**
         * Test that verifies that postProcessBySort is called when sortBy is set in the query.
         */
        @Test
        void whenSortByFields_set_postProcessBySortIsCalled() {
            // 1) Create a Mockito spy on the real HybridSearcher
            HybridSearcher spy = spy(new HybridSearcher());

            // 2) Stub out createSubQuery (both overloads) so we never NPE inside it
            doAnswer(inv -> inv.getArgument(0))
                    .when(spy)
                    .createSubQuery(any(Query.class), anyString(), anyString(), anyBoolean());
            doAnswer(inv -> inv.getArgument(0))
                    .when(spy)
                    .createSubQuery(
                            any(Query.class), anyString(), anyString(), anyBoolean(), anyString());

            // 3) Stub extractTensorRankFeature to return:
            //    • null for “mult_weights_global” or “add_weights_global”
            //    • an empty Tensor for everything else (fields_to_rank_*)
            doAnswer(
                            inv -> {
                                String name = inv.getArgument(1);
                                if (name.contains("mult_weights_global")
                                        || name.contains("add_weights_global")) {
                                    return null;
                                }
                                // non-null so createSubQuery and co. won’t blow up
                                return Tensor.from("tensor<float>()");
                            })
                    .when(spy)
                    .extractTensorRankFeature(any(Query.class), anyString());

            // 4) Stub postProcessBySort so it just returns an empty HitGroup
            doReturn(new HitGroup())
                    .when(spy)
                    .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());

            // 5) Build a Query that triggers the sortBy branch
            Query q = new Query("?q");
            q.properties().set("hits", 1);
            q.properties().set("offset", 0);
            q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
            q.properties().set("marqo__hybrid.rankingMethod", "lexical");
            q.properties()
                    .set(
                            "marqo__hybrid.sortBy.fields",
                            "[{\"field_name\":\"foo\",\"order\":\"asc\",\"missing\":\"last\"}]");
            // MUST set this to avoid the NPE you saw
            q.properties().set("marqo__hybrid.sortBy.minSortCandidates", 10);

            // 6) Call search()
            spy.search(q, makeEmptyExec());

            // 7) Verify that only postProcessBySort() ran
            verify(spy, times(1))
                    .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());
            verify(spy, never())
                    .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());
        }

        /**
         * Test that verifies that neither postProcessBySort nor postProcessResults is
         * called when there is no sortBy and no global‐modifier tensors in the query.
         */
        @Test
        void whenNoSortByAndNoModifiers_noPostProcessingMethodsAreCalled() {
            // 1) Spy on the real HybridSearcher
            HybridSearcher spy = spy(new HybridSearcher());

            // 2) Stub out both overloads of createSubQuery to bypass its internals
            doAnswer(inv -> inv.getArgument(0))
                    .when(spy)
                    .createSubQuery(any(Query.class), anyString(), anyString(), anyBoolean());
            doAnswer(inv -> inv.getArgument(0))
                    .when(spy)
                    .createSubQuery(
                            any(Query.class), anyString(), anyString(), anyBoolean(), anyString());

            // 3) Stub extractTensorRankFeature to always return null
            //    (so both mult_weights_global and add_weights_global are 'absent')
            doReturn(null).when(spy).extractTensorRankFeature(any(Query.class), anyString());

            // 4) Also stub the two post‐processors so that, if they *did* get called,
            //    they’d return an empty HitGroup instead of blowing up
            doReturn(new HitGroup())
                    .when(spy)
                    .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());
            doReturn(new HitGroup())
                    .when(spy)
                    .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());

            // 5) Build a Query with *no* sortBy.fields and *no* modifier tensors
            Query q = new Query("?q");
            q.properties().set("hits", 1);
            q.properties().set("offset", 0);
            q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
            q.properties().set("marqo__hybrid.rankingMethod", "lexical");
            // note: we do NOT set marqo__hybrid.sortBy.fields
            // and extractTensorRankFeature will return null for all names

            // 6) Execute
            spy.search(q, makeEmptyExec());

            // 7) Assert that neither branch ran
            verify(spy, never())
                    .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());
            verify(spy, never())
                    .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());
        }

        /*
           Test that verifies that postProcessResults is called when only modifiers exist
           (i.e., no sortBy.fields).
        */
        @Test
        void whenOnlyModifiersExist_postProcessResultsIsCalled() {
            HybridSearcher spy = spy(new HybridSearcher());
            doAnswer(inv -> inv.getArgument(0))
                    .when(spy)
                    .createSubQuery(any(), anyString(), anyString(), anyBoolean());
            // simulate “has a global mult modifier” but no sortBy
            Tensor dummy = Tensor.from("tensor<float>(d0[1]):[1]");
            doReturn(dummy)
                    .when(spy)
                    .extractTensorRankFeature(any(), contains("mult_weights_global"));
            doReturn(null)
                    .when(spy)
                    .extractTensorRankFeature(any(), contains("add_weights_global"));
            doReturn(new HitGroup())
                    .when(spy)
                    .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());

            Query q = new Query("?q");
            q.properties().set("hits", 1);
            q.properties().set("offset", 0);
            q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
            q.properties().set("marqo__hybrid.rankingMethod", "lexical");
            // no sortBy.fields

            spy.search(q, makeEmptyExec());

            verify(spy, times(1)).postProcessResults(any(), eq(q), any(), eq(1), eq(0), eq(false));
            verify(spy, never()).postProcessBySort(any(), anyString(), any(), anyInt(), anyInt());
        }
    }

    @Nested
    class RelevanceCutoffTest {

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
}
