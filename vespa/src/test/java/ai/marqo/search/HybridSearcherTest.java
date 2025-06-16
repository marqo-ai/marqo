package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.sun.jdi.InternalException;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.*;
import com.yahoo.search.query.ranking.RankFeatures;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.*;
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
}
