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
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.*;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.List;
import java.util.Map;
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
            assertThat(result.asList()).hasSize(6);

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
                            new Hit("index:test/0/tensor2", alpha * (1.0 / (2 + k))));

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
            assertThat(result.asList()).hasSize(6);

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
                            new Hit("index:test/6/tensor2", alpha * (1.0 / (2 + k))));

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
        }
    }

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
}
