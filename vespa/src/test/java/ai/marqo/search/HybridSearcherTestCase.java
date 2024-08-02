package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.*;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.*;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.Tensor.Cell;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class HybridSearcherTestCase {

    private HybridSearcher hybridSearcher;

    @Mock private Searcher downstreamSearcher;

    @Before
    public void setUp() {
        hybridSearcher = new HybridSearcher();
    }

    @Test
    public void testHybridSearcher() {
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher, downstreamSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);

        int k = 60;
        double alpha = 0.5;

        Query query = getHybridQuery(k, alpha, "test", "disjunction", "rrf");

        HitGroup hitsTensor = new HitGroup();
        hitsTensor.add(new Hit("tensor1", 1.0));
        hitsTensor.add(new Hit("both", 0.4));

        HitGroup hitsLexical = new HitGroup();
        hitsLexical.add(new Hit("both", 0.45));

        ArgumentCaptor<Query> queryArgumentCaptor = ArgumentCaptor.forClass(Query.class);

        when(downstreamSearcher.process(queryArgumentCaptor.capture(), any(Execution.class)))
                .thenReturn(new Result(query, hitsLexical))
                .thenReturn(new Result(query, hitsTensor));

        Result result = execution.search(query);
        // verify the result is the fused hit group
        assertThat(result).isNotNull();
        assertThat(result.hits().get(0))
                .isEqualTo(new Hit("both", alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))));
        assertThat(result.hits().get(0).fields())
                .containsAllEntriesOf(
                        Map.of("marqo__raw_tensor_score", 0.4, "marqo__raw_lexical_score", 0.45));

        // verify the correct queries are constructed
        List<Query> allQueries = queryArgumentCaptor.getAllValues();
        assertThat(allQueries).hasSize(2);
        assertThat(allQueries.get(0).properties().get("yql")).isEqualTo("lexical yql");
        assertThat(allQueries.get(1).properties().get("yql")).isEqualTo("tensor yql");
    }

    private static Query getHybridQuery(int k, double alpha, String queryString,
                                        String retrievalMethod, String rankingMethod) {
        Query query = new Query("search/?query=" + queryString);
        query.properties().set("marqo__hybrid.retrievalMethod", retrievalMethod);
        query.properties().set("marqo__hybrid.rankingMethod", rankingMethod);
        query.properties().set("marqo__hybrid.rrf_k", k);
        query.properties().set("marqo__hybrid.alpha", alpha);
        query.properties().set("marqo__yql.lexical", "lexical yql");
        query.properties().set("marqo__yql.tensor", "tensor yql");

        // Define the tensor type
        TensorType tensorType = new TensorType.Builder()
                .mapped("test_tensor")
                .build();

        // Create the tensor using the map
        Tensor fields_to_rank_lexical = Tensor.Builder.of(tensorType)
                .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                .cell(TensorAddress.ofLabels("marqo__lexical_text_field_2"), 1.0)
                .build();

        Tensor fields_to_rank_tensor = Tensor.Builder.of(tensorType)
                .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_1"), 1.0)
                .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_2"), 1.0)
                .build();

        query.getRanking().getFeatures().put("query(marqo__fields_to_rank_lexical)", fields_to_rank_lexical);
        query.getRanking().getFeatures().put("query(marqo__fields_to_rank_tensor)", fields_to_rank_tensor);
        return query;
    }

    @Test
    public void testRRF() {
        // Create Hybrid Searcher
        HybridSearcher testSearcher = new HybridSearcher();

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
        hitsTensor.add(new Hit("tensor1", 1.0));
        hitsTensor.add(new Hit("tensor2", 0.8));
        hitsTensor.add(new Hit("tensor3", 0.6));
        hitsTensor.add(new Hit("tensor4", 0.5));
        hitsTensor.add(new Hit("both1", 0.4));
        hitsTensor.add(new Hit("both2", 0.3));

        // Create lexical hits
        HitGroup hitsLexical = new HitGroup();
        hitsLexical.add(new Hit("lexical1", 1.0));
        hitsLexical.add(new Hit("lexical2", 0.7));
        hitsLexical.add(new Hit("lexical3", 0.5));
        hitsLexical.add(new Hit("both1", 0.45));
        hitsLexical.add(new Hit("both2", 0.44));

        // Set parameters
        int k = 60;
        double alpha = 0.5;
        boolean verbose = false;

        // Call the rrf function
        HitGroup result = testSearcher.rrf(hitsTensor, hitsLexical, k, alpha, verbose);

        // Check that the result size is correct
        assertThat(result.asList()).hasSize(6);

        // Check that result order and scores are correct
        assertThat(result.asList())
                .containsExactly(
                        // Score should be a sum (tensor rank and lexical rank)
                        new Hit("both1", alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))),
                        // Score should be a sum (tensor rank and lexical rank)
                        new Hit("both2", alpha * (1.0 / (6 + k)) + alpha * (1.0 / (5 + k))),
                        // Since tie, lexical was put first. Likely due to alphabetical ID.
                        new Hit("lexical1", alpha * (1.0 / (1 + k))),
                        new Hit("tensor1", alpha * (1.0 / (1 + k))),
                        new Hit("lexical2", alpha * (1.0 / (2 + k))),
                        new Hit("tensor2", alpha * (1.0 / (2 + k))));

        assertThat(result.get(0).fields())
                .containsAllEntriesOf(
                        Map.of("marqo__raw_tensor_score", 0.4, "marqo__raw_lexical_score", 0.45));
        assertThat(result.get(1).fields())
                .containsAllEntriesOf(
                        Map.of("marqo__raw_tensor_score", 0.3, "marqo__raw_lexical_score", 0.44));
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
    public void testCreateSubQueryAddsFieldsToRank() {
        /*
        Tests that for fields to rank (rank features) are set for both retrieval and rank method.
         */
        // Create Hybrid Searcher
        HybridSearcher testSearcher = new HybridSearcher();

        String[][] testCases = {
                {"lexical", "lexical"},
                {"tensor", "tensor"},
                {"lexical", "tensor"},
                {"tensor", "lexical"}
        };

        for (String[] testCase : testCases) {
            String retrievalMethod = testCase[0];
            String rankingMethod = testCase[1];

            // Create test query
            Query query = getHybridQuery(60, 0.5, "test", retrievalMethod, rankingMethod);

            // Create sub query
            Query subQuery = testSearcher.createSubQuery(query, retrievalMethod, rankingMethod, true);
            if (retrievalMethod.equals("lexical") || rankingMethod.equals("lexical")) {
                assertThat(subQuery.getRanking().getFeatures().getDouble("query(marqo__lexical_text_field_1)").getAsDouble()).isEqualTo(1.0);
                assertThat(subQuery.getRanking().getFeatures().getDouble("query(marqo__lexical_text_field_2)").getAsDouble()).isEqualTo(1.0);
            }
            if (retrievalMethod.equals("tensor") || rankingMethod.equals("tensor")) {
                assertThat(subQuery.getRanking().getFeatures().getDouble("query(marqo__embeddings_text_field_1)").getAsDouble()).isEqualTo(1.0);
                assertThat(subQuery.getRanking().getFeatures().getDouble("query(marqo__embeddings_text_field_2)").getAsDouble()).isEqualTo(1.0);
            }
        }


    }
}
