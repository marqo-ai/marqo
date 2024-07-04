package ai.marqo.search;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.*;
import com.yahoo.search.searchchain.*;
import org.junit.Test;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.result.Hit;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;


public class HybridSearcherTestCase {

    public void testHybridSearcher() {
        // Create chain
        Chain<Searcher> searchChain = new Chain<Searcher>(new HybridSearcher());

        // Create an empty context, in a running container this would be
        // populated with settings used by different searcher. Tests must
        // set this according to their own requirements.
        SearchChainRegistry searchChainRegistry = new SearchChainRegistry();
        Execution.Context context = Execution.Context.createContextStub(searchChainRegistry);
        Execution execution = new Execution(searchChain, context);

        // Execute it
        //Result result = execution.search(new Query("search/?query=somequery"));

        //assertNotNull(result.hits());
        // Assert the result has the expected hit by scanning for the ID
        //assertNotNull(result.hits().get("test"));
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
        Integer k = 60;
        Double alpha = 0.5;
        boolean verbose = false;

        // Call the rrf function
        HitGroup result = testSearcher.rrf(hitsTensor, hitsLexical, k, alpha, verbose);

        // Check that the result size is correct
        assertThat(result.asList()).hasSize(6);

        // Check that result order and scores are correct
        assertThat(result.asList()).containsExactly(
                new Hit("both1", alpha * (1.0 / (5 + k)) + alpha * (1.0 / (4 + k))),    // Score should be a sum (tensor rank and lexical rank)
                new Hit("both2", alpha * (1.0 / (6 + k)) + alpha * (1.0 / (5 + k))),    // Score should be a sum (tensor rank and lexical rank)
                new Hit("lexical1", alpha * (1.0 / (1 + k))),           // Since tie, lexical was put first. Likely due to alphabetical ID.
                new Hit("tensor1", alpha * (1.0 / (1 + k))),
                new Hit("lexical2", alpha * (1.0 / (2 + k))),
                new Hit("tensor2", alpha * (1.0 / (2 + k)))
        );

        // TODO: Add back when Map is fixed
        //assertThat(result.get(0).fields()).containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.4, "marqo__raw_lexical_score", 0.45));
        //assertThat(result.get(1).fields()).containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.3, "marqo__raw_lexical_score", 0.44));
        //assertThat(result.get(2).fields()).containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 1.0));
        //assertThat(result.get(3).fields()).containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 1.0));
        //assertThat(result.get(4).fields()).containsAllEntriesOf(Map.of("marqo__raw_lexical_score", 0.7));
        //assertThat(result.get(5).fields()).containsAllEntriesOf(Map.of("marqo__raw_tensor_score", 0.8));
    }

    @Test
    public void testCreateSubQuery() {
        // Create Hybrid Searcher
        HybridSearcher testSearcher = new HybridSearcher();
    }
}