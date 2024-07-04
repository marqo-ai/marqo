package ai.marqo.search;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.*;
import com.yahoo.search.searchchain.*;
import org.junit.Test;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.result.Hit;
import static org.assertj.core.api.Assertions.assertThat;
import org.assertj.core.api.Assertions;


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
        Assertions.assertThat(result).hasSize(6);

        // Check if the hits are interleaved correctly
//        assertThat(result.get(0).getId().toString().equals("both1"));
//        assertThat(result.get(1).getId().toString().equals("both2"));
//        assertThat(result.get(2).getId().toString().equals("tensor1") || result.get(2).getId().toString().equals("lexical1"));  // Result ordering is ambiguous for ties
//        assertThat(result.get(3).getId().toString().equals("tensor1") || result.get(3).getId().toString().equals("lexical1"));
//        assertThat(result.get(4).getId().toString().equals("tensor2") || result.get(4).getId().toString().equals("lexical2"));
//        assertThat(result.get(5).getId().toString().equals("tensor2") || result.get(5).getId().toString().equals("lexical2"));

        // Check RRF score is calculated correctly
        //assertEquals(result.get(0).getRelevance().getScore(), (alpha * (1.0 / (5 + k))) + (alpha * (1.0 / (4 + k)))); // appears in both lists
        //assertEquals(result.get(1).getRelevance().getScore(), (alpha * (1.0 / (6 + k))) + (alpha * (1.0 / (5 + k)))); // appears in both lists
        //assertEquals(result.get(2).getRelevance().getScore(), alpha * (1.0 / (1 + k)));     // 1st in tensor/lexical list
        //assertEquals(result.get(3).getRelevance().getScore(), alpha * (1.0 / (1 + k)));     // 1st in tensor/lexical list
        //assertEquals(result.get(4).getRelevance().getScore(), alpha * (1.0 / (2 + k)));     // 2nd in tensor/lexical list
        //assertEquals(result.get(5).getRelevance().getScore(), alpha * (1.0 / (2 + k)));     // 2nd in tensor/lexical list

        // Check raw score is encoded
        /*assertEquals(result.get(0).getField("marqo__raw_tensor_score"), 0.4);
        assertEquals(result.get(0).getField("marqo__raw_lexical_score"), 0.45);
        assertEquals(result.get(1).getField("marqo__raw_tensor_score"), 0.3);
        assertEquals(result.get(1).getField("marqo__raw_lexical_score"), 0.44);

        assertEquals(result.get(1).getRelevance().getScore(), (alpha * (1.0 / (6 + k))) + (alpha * (1.0 / (5 + k)))); // appears in both lists
        assertEquals(result.get(2).getRelevance().getScore(), alpha * (1.0 / (1 + k)));     // 1st in tensor/lexical list
        assertEquals(result.get(3).getRelevance().getScore(), alpha * (1.0 / (1 + k)));     // 1st in tensor/lexical list
        assertEquals(result.get(4).getRelevance().getScore(), alpha * (1.0 / (2 + k)));     // 2nd in tensor/lexical list
        assertEquals(result.get(5).getRelevance().getScore(), alpha * (1.0 / (2 + k)));     // 2nd in tensor/lexical list*/
    }

    @Test
    public void testCreateSubQuery() {
        // Create Hybrid Searcher
        HybridSearcher testSearcher = new HybridSearcher();
    }
}