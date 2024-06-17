package ai.marqo.search.test;

import com.yahoo.component.chain.Chain;
import com.yahoo.schema.derived.SchemaInfo;
import com.yahoo.search.*;
import com.yahoo.search.searchchain.*;
import ai.marqo.search.HybridSearcher;
import ai.marqo.search.HybridSearcher;

public class HybridSearcherTestCase extends junit.framework.TestCase {

    public void testBasics() {
        // TODOS:
        // Do logging solution first tbh lol. Better than trace.
        // Create Context with correct info. Schemas should have proper TENSOR/LEXICAL rank profiles.
        // Add fake docs (maybe 6)
        // Have tensor and lexical search return different results with specific scores.
        // Show RRF interleaves the results.



        // Create chain
        Chain<Searcher> searchChain = new Chain<Searcher>(new HybridSearcher());

        // Create an empty context, in a running container this would be
        // populated with settings used by different searcher. Tests must
        // set this according to their own requirements.
        SearchChainRegistry searchChainRegistry = new SearchChainRegistry();
        Execution.Context context = Execution.Context.createContextStub(searchChainRegistry);
        Execution execution = new Execution(searchChain, context);

        // Execute it
        Result result = execution.search(new Query("search/?query=somequery"));

        assertNotNull(result.hits());
        // Assert the result has the expected hit by scanning for the ID
        //assertNotNull(result.hits().get("test"));
    }

}