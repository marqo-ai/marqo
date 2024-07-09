package ai.marqo.search;

import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.AsyncExecution;
import com.yahoo.search.searchchain.Execution;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This searcher takes the YQL for both a lexical and tensor search from the query,
 * Creates 2 clone queries
 *
 */
@Before("ExternalYql")
@Provides("HybridReRanking")
public class HybridSearcher extends Searcher {

    Logger logger = LoggerFactory.getLogger(HybridSearcher.class);

    @Override
    public Result search(Query query, Execution execution) {

        MarqoQuery marqoQuery = MarqoQuery.from(query);

        if (marqoQuery.getRetrievalMethod() == RetrievalMethod.disjunction) {
            Result resultLexical, resultTensor;
            Query queryLexical =
                    marqoQuery.createSubQuery(RetrievalMethod.lexical, RankingMethod.lexical);
            Query queryTensor =
                    marqoQuery.createSubQuery(RetrievalMethod.tensor, RankingMethod.tensor);

            // Execute both searches async
            AsyncExecution asyncExecutionLexical = new AsyncExecution(execution);
            Future<Result> futureLexical = asyncExecutionLexical.search(queryLexical);
            AsyncExecution asyncExecutionTensor = new AsyncExecution(execution);
            Future<Result> futureTensor = asyncExecutionTensor.search(queryTensor);
            try {
                resultLexical =
                        futureLexical.get(marqoQuery.getTimeoutMillis(), TimeUnit.MILLISECONDS);
                resultTensor =
                        futureTensor.get(marqoQuery.getTimeoutMillis(), TimeUnit.MILLISECONDS);
            } catch (TimeoutException | InterruptedException | ExecutionException e) {
                logger.warn(
                        "Hybrid search disjunction timeout error. Current timeout:{}",
                        marqoQuery.getTimeoutMillis(),
                        e);
                throw new RuntimeException(e);
            }

            // Execute fusion ranking on 2 results.
            HitGroup fusedHitList =
                    rrf(
                            resultTensor.hits(),
                            resultLexical.hits(),
                            marqoQuery.getRrfK(),
                            marqoQuery.getAlpha(),
                            query);
            query.trace(new NamedHitGroup("RRF Fused", fusedHitList), 1);
            return new Result(query, fusedHitList);

        } else {
            Query combinedQuery = marqoQuery.createSubQuery();
            Result result = execution.search(combinedQuery);
            query.trace(
                    new NamedHitGroup(marqoQuery.getRetrievalMethod() + " Results:", result.hits()),
                    1);
            return result;
        }
    }

    /**
     * Implement feature score scaling and normalization
     * @param hitsTensor
     * @param hitsLexical
     * @param k
     * @param alpha
     * @param query
     */
    HitGroup rrf(HitGroup hitsTensor, HitGroup hitsLexical, Integer k, Double alpha, Query query) {

        HashMap<String, Double> rrfScores = new HashMap<>();
        HitGroup result = new HitGroup();
        Double reciprocalRank, existingScore, newScore;

        query.trace(String.format("Beginning RRF process. k=%d, alpha=%.2f", k, alpha), 2);

        // Iterate through tensor hits list
        int rank = 1;
        if (alpha > 0.0) {
            query.trace(
                    String.format(
                            "Iterating through tensor result list. Size: %d", hitsTensor.size()),
                    2);

            for (Hit hit : hitsTensor) {
                reciprocalRank = alpha * (1.0 / (rank + k));
                query.trace(
                        String.format(
                                "Tensor hit at rank: %d, hit: %s, Calculated RRF (Tensor): %.7f",
                                rank, hit, reciprocalRank),
                        2);
                // Store hit's score via its URI
                rrfScores.put(hit.getId().toString(), reciprocalRank);
                // Encode raw score for Marqo debugging purposes
                hit.setField("marqo__raw_tensor_score", hit.getRelevance().getScore());
                // Update score to be weighted RR (tensor)
                hit.setRelevance(reciprocalRank);
                result.add(hit);
                rank++;
            }
        }

        // Iterate through lexical hits list
        rank = 1;
        if (alpha < 1.0) {
            query.trace(
                    String.format(
                            "Iterating through lexical result list. Size: %d", hitsLexical.size()),
                    2);

            for (Hit hit : hitsLexical) {

                reciprocalRank = (1.0 - alpha) * (1.0 / (rank + k));
                query.trace(
                        String.format(
                                "Tensor hit at rank: %d, hit: %s, Calculated RRF (lexical): %.7f",
                                rank, hit, reciprocalRank),
                        2);

                // Check if score already exists. If so, add to it.
                existingScore = rrfScores.get(hit.getId().toString());
                if (existingScore == null) {
                    // If the score doesn't exist, add new hit to result list (with rrf score).
                    query.trace("No existing score found! Starting at 0.0.", 2);
                    // Encode raw score for Marqo debugging purposes
                    hit.setField("marqo__raw_lexical_score", hit.getRelevance().getScore());
                    // Update score to be weighted RR (lexical)
                    hit.setRelevance(reciprocalRank);
                    rrfScores.put(hit.getId().toString(), reciprocalRank); // Log score in hashmap
                    result.add(hit);

                } else {
                    // If it does, find that hit in the result list and update it, adding new rrf to
                    // its score.
                    newScore = existingScore + reciprocalRank;
                    rrfScores.put(hit.getId().toString(), newScore);

                    // Update existing hit in result list
                    Hit existingHit = result.get(hit.getId().toString());
                    // Encode raw score (of lexical hit) for Marqo debugging purposes
                    existingHit.setField("marqo__raw_lexical_score", hit.getRelevance().getScore());
                    existingHit.setRelevance(newScore);

                    query.trace(
                            String.format(
                                    "Existing score found for hit: %s. Existing score is: %.7f. New"
                                            + " score is: %.7f",
                                    hit.getId().toString(), existingScore, newScore),
                            2);
                }

                query.trace(
                        String.format("Modified lexical hit at rank: %d. hit: %s", rank, hit), 2);
                rank++;
            }
        }

        // Sort and trim results.
        query.trace(new NamedHitGroup("Combined list (UNSORTED)", result), 2);

        result.sort();
        query.trace(new NamedHitGroup("Combined list (SORTED)", result), 2);

        // Only return top hits (max length)
        int finalLength = Math.max(hitsTensor.size(), hitsLexical.size());
        result.trim(0, finalLength);
        query.trace(new NamedHitGroup("Combined list (TRIMMED)", result), 2);

        return result;
    }
}
