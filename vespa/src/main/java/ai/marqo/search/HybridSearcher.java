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
        boolean verbose = query.properties().getBoolean("marqo__hybrid.verbose", false);
        logIfVerbose("Starting Hybrid Search script.", verbose);

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
                            verbose);
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
     * @param verbose
     */
    HitGroup rrf(
            HitGroup hitsTensor, HitGroup hitsLexical, Integer k, Double alpha, boolean verbose) {

        HashMap<String, Double> rrfScores = new HashMap<>();
        HitGroup result = new HitGroup();
        Double reciprocalRank, existingScore, newScore;

        logIfVerbose("Beginning RRF process.", verbose);
        logIfVerbose("Beginning (empty) result state: ", verbose);
        logHitGroup(result, verbose);

        logIfVerbose(String.format("alpha is %.2f", alpha), verbose);
        logIfVerbose(String.format("k is %d", k), verbose);

        // Iterate through tensor hits list

        int rank = 1;
        if (alpha > 0.0) {
            logIfVerbose(
                    String.format(
                            "Iterating through tensor result list. Size: %d", hitsTensor.size()),
                    verbose);

            for (Hit hit : hitsTensor) {
                logIfVerbose(
                        String.format("Tensor hit at rank: %d", rank),
                        verbose); // TODO: Expose marqo__id
                logIfVerbose(hit.toString(), verbose);

                reciprocalRank = alpha * (1.0 / (rank + k));
                rrfScores.put(
                        hit.getId().toString(), reciprocalRank); // Store hit's score via its URI
                hit.setField(
                        "marqo__raw_tensor_score",
                        hit.getRelevance()
                                .getScore()); // Encode raw score for Marqo debugging purposes
                hit.setRelevance(reciprocalRank); // Update score to be weighted RR (tensor)
                result.add(hit);
                logIfVerbose(String.format("Set relevance to: %.7f", reciprocalRank), verbose);
                logIfVerbose("Current result state: ", verbose);
                logHitGroup(result, verbose);
                rank++;
            }
        }

        // Iterate through lexical hits list
        rank = 1;
        if (alpha < 1.0) {
            logIfVerbose(
                    String.format(
                            "Iterating through lexical result list. Size: %d", hitsLexical.size()),
                    verbose);

            for (Hit hit : hitsLexical) {
                logIfVerbose(
                        String.format("Lexical hit at rank: %d", rank),
                        verbose); // TODO: Expose marqo__id
                logIfVerbose(hit.toString(), verbose);

                reciprocalRank = (1.0 - alpha) * (1.0 / (rank + k));
                logIfVerbose(
                        String.format("Calculated RRF (lexical) is: %.7f", reciprocalRank),
                        verbose);

                // Check if score already exists. If so, add to it.
                existingScore = rrfScores.get(hit.getId().toString());
                if (existingScore == null) {
                    // If the score doesn't exist, add new hit to result list (with rrf score).
                    logIfVerbose("No existing score found! Starting at 0.0.", verbose);
                    hit.setField(
                            "marqo__raw_lexical_score",
                            hit.getRelevance()
                                    .getScore()); // Encode raw score for Marqo debugging purposes
                    hit.setRelevance(reciprocalRank); // Update score to be weighted RR (lexical)
                    rrfScores.put(hit.getId().toString(), reciprocalRank); // Log score in hashmap
                    result.add(hit);

                } else {
                    // If it does, find that hit in the result list and update it, adding new rrf to
                    // its
                    // score.
                    newScore = existingScore + reciprocalRank;
                    rrfScores.put(hit.getId().toString(), newScore);

                    // Update existing hit in result list
                    Hit existingHit = result.get(hit.getId().toString());
                    existingHit.setField(
                            "marqo__raw_lexical_score",
                            hit.getRelevance()
                                    .getScore()); // Encode raw score (of lexical hit) for Marqo
                    // debugging purposes
                    existingHit.setRelevance(newScore);

                    logIfVerbose(
                            String.format(
                                    "Existing score found for hit: %s.", hit.getId().toString()),
                            verbose);
                    logIfVerbose(String.format("Existing score is: %.7f", existingScore), verbose);
                    logIfVerbose(String.format("New score is: %.7f", newScore), verbose);
                }

                logIfVerbose(String.format("Modified lexical hit at rank: %d", rank), verbose);
                logIfVerbose(hit.toString(), verbose);

                rank++;

                logIfVerbose("Current result state: ", verbose);
                logHitGroup(result, verbose);
            }
        }

        // Sort and trim results.
        logIfVerbose("Combined list (UNSORTED)", verbose);
        logHitGroup(result, verbose);

        result.sort();
        logIfVerbose("Combined list (SORTED)", verbose);
        logHitGroup(result, verbose);

        // Only return top hits (max length)
        Integer finalLength = Math.max(hitsTensor.size(), hitsLexical.size());
        result.trim(0, finalLength);
        logIfVerbose("Combined list (TRIMMED)", verbose);
        logHitGroup(result, verbose);

        return result;
    }

    /**
     * Print human-readable list of hits with relevances.
     * @param hits
     * @param verbose
     */
    public void logHitGroup(HitGroup hits, boolean verbose) {
        if (verbose) {
            logger.info(String.format("Hit Group has size: %s", hits.size()));
            logger.info("=======================");
            int idx = 0;
            for (Hit hit : hits) {
                logger.info(
                        String.format(
                                "{IDX: %s, HIT ID: %s, RELEVANCE: %.7f}",
                                idx, hit.getId().toString(), hit.getRelevance().getScore()));
                idx++;
            }
            logger.info("=======================");
        }
    }

    /**
     * Log to info if the verbose flag is turned on.
     * @param str
     * @param verbose
     */
    void logIfVerbose(String str, boolean verbose) {
        if (verbose) {
            logger.info(str);
        }
    }
}
