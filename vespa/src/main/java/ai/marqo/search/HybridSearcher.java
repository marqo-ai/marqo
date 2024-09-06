package ai.marqo.search;

import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.net.URI;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.AsyncExecution;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.Tensor.Cell;
import com.yahoo.tensor.TensorAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This searcher takes the YQL for both a lexical and tensor search from the
 * query, Creates 2 clone queries
 *
 */
@Before("ExternalYql")
@Provides("HybridReRanking")
public class HybridSearcher extends Searcher {

    Logger logger = LoggerFactory.getLogger(HybridSearcher.class);

    private static String QUERY_INPUT_FIELDS_TO_RANK = "marqo__fields_to_rank";
    private static String MARQO_SEARCH_METHOD_LEXICAL = "lexical";
    private static String MARQO_SEARCH_METHOD_TENSOR = "tensor";
    private List<String> STANDARD_SEARCH_TYPES = new ArrayList<>();

    @Override
    public Result search(Query query, Execution execution) {
        // All query parameters starting with 'marqo__' are custom for Marqo hybrid search.

        // Retrieval methods: disjunction, tensor, lexical
        // Ranking methods: rrf, normalize_linear, tensor, lexical
        STANDARD_SEARCH_TYPES.add(MARQO_SEARCH_METHOD_LEXICAL);
        STANDARD_SEARCH_TYPES.add(MARQO_SEARCH_METHOD_TENSOR);

        boolean verbose = query.properties().getBoolean("marqo__hybrid.verbose", false);

        logIfVerbose("Starting Hybrid Search script.", verbose);

        String retrievalMethod = query.properties().getString("marqo__hybrid.retrievalMethod", "");
        String rankingMethod = query.properties().getString("marqo__hybrid.rankingMethod", "");

        Integer rrf_k = query.properties().getInteger("marqo__hybrid.rrf_k", 60);
        Double alpha = query.properties().getDouble("marqo__hybrid.alpha", 0.5);
        Integer timeout = query.properties().getInteger("timeout", 1000);

        // Log fetched variables
        logIfVerbose(String.format("Retrieval method found: %s", retrievalMethod), verbose);
        logIfVerbose(String.format("Ranking method found: %s", rankingMethod), verbose);
        logIfVerbose(String.format("alpha found: %.2f", alpha), verbose);
        logIfVerbose(String.format("RRF k found: %d", rrf_k), verbose);
        logIfVerbose(String.format("Timeout int found: %d", timeout), verbose);

        logIfVerbose(String.format("Base Query is: "), verbose);
        logIfVerbose(query.toDetailString(), verbose);

        if (retrievalMethod.equals("disjunction")) {
            Result resultLexical, resultTensor;
            Query queryLexical =
                    createSubQuery(
                            query,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            verbose);
            Query queryTensor =
                    createSubQuery(
                            query, MARQO_SEARCH_METHOD_TENSOR, MARQO_SEARCH_METHOD_TENSOR, verbose);

            // Execute both searches async
            AsyncExecution asyncExecutionLexical = new AsyncExecution(execution);
            Future<Result> futureLexical = asyncExecutionLexical.search(queryLexical);
            AsyncExecution asyncExecutionTensor = new AsyncExecution(execution);
            Future<Result> futureTensor = asyncExecutionTensor.search(queryTensor);
            try {
                resultLexical = futureLexical.get(timeout, TimeUnit.MILLISECONDS);
                resultTensor = futureTensor.get(timeout, TimeUnit.MILLISECONDS);
            } catch (TimeoutException | InterruptedException | ExecutionException e) {
                throw new RuntimeException(
                        String.format(
                                        "Hybrid search disjunction timeout error. Current timeout:"
                                                + " %d. ",
                                        timeout)
                                + e.toString());
            }

            logIfVerbose(
                    "LEXICAL RESULTS: "
                            + resultLexical.toString()
                            + " || TENSOR RESULTS: "
                            + resultTensor.toString(),
                    verbose);

            // Execute fusion ranking on 2 results.
            if (rankingMethod.equals("rrf")) {
                HitGroup fusedHitList =
                        rrf(resultTensor.hits(), resultLexical.hits(), rrf_k, alpha, verbose);
                logIfVerbose("RRF Fused Hit Group", verbose);
                logHitGroup(fusedHitList, verbose);
                return new Result(query, fusedHitList);
            } else if (rankingMethod.equals("copeland")) {
                HitGroup fusedHitList =
                        copelandFusion(resultTensor.hits(), resultLexical.hits(), verbose);
                logIfVerbose("Copeland Fused Hit Group", verbose);
                logHitGroup(fusedHitList, verbose);
                return new Result(query, fusedHitList);
            } else {
                throw new RuntimeException(
                        "For retrievalMethod='disjunction', rankingMethod must be 'rrf' or"
                                + " 'copeland'.");
            }

        } else if (STANDARD_SEARCH_TYPES.contains(retrievalMethod)) {
            if (STANDARD_SEARCH_TYPES.contains(rankingMethod)) {
                Query combinedQuery =
                        createSubQuery(query, retrievalMethod, rankingMethod, verbose);
                Result result = execution.search(combinedQuery);
                logIfVerbose("Results: ", verbose);
                logHitGroup(result.hits(), verbose);
                return result;
            } else {
                throw new RuntimeException(
                        "If retrievalMethod is 'lexical' or 'tensor', rankingMethod can only be"
                                + " 'lexical', or 'tensor'.");
            }
        } else {
            throw new RuntimeException(
                    "retrievalMethod can only be 'disjunction', 'lexical', or 'tensor'.");
        }
    }

    /**
     * Implement feature score scaling and normalization
     *
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
                        verbose); // TODO: For easier debugging, expose marqo__id
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
                        verbose); // TODO: For easier debugging, expose marqo__id
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
                    // its score.
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
     * Implement copeland based fusion as proposed here https://dl.acm.org/doi/pdf/10.1145/3626772.3657912
     *
     * @param hitsTensor
     * @param hitsLexical
     * @param verbose
     */
    HitGroup copelandFusion(HitGroup hitsTensor, HitGroup hitsLexical, boolean verbose) {
        int finalLength = Math.max(hitsTensor.size(), hitsLexical.size());

        // Combine hits from both lists and update the raw score attributes
        Map<URI, Hit> combinedHitsMap = new LinkedHashMap<>();
        for (Hit hit : hitsTensor) {
            hit.setField("marqo__raw_tensor_score", hit.getRelevance().getScore());
            combinedHitsMap.put(hit.getId(), hit);
        }
        for (Hit hit : hitsLexical) {
            if (combinedHitsMap.containsKey(hit.getId())) {
                Hit tensorHit = combinedHitsMap.get(hit.getId());
                tensorHit.setField("marqo__raw_lexical_score", hit.getRelevance().getScore());
            } else {
                hit.setField("marqo__raw_lexical_score", hit.getRelevance().getScore());
                combinedHitsMap.put(hit.getId(), hit);
            }
        }

        List<Hit> uniqueHits = new ArrayList<>(combinedHitsMap.values());

        // Initialize rank maps
        HashMap<URI, Integer> idToRankTensor = new HashMap<>();
        for (int i = 0; i < hitsTensor.size(); i++) {
            idToRankTensor.put(hitsTensor.get(i).getId(), i);
        }

        HashMap<URI, Integer> idToRankLexical = new HashMap<>();
        for (int i = 0; i < hitsLexical.size(); i++) {
            idToRankLexical.put(hitsLexical.get(i).getId(), i);
        }

        // Calculate Copeland scores
        HashMap<URI, Integer> copelandScores = new HashMap<>();
        for (Hit hit1 : uniqueHits) {
            URI hitId1 = hit1.getId();
            int wins = 0;
            int losses = 0;
            for (Hit hit2 : uniqueHits) {
                if (hit1.equals(hit2)) continue;
                URI hitId2 = hit2.getId();

                int rank1Tensor = idToRankTensor.getOrDefault(hitId1, Integer.MAX_VALUE);
                int rank1Lexical = idToRankLexical.getOrDefault(hitId1, Integer.MAX_VALUE);
                int rank2Tensor = idToRankTensor.getOrDefault(hitId2, Integer.MAX_VALUE);
                int rank2Lexical = idToRankLexical.getOrDefault(hitId2, Integer.MAX_VALUE);

                int winsForHit1 = 0;
                int lossesForHit1 = 0;
                if (rank1Tensor < rank2Tensor) winsForHit1++;
                if (rank1Tensor > rank2Tensor) lossesForHit1++;
                if (rank1Lexical < rank2Lexical) winsForHit1++;
                if (rank1Lexical > rank2Lexical) lossesForHit1++;

                if (winsForHit1 > lossesForHit1) {
                    wins++;
                } else if (lossesForHit1 > winsForHit1) {
                    losses++;
                }
            }
            copelandScores.put(hitId1, wins - losses);
        }

        // set relevance scores and sort hits
        for (Hit hit : uniqueHits) {
            URI hitId = hit.getId();
            int score = copelandScores.getOrDefault(hitId, 0);
            hit.setRelevance(score);
        }

        // create and populate the final results
        HitGroup result = new HitGroup();
        result.addAll(uniqueHits);

        logIfVerbose("Combined list (UNSORTED)", verbose);
        logHitGroup(result, verbose);

        // sort and trim to final length
        result.sort();
        logIfVerbose("Combined list (SORTED)", verbose);
        logHitGroup(result, verbose);

        result.trim(0, finalLength);
        logIfVerbose("Combined list (TRIMMED)", verbose);
        logHitGroup(result, verbose);

        return result;
    }

    /**
     * Extracts mapped Tensor Address from cell then adds it as key to rank
     * features, with cell value as the value.
     *
     * @param cell
     * @param query
     * @param verbose
     */
    void addFieldToRankFeatures(Cell cell, Query query, boolean verbose) {
        TensorAddress cellKey = cell.getKey();
        String queryInputString;
        int dimensions = cellKey.size();
        for (int i = 0; i < dimensions; i++) {
            queryInputString = addQueryWrapper(cellKey.label(i));
            query.getRanking().getFeatures().put(queryInputString, cell.getValue());
            logIfVerbose(
                    String.format(
                            "Setting Rank Feature %s to %s", queryInputString, cell.getValue()),
                    verbose);
        }
    }

    /**
     * Creates custom sub-query from the original query. Clone original query,
     * Update the following: 'yql' (based on RETRIEVAL method) 'ranking.profile'
     * (based on RANKING method) 'ranking.features' fields to search (based on
     * ??? method) score modifiers (based on RANKING method)
     *
     * @param query
     * @param retrievalMethod
     * @param rankingMethod
     * @param verbose
     */
    Query createSubQuery(
            Query query, String retrievalMethod, String rankingMethod, boolean verbose) {
        logIfVerbose(
                String.format(
                        "Creating subquery with retrieval: %s, ranking: %s",
                        retrievalMethod, rankingMethod),
                verbose);

        // Extract relevant properties
        // YQL uses RETRIEVAL method
        String yqlNew = query.properties().getString("marqo__yql." + retrievalMethod, "");
        // Rank Profile uses RETRIEVAL + RANKING method
        String rankProfileNew =
                query.properties()
                        .getString("marqo__ranking." + retrievalMethod + "." + rankingMethod, "");

        // Log fetched properties
        logIfVerbose(String.format("YQL %s found: %s", retrievalMethod, yqlNew), verbose);
        logIfVerbose(
                String.format(
                        "Rank Profile %s.%s found: %s",
                        retrievalMethod, rankingMethod, rankProfileNew),
                verbose);

        // Create New Subquery
        Query queryNew = query.clone();
        queryNew.properties().set("yql", yqlNew);

        // Set fields to rank
        // Extract using RETRIEVAL method (first-phase)
        String featureNameFieldsToRank =
                addQueryWrapper(QUERY_INPUT_FIELDS_TO_RANK + "_" + retrievalMethod);
        logIfVerbose(
                "Extracting using fields to rank from RETRIEVAL method: " + featureNameFieldsToRank,
                verbose);
        Tensor fieldsToRank = extractTensorRankFeature(query, featureNameFieldsToRank);
        Iterator<Cell> cells = fieldsToRank.cellIterator();
        cells.forEachRemaining((cell) -> addFieldToRankFeatures(cell, queryNew, verbose));

        // Extract using RANKING method (second-phase)
        if (!(retrievalMethod.equals(rankingMethod))) {
            featureNameFieldsToRank =
                    addQueryWrapper(QUERY_INPUT_FIELDS_TO_RANK + "_" + rankingMethod);
            logIfVerbose(
                    "Extracting using fields to rank from RANKING method: "
                            + featureNameFieldsToRank,
                    verbose);
            fieldsToRank = extractTensorRankFeature(query, featureNameFieldsToRank);
            cells = fieldsToRank.cellIterator();
            cells.forEachRemaining((cell) -> addFieldToRankFeatures(cell, queryNew, verbose));
        }

        // Set rank profile (using RANKING method)
        queryNew.getRanking().setProfile(rankProfileNew);

        // Log tensor query final state
        logIfVerbose("FINAL QUERY: ", verbose);
        logIfVerbose(queryNew.toDetailString(), verbose);
        logIfVerbose(queryNew.getModel().getQueryString(), verbose);
        logIfVerbose(queryNew.properties().getString("yql", ""), verbose);
        logIfVerbose(queryNew.getRanking().getFeatures().toString(), verbose);
        logIfVerbose(
                String.format("Rank Profile: %s", queryNew.getRanking().getProfile()), verbose);

        return queryNew;
    }

    /**
     * Print human-readable list of hits with relevances.
     *
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
     *
     * @param str
     * @param verbose
     */
    void logIfVerbose(String str, boolean verbose) {
        if (verbose) {
            logger.info(str);
        }
    }

    /**
     * Extract a tensor rank feature, throwing an error if it does not exist
     *
     * @param query
     * @param featureName
     */
    Tensor extractTensorRankFeature(Query query, String featureName) {
        Optional<Tensor> optionalTensor = query.getRanking().getFeatures().getTensor(featureName);
        Tensor resultTensor;

        if (optionalTensor.isPresent()) {
            resultTensor = optionalTensor.get();
        } else {
            throw new RuntimeException("Rank Feature: " + featureName + " not found in query!");
        }

        return resultTensor;
    }

    /**
     * Enclose string in query()
     *
     * @param str
     */
    String addQueryWrapper(String str) {
        return "query(" + str + ")";
    }
}
