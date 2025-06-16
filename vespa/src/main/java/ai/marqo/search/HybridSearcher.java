package ai.marqo.search;

import com.sun.jdi.InternalException;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.FeatureData;
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
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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

    private static String QUERY_INPUT_FIELDS_TO_RANK = "marqo__fields_to_rank";
    private static String QUERY_INPUT_MULT_WEIGHTS_GLOBAL = "marqo__mult_weights_global";
    private static String QUERY_INPUT_ADD_WEIGHTS_GLOBAL = "marqo__add_weights_global";
    private static String MARQO_SEARCH_METHOD_LEXICAL = "lexical";
    private static String MARQO_SEARCH_METHOD_TENSOR = "tensor";
    private List<String> STANDARD_SEARCH_TYPES = new ArrayList<>();

    // Compile the regex pattern once and store it as a static final variable
    private static final Pattern PATTERN = Pattern.compile("^index\\:[^\\s\\/]+\\/\\d+\\/(.+)$");

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
        Integer rerankDepthGlobal =
                query.properties().getInteger("marqo__hybrid.rerankDepthGlobal", null);
        Integer limit = query.properties().getInteger("hits", null);
        Integer offset = query.properties().getInteger("offset", 0);
        Integer timeout = query.properties().getInteger("timeout", 1000);

        // Log fetched variables
        logIfVerbose(String.format("Retrieval method found: %s", retrievalMethod), verbose);
        logIfVerbose(String.format("Ranking method found: %s", rankingMethod), verbose);
        logIfVerbose(String.format("alpha found: %.2f", alpha), verbose);
        logIfVerbose(String.format("RRF k found: %d", rrf_k), verbose);
        logIfVerbose(String.format("Rerank count global found: %d", rerankDepthGlobal), verbose);
        logIfVerbose(String.format("Limit found: %d", limit), verbose);
        logIfVerbose(String.format("Offset found: %d", offset), verbose);
        logIfVerbose(String.format("Timeout int found: %d", timeout), verbose);

        logIfVerbose(String.format("Base Query is: "), verbose);
        logIfVerbose(query.toDetailString(), verbose);

        // Validation for limit
        if (limit == null) {
            throw new RuntimeException("Query limit cannot be null.");
        }

        // --- Begin facets subquery handling ---
        // Check for custom facets YQL properties - expect array of strings
        String[] facetsYqlQueries =
                query.properties()
                        .getString("marqo__yql.facets", "")
                        .split("\n---MARQO-YQL-QUERY-DELIMITER---\n");
        List<Future<Result>> futureFacets = new ArrayList<>();

        for (String facetsYql : facetsYqlQueries) {
            if (!facetsYql.isEmpty()) {
                // Create a subquery for each facet query
                Query queryFacets =
                        createSubQuery(
                                query,
                                MARQO_SEARCH_METHOD_LEXICAL,
                                MARQO_SEARCH_METHOD_LEXICAL,
                                verbose,
                                facetsYql);
                AsyncExecution asyncExecutionFacets = new AsyncExecution(execution);
                futureFacets.add(asyncExecutionFacets.search(queryFacets));
            }
        }
        // --- End facets subquery handling ---

        HitGroup hitsForPostProcessing;
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

            // Execute both lexical and tensor queries asynchronously.
            AsyncExecution asyncExecutionLexical = new AsyncExecution(execution);
            Future<Result> futureLexical = asyncExecutionLexical.search(queryLexical);
            AsyncExecution asyncExecutionTensor = new AsyncExecution(execution);
            Future<Result> futureTensor = asyncExecutionTensor.search(queryTensor);
            try {
                resultLexical = futureLexical.get(timeout, TimeUnit.MILLISECONDS);
                resultTensor = futureTensor.get(timeout, TimeUnit.MILLISECONDS);
            } catch (TimeoutException | InterruptedException | ExecutionException e) {
                throw new RuntimeException(
                        "Hybrid search disjunction timeout error. Current timeout: "
                                + timeout
                                + ". "
                                + e.toString());
            }

            // Collect errors from lexical and tensor results.
            HitGroup combinedErrors = collectErrorsFromResults(resultLexical, resultTensor);
            if (combinedErrors.getError() != null) {
                return new Result(query, combinedErrors);
            }

            logIfVerbose(
                    "LEXICAL RESULTS: "
                            + resultLexical.toString()
                            + " || TENSOR RESULTS: "
                            + resultTensor.toString(),
                    verbose);

            // Execute fusion ranking on the two result sets.
            if (rankingMethod.equals("rrf")) {
                hitsForPostProcessing =
                        rrf(resultTensor.hits(), resultLexical.hits(), rrf_k, alpha, verbose);
            } else {
                throw new RuntimeException(
                        "For retrievalMethod='disjunction', rankingMethod must be 'rrf'.");
            }

        } else if (STANDARD_SEARCH_TYPES.contains(retrievalMethod)) {
            if (STANDARD_SEARCH_TYPES.contains(rankingMethod)) {
                Query combinedQuery =
                        createSubQuery(query, retrievalMethod, rankingMethod, verbose);
                Result result = execution.search(combinedQuery);
                hitsForPostProcessing = result.hits();
                logIfVerbose("Unprocessed results: ", verbose);
                logHitGroup(hitsForPostProcessing, verbose);
            } else {
                throw new RuntimeException(
                        "If retrievalMethod is 'lexical' or 'tensor', rankingMethod can only be"
                                + " 'lexical' or 'tensor'.");
            }
        } else {
            throw new RuntimeException(
                    "retrievalMethod can only be 'disjunction', 'lexical', or 'tensor'.");
        }

        // Post-process the main hits result list.
        HitGroup processedHits =
                postProcessResults(
                        hitsForPostProcessing, query, rerankDepthGlobal, limit, offset, verbose);

        // --- Attach facets results if available ---
        if (!futureFacets.isEmpty()) {
            try {
                long startTime = System.currentTimeMillis();
                int facetCounter = 0;
                for (Future<Result> futureFacet : futureFacets) {
                    Result facetsResult = futureFacet.get(timeout, TimeUnit.MILLISECONDS);
                    if (facetsResult != null && facetsResult.hits() != null) {
                        // Ensure unique IDs for each facet group by adding counter
                        int hitCounter = 0;
                        for (Hit hit : facetsResult.hits().asList()) {
                            String originalId = hit.getId().toString();
                            if (originalId.startsWith("group:")) {
                                hit.setId("group:facet:" + facetCounter + ":" + hitCounter);
                                hitCounter++;
                            }
                        }
                        // Add facets as children to the processed hits
                        processedHits.addAll(facetsResult.hits().asList());
                        facetCounter++;
                    }
                }
                long facetsTime = System.currentTimeMillis() - startTime;
                logIfVerbose(
                        String.format(
                                "Took %.3fms to process and attach %d facet queries",
                                facetsTime / 1000.0, futureFacets.size()),
                        verbose);
            } catch (TimeoutException | InterruptedException | ExecutionException e) {
                throw new RuntimeException(
                        "Hybrid search facets timeout error. Current timeout: "
                                + timeout
                                + ". "
                                + e.toString());
            }
        }
        // --- End facets attachment ---

        return new Result(query, processedHits);
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
        HashMap<String, String> docIdsToHitIds = new HashMap<>();
        HitGroup result = new HitGroup();
        Double reciprocalRank, existingScore, newScore;
        String extractedDocId;

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

                extractedDocId = extractDocIdFromHitId(hit.getId().toString());
                reciprocalRank = alpha * (1.0 / (rank + k));
                // Map hit's score to its shortened doc ID
                rrfScores.put(extractedDocId, reciprocalRank);
                // Map hit's full URI to its shortened doc ID
                docIdsToHitIds.put(extractedDocId, hit.getId().toString());
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
                extractedDocId = extractDocIdFromHitId(hit.getId().toString());
                existingScore = rrfScores.get(extractedDocId);
                if (existingScore == null) {
                    // If the score doesn't exist, add new hit to result list (with rrf score).
                    logIfVerbose("No existing score found! Starting at 0.0.", verbose);
                    hit.setField(
                            "marqo__raw_lexical_score",
                            hit.getRelevance()
                                    .getScore()); // Encode raw score for Marqo debugging purposes
                    hit.setRelevance(reciprocalRank); // Update score to be weighted RR (lexical)
                    // Map hit's score to its shortened doc ID
                    rrfScores.put(extractedDocId, reciprocalRank);
                    // Map hit's full URI to its shortened doc ID
                    docIdsToHitIds.put(extractedDocId, hit.getId().toString());
                    result.add(hit);

                } else {
                    // If it does, find that hit in the result list and update it, adding new rrf to
                    // its score.
                    newScore = existingScore + reciprocalRank;
                    rrfScores.put(extractedDocId, newScore);

                    // Update existing hit in result list (use map to find the full hit ID)
                    Hit existingHit = result.get(docIdsToHitIds.get(extractedDocId));

                    existingHit.setField(
                            "marqo__raw_lexical_score",
                            hit.getRelevance()
                                    .getScore()); // Encode raw score (of lexical hit) for Marqo
                    // debugging purposes
                    existingHit.setRelevance(newScore);

                    logIfVerbose(
                            String.format(
                                    "Existing score found for hit: %s.",
                                    extractDocIdFromHitId(hit.getId().toString())),
                            verbose);
                    logIfVerbose(String.format("Existing score is: %.7f", existingScore), verbose);
                    logIfVerbose(String.format("New score is: %.7f", newScore), verbose);
                }

                logIfVerbose(String.format("Modified lexical hit at rank: %d", rank), verbose);
                logIfVerbose(hit.toString(), verbose);

                rank++;
            }
        }

        return result;
    }

    HitGroup collectErrorsFromResults(Result resultLexical, Result resultTensor) {
        // Return errors if either result list has an error. Make sure all errors are returned.
        HitGroup combinedErrors = new HitGroup();
        logIfVerbose(
                String.format("Tensor Errors found: %s", resultTensor.hits().getError()), true);
        logIfVerbose(
                String.format("Lexical Errors found: %s", resultLexical.hits().getError()), true);
        combinedErrors.addErrorsFrom(resultTensor.hits());
        combinedErrors.addErrorsFrom(resultLexical.hits());
        return combinedErrors;
    }

    /**
     * Post-processes the result list, applying global score modifiers and reranking.
     */
    HitGroup postProcessResults(
            HitGroup hitsForPostProcessing,
            Query query,
            Integer rerankDepthGlobal,
            int limit,
            int offset,
            boolean verbose) {
        // Split original hits into 2 lists: result to rerank and excess hits
        // Excess hits will not be reranked, and will be added back after reranking the other
        // results
        HitGroup resultToRerank = new HitGroup();
        HitGroup excessHits = new HitGroup();

        int idx = 0;
        // If rerank count global is not set, rerank all hits
        if (rerankDepthGlobal == null) {
            rerankDepthGlobal = hitsForPostProcessing.size();
        }
        for (Hit hit : hitsForPostProcessing) {
            if (idx < rerankDepthGlobal) {
                resultToRerank.add(hit);
            } else if (idx < limit) {
                // Total hits to return caps out at limit
                excessHits.add(hit);
            } else {
                // Ignore all hits after limit
                break;
            }
            idx++;
        }

        logIfVerbose("Result list to rerank: ", verbose);
        logHitGroup(resultToRerank, verbose);
        if (excessHits.size() > 0) {
            logIfVerbose("Excess hits (will not be rescored): ", verbose);
            logHitGroup(excessHits, verbose);
        }

        // Apply global score modifiers and rerank
        // Skip whole process if global modifier weight tensors don't exist in query
        Tensor queryMultWeightsGlobal =
                extractTensorRankFeature(query, addQueryWrapper(QUERY_INPUT_MULT_WEIGHTS_GLOBAL));
        Tensor queryAddWeightsGlobal =
                extractTensorRankFeature(query, addQueryWrapper(QUERY_INPUT_ADD_WEIGHTS_GLOBAL));

        if ((queryMultWeightsGlobal != null && !queryMultWeightsGlobal.isEmpty())
                || (queryAddWeightsGlobal != null && !queryAddWeightsGlobal.isEmpty())) {
            logIfVerbose("Applying global score modifiers and reranking.", verbose);
            resultToRerank = applyGlobalScoreModifiers(resultToRerank, verbose);
        } else {
            logIfVerbose("No weights found. Skipping applying global score modifiers.", verbose);
        }

        logIfVerbose("Rescored result list (UNSORTED): ", verbose);
        logHitGroup(resultToRerank, verbose);

        resultToRerank.sort();

        logIfVerbose("Reranked result list (SORTED): ", verbose);
        logHitGroup(resultToRerank, verbose);

        if (limit > rerankDepthGlobal) {
            // Add excess hits to the end of reranked results then sort
            logIfVerbose(
                    String.format(
                            "Adding %d excess hits to the end of reranked results and sorting.",
                            excessHits.size()),
                    verbose);
            resultToRerank.addAll(excessHits.asList());
        }

        // Paginate and/or trim
        // Result list should always have limit length (if possible)
        logIfVerbose(
                String.format("Trimming result list. " + "limit: %d, offset: %d", limit, offset),
                verbose);
        resultToRerank.trim(0, limit);

        logIfVerbose("Final result list (EXCESS HITS ADDED/REMOVED): ", verbose);
        logHitGroup(resultToRerank, verbose);

        return resultToRerank;
    }

    void raiseErrorIfPresent(Result resultLexical, Result resultTensor) {
        // Raise error if either result list has an error. Make sure error messages are combined
        String tensorOrLexicalErrors = "";
        ErrorMessage tensorError = resultTensor.hits().getError();
        if (tensorError != null) {
            tensorOrLexicalErrors += "Error in TENSOR search in RRF: " + tensorError;
        }

        ErrorMessage lexicalError = resultLexical.hits().getError();
        if (lexicalError != null) {
            tensorOrLexicalErrors += "Error in LEXICAL search in RRF: " + lexicalError;
        }

        if (!tensorOrLexicalErrors.isEmpty()) {
            throw new RuntimeException(tensorOrLexicalErrors);
        }
    }

    /**
     * Extracts mapped Tensor Address from cell then adds it as key to rank features, with cell value as the value.
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

    public Query createSubQuery(
            Query query, String retrievalMethod, String rankingMethod, boolean verbose) {
        // Default exactQuery to an empty string (or any default value you prefer)
        return createSubQuery(query, retrievalMethod, rankingMethod, verbose, "");
    }

    /**
     * Creates custom sub-query from the original query.
     * Clone original query, Update the following:
     * 'yql' (based on RETRIEVAL method)
     * 'ranking.profile'    (based on RANKING method)
     * 'ranking.features'
     *      fields to search  (based on ??? method)
     *      score modifiers (based on RANKING method)
     * @param query
     * @param retrievalMethod
     * @param rankingMethod
     * @param verbose
     */
    Query createSubQuery(
            Query query,
            String retrievalMethod,
            String rankingMethod,
            boolean verbose,
            String exactQuery) {
        logIfVerbose(
                String.format(
                        "Creating subquery with retrieval: %s, ranking: %s",
                        retrievalMethod, rankingMethod),
                verbose);

        // Extract relevant properties
        // YQL uses RETRIEVAL method
        String yqlNew;
        if (!exactQuery.isEmpty()) {
            yqlNew = exactQuery;
        } else {
            yqlNew = query.properties().getString("marqo__yql." + retrievalMethod, "");
        }
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
                                idx,
                                extractDocIdFromHitId(hit.getId().toString()),
                                hit.getRelevance().getScore()));
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

    /**
     * Extract a tensor rank feature, returning null if it does not exist
     * @param query
     * @param featureName
     */
    Tensor extractTensorRankFeature(Query query, String featureName) {
        Optional<Tensor> optionalTensor = query.getRanking().getFeatures().getTensor(featureName);
        Tensor resultTensor;
        return optionalTensor.orElse(null);
    }

    /**
     * Enclose string in query()
     * @param str
     */
    String addQueryWrapper(String str) {
        return "query(" + str + ")";
    }

    /*
     * Extracts the document ID from a hit ID (use regex to extract the doc ID from the hit's URI)
     */
    static String extractDocIdFromHitId(String fullPath) {
        // Create a matcher for the input string using the precompiled pattern
        Matcher matcher = PATTERN.matcher(fullPath);

        // Check if the pattern matches and extract the document ID
        if (matcher.find()) {
            return matcher.group(1); // Return the captured group (document ID)
        } else {
            throw new InternalException(
                    "Vespa doc ID could not be extracted from the full hit ID: " + fullPath + ".");
        }
    }

    /**
     * Apply global score modifiers to the hit group. Modifies hit scores, does not add/remove hits.
     * @param hits
     * @param verbose
     */
    HitGroup applyGlobalScoreModifiers(HitGroup hits, boolean verbose) {
        FeatureData hitMatchFeatures;
        Double mult_modifier, add_modifier, original_score, modified_score;
        if (hits.size() == 0) {
            logIfVerbose("No hits to apply score modifiers to. Returning.", verbose);
            return hits;
        }

        for (Hit hit : hits) {
            logIfVerbose("Applying score modifiers to hit: " + hit.getId(), verbose);
            // Extract the mult and add modifiers from match-features
            hitMatchFeatures = (FeatureData) hit.getField("matchfeatures");
            if (hitMatchFeatures != null) {
                mult_modifier = hitMatchFeatures.getDouble("global_mult_modifier");
                add_modifier = hitMatchFeatures.getDouble("global_add_modifier");

                if (mult_modifier != null && add_modifier != null) {
                    // Apply the modifiers to the hit's relevance
                    original_score = hit.getRelevance().getScore();
                    modified_score = original_score * mult_modifier + add_modifier;
                    logIfVerbose(
                            String.format(
                                    "Original score: %.7f, mult modifier: %.5f, add modifier: %.5f,"
                                            + " Modified score: %.7f",
                                    original_score, mult_modifier, add_modifier, modified_score),
                            verbose);
                    hit.setRelevance(modified_score);
                } else {
                    throw new RuntimeException(
                            "Failed to apply global score modifiers. Hit "
                                    + hit.getId()
                                    + " is missing either global_mult_modifier or"
                                    + " global_add_modifier match-feature.");
                }
            } else {
                throw new RuntimeException(
                        "Failed to apply global score modifiers. Hit "
                                + hit.getId()
                                + " is missing matchfeatures.");
            }
        }
        return hits;
    }
}
