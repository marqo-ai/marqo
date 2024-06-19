package ai.marqo.search;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.component.chain.dependencies.After;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.AsyncExecution;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.Tensor.Cell;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.net.URI;
import java.util.HashMap;
import java.util.Optional;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.lang.InterruptedException;
import java.util.concurrent.ExecutionException;
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

    // Logger logger = LogManager.getLogger(HybridSearcher.class);
    Logger logger = LoggerFactory.getLogger(HybridSearcher.class);

    private static String MATCH_FEATURES_FIELD = "matchfeatures";
    private static String QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS = "marqo__mult_weights";
    private static String QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL = "marqo__mult_weights_lexical";
    private static String QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR = "marqo__mult_weights_tensor";
    private static String QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS = "marqo__add_weights";
    private static String QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL = "marqo__add_weights_lexical";
    private static String QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR = "marqo__add_weights_tensor";
    private static String QUERY_INPUT_FIELDS_TO_SEARCH = "marqo__fields_to_search";
    private List<String> STANDARD_SEARCH_TYPES = new ArrayList<>();

    @Override
    public Result search(Query query, Execution execution) {
        // Retrieval methods: disjunction, tensor, lexical
        // Ranking methods: rrf, normalize_linear, tensor, lexical
        STANDARD_SEARCH_TYPES.add("lexical");
        STANDARD_SEARCH_TYPES.add("tensor");
        
        logger.info("Starting Hybrid Search script.");

        String retrieval_method = query.properties().
                getString("hybrid.retrievalMethod", "");
        String ranking_method = query.properties().
                getString("hybrid.rankingMethod", "");
        
        Integer rrf_k = query.properties().getInteger("hybrid.rrf_k", 60);
        Double alpha = query.properties().getDouble("hybrid.alpha", 0.5);
        
        // TODO: Parse this into an int
        String timeout_string = query.properties().getString("timeout", "1000ms");

        // Log fetched variables
        logger.info(String.format("Retrieval method found: %s", retrieval_method));
        logger.info(String.format("Ranking method found: %s", ranking_method));
        logger.info(String.format("alpha found: %.2f", alpha));
        logger.info(String.format("RRF k found: %d", rrf_k));

        logger.info(String.format("Base Query is: "));
        logger.info(query.toDetailString());
        
        if (retrieval_method.equals("disjunction")) {
            Result result_lexical, result_tensor;
            Query query_lexical = create_sub_query(query, "lexical", "lexical");
            Query query_tensor = create_sub_query(query, "tensor", "tensor");
            
            // Execute both searches async
            AsyncExecution async_execution_lexical = new AsyncExecution(execution);
            Future<Result> future_lexical = async_execution_lexical.search(query_lexical);
            AsyncExecution async_execution_tensor = new AsyncExecution(execution);
            Future<Result> future_tensor = async_execution_tensor.search(query_tensor);
            int timeout = 1000 * 1; // TODO: Change this to input.query(timeout)
            try {
                result_lexical = future_lexical.get(timeout, TimeUnit.MILLISECONDS);
                result_tensor = future_tensor.get(timeout, TimeUnit.MILLISECONDS);
            } catch(TimeoutException | InterruptedException | ExecutionException e) {
                // TODO: Handle timeout better
                throw new RuntimeException(e.toString());
            }

            logger.info("LEXICAL RESULTS: " + result_lexical.toString());
            logger.info("TENSOR RESULTS: " + result_tensor.toString());

            // Execute fusion ranking on 2 results.
            if (ranking_method.equals("rrf")) {
                HitGroup fused_hit_list = rrf(result_tensor.hits(), result_lexical.hits(), rrf_k, alpha);
                logger.info("RRF Fused Hit Group");
                printHitGroup(fused_hit_list);
                return new Result(query, fused_hit_list);
            } else {
                throw new RuntimeException(String.format("For retrieval_method='disjunction', ranking_method must be 'rrf'."));
            }
            
        } else if (STANDARD_SEARCH_TYPES.contains(retrieval_method)){
            if (STANDARD_SEARCH_TYPES.contains(ranking_method)){
                Query combined_query = create_sub_query(query, retrieval_method, ranking_method);
                Result result = execution.search(combined_query);
                logger.info("Results: ");
                printHitGroup(result.hits());
                return execution.search(combined_query);
            } else {
                throw new RuntimeException("If retrieval_method is 'lexical' or 'tensor', ranking_method can only be 'lexical', or 'tensor'.");
            }
        } else {
            throw new RuntimeException("retrieval_method can only be 'disjunction', 'lexical', or 'tensor'.");
        }
    }

    /**
     * Implement feature score scaling and normalization
     * @param hits_tensor
     * @param hits_lexical
     * @param features
     */
    HitGroup rrf(HitGroup hits_tensor, HitGroup hits_lexical, Integer k, Double alpha) {

        HashMap<String, Double> rrf_scores = new HashMap<>();
        HitGroup result = new HitGroup();
        Double reciprocal_rank, existing_score, new_score;

        logger.info("Beginning RRF process.");
        logger.info("Beginning (empty) result state: ");
        printHitGroup(result);

        logger.info(String.format("alpha is %.2f", alpha));
        logger.info(String.format("k is %d", k));

        // Iterate through tensor hits list
        
        int rank = 1;
        if (alpha > 0.0) {
            logger.info(String.format("Iterating through tensor result list. Size: %d", hits_tensor.size()));

            for (Hit hit : hits_tensor) {
                reciprocal_rank = alpha * (1.0 / (rank + k));
                rrf_scores.put(hit.getId().toString(), reciprocal_rank);   // Store hit's score via its URI
                hit.setRelevance(reciprocal_rank);                 // Update score to be weighted RR (tensor)
                result.add(hit);
                logger.info(String.format("Set relevance to: %.7f", reciprocal_rank));
                logger.info(String.format("Modified tensor hit at rank: %d", rank));
                logger.info(hit.toString());

                logger.info("Current result state: ");
                printHitGroup(result);
                rank++;
            }
        }

        // Iterate through lexical hits list
        rank = 1;
        if (alpha < 1.0){
            logger.info(String.format("Iterating through lexical result list. Size: %d", hits_lexical.size()));

            for (Hit hit : hits_lexical) {
                reciprocal_rank = (1.0-alpha) * (1.0 / (rank + k));
                logger.info(String.format("Calculated RRF (lexical) is: %.7f", reciprocal_rank));

                // Check if score already exists. If so, add to it.
                existing_score = rrf_scores.get(hit.getId().toString());
                if (existing_score == null){
                    // If the score doesn't exist, add new hit to result list (with rrf score).
                    logger.info("No existing score found! Starting at 0.0.");
                    hit.setRelevance(reciprocal_rank);      // Update score to be weighted RR (lexical)
                    rrf_scores.put(hit.getId().toString(), reciprocal_rank);    // Log score in hashmap
                    result.add(hit);

                    logger.info(String.format("Modified lexical hit at rank: %d", rank));
                    logger.info(hit.toString());

                } else {
                    // If it does, find that hit in the result list and update it, adding new rrf to its score.
                    new_score = existing_score + reciprocal_rank;
                    rrf_scores.put(hit.getId().toString(), new_score);

                    // Update existing hit in result list
                    result.get(hit.getId().toString()).setRelevance(new_score); 

                    logger.info(String.format("Existing score found for hit: %s.", hit.getId().toString()));
                    logger.info(String.format("Existing score is: %.7f", existing_score));
                    logger.info(String.format("New score is: %.7f", new_score));
                }

                logger.info(String.format("Modified lexical hit at rank: %d", rank));
                logger.info(hit.toString());

                rank++;

                logger.info("Current result state: ");
                printHitGroup(result);
            }
        }

        // Sort and trim results.
        logger.info("Combined list (UNSORTED)");
        printHitGroup(result);

        result.sort();
        logger.info("Combined list (SORTED)");
        printHitGroup(result);

        // Only return top hits (max length)
        Integer final_length = Math.max(hits_tensor.size(), hits_lexical.size());
        result.trim(0, final_length);
        logger.info("Combined list (TRIMMED)");
        printHitGroup(result);

        return result;
    }

    /**
     * Extracts mapped Tensor Address from cell then adds it as key to rank features, with cell value as the value.
     * @param cell
     * @param query
     */
    void add_field_to_rank_features(Cell cell, Query query){
        TensorAddress cell_key = cell.getKey();
        int dimensions = cell_key.size();
        for (int i = 0; i < dimensions; i++){
            String query_input_string = add_query_wrapper(cell_key.label(i));
            logger.info(String.format("Setting Rank Feature %s to %s", query_input_string, cell.getValue()));
            query.getRanking().getFeatures().put(query_input_string, cell.getValue());
        }
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
     */
    Query create_sub_query(Query query, String retrieval_method, String ranking_method){
        logger.info(String.format("Creating subquery with retrieval: %s, ranking: %s", retrieval_method, ranking_method));

        // Extract relevant properties
        // YQL uses RETRIEVAL method
        String yql_new = query.properties().
                getString("yql." + retrieval_method, "");
        // Rank Profile uses RANKING method
        String rank_profile_new = query.properties().
                getString("ranking." + ranking_method, "");
        String rank_profile_new_score_modifiers = query.properties().
                getString("ranking." + ranking_method + "ScoreModifiers", "");
        
        // Log fetched properties
        logger.info(String.format("YQL %s found: %s", retrieval_method, yql_new));
        logger.info(String.format("Rank Profile %s found: %s", ranking_method, rank_profile_new));
        logger.info(String.format("Rank Profile %s score modifiers found: %s", ranking_method, rank_profile_new_score_modifiers));

        // Create New Subquery
        Query query_new = query.clone();
        query_new.properties().set("yql", yql_new);     // TODO: figure out if this works, output

        // Set fields to search (extract using RETRIEVAL method)
        String feature_name_fields_to_search = add_query_wrapper(QUERY_INPUT_FIELDS_TO_SEARCH + "_" + retrieval_method);
        logger.info("Using fields to search from " + feature_name_fields_to_search);
        Tensor fields_to_search = extract_tensor_rank_feature(query, feature_name_fields_to_search);
        Iterator<Cell> cells = fields_to_search.cellIterator();
        cells.forEachRemaining((cell) -> add_field_to_rank_features(cell, query_new));

        // Set rank profile (using RANKING method)
        if (query.properties().getBoolean("hybrid." + ranking_method + "ScoreModifiersPresent")){
            // With Score Modifiers (using RANKING method)
            query_new.getRanking().setProfile(rank_profile_new_score_modifiers);

            // Extract lexical/tensor rank features and reassign to main rank features.
            // marqo__add_weights_tensor --> marqo__add_weights
            // marqo__mult_weights_tensor --> marqo__mult_weights
            String feature_name_score_modifiers_add_weights = add_query_wrapper(QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS + "_" + ranking_method);
            String feature_name_score_modifiers_mult_weights = add_query_wrapper(QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS + "_" + ranking_method);
            Tensor add_weights = extract_tensor_rank_feature(query, feature_name_score_modifiers_add_weights);
            Tensor mult_weights = extract_tensor_rank_feature(query, feature_name_score_modifiers_mult_weights);
            query_new.getRanking().getFeatures().put(add_query_wrapper(QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS), add_weights);
            query_new.getRanking().getFeatures().put(add_query_wrapper(QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS), mult_weights);

        } else {
            // Without Score Modifiers
            query_new.getRanking().setProfile(rank_profile_new);
        }

        // Log tensor query final state
        logger.info("FINAL QUERY: ");
        logger.info(query_new.toDetailString());
        logger.info(query_new.getModel().getQueryString());
        logger.info(query_new.properties().getString("yql", ""));
        logger.info(query_new.getRanking().getFeatures().toString());

        return query_new;
    }

    /**
     * Implement feature score scaling and normalization
     * @param hits
     * @param features
     */
    void normalize(HitGroup hits, String[] features) {
        // Min - Max normalization
        double[] minValues = new double[features.length];
        double[] maxValues = new double[features.length];
        for(int i = 0; i < features.length;i++) {
            minValues[i] = Double.MAX_VALUE;
            maxValues[i] = Double.MIN_VALUE;
        }

        //Find min and max value in the re-ranking window
        for (Hit hit : hits) {
            if(hit.isAuxiliary())
                continue;
            FeatureData featureData = (FeatureData) hit.getField(MATCH_FEATURES_FIELD);
            if(featureData == null)
                throw new RuntimeException("No feature data in hit - wrong rank profile used?");
            for(int i = 0; i < features.length; i++) {
                // loop through bm25, colbert_maxsim
                double score = featureData.getDouble(features[i]);
                if(score < minValues[i])
                    minValues[i] = score;
                if(score > maxValues[i])
                    maxValues[i] = score;
            }
        }
        //re-score using normalized value
        for (Hit hit : hits) {
            if(hit.isAuxiliary())
                continue;
            FeatureData featureData = (FeatureData) hit.getField(MATCH_FEATURES_FIELD);
            double finalScore = 0;
            for(int i = 0; i < features.length; i++) {
                // loop through bm25, colbert_maxsim
                double score = featureData.getDouble(features[i]);
                // No alpha implemented yet. can be added here
                finalScore += (score - minValues[i]) / (maxValues[i] - minValues[i]);
            }
            // average bm25 and colbert_maxsim. 
            finalScore = finalScore / features.length;
            hit.setRelevance(finalScore);
        }
    }

    /**
     * Print human-readable list of hits with relevances.
     * @param hits
     */
    void printHitGroup(HitGroup hits) {
        logger.info(String.format("Hit Group has size: %s", hits.size()));
        logger.info("=======================");
        int idx = 0;
        for (Hit hit : hits) {
            logger.info(String.format("{IDX: %s, HIT ID: %s, RELEVANCE: %.7f}", idx, hit.getId().toString(), hit.getRelevance().getScore()));
            idx++;
        }
        logger.info("=======================");
    }

    /**
     * Extract a tensor rank feature, throwing an error if it does not exist
     * @param query
     * @param feature_name
     */
    Tensor extract_tensor_rank_feature(Query query, String feature_name){
        Optional<Tensor> optional_tensor = query.getRanking().getFeatures().
            getTensor(feature_name);
        Tensor result_tensor;

        if (optional_tensor.isPresent()){
            result_tensor = optional_tensor.get();
        } else {
            throw new RuntimeException("Rank Feature: " + feature_name + " not found in query!");
        }

        return result_tensor;
    }

    /**
     * Enclose string in query()
     * @param str
     */
    String add_query_wrapper(String str){
        return "query(" + str + ")";
    }
}