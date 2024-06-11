package com.yahoo.search.example;
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
import com.yahoo.net.URI;
import java.util.HashMap;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.lang.InterruptedException;
import java.util.concurrent.ExecutionException;


/**
 * This searcher asks the backend for 2K hits (merged if there are more than one backend content node).
 * The searcher computes the max and min scores for bm25 and colbert_maxsim scores returned
 * using matchfeatures. The scores are then normalized using max-min normalization so that
 * they are in the range 0-1. Finally, the scores are averaged and the 2K hits are re-sorted
 * using this new hybrid score.
 *
 * Using matchfeatures is a cost-efficient way to transfer features calculated by the content nodes
 * to stateless containers.
 *
 */
@Before("ExternalYql")
@Provides("HybridReRanking")
public class HybridSearcher extends Searcher {

    private static String MATCH_FEATURES_FIELD = "matchfeatures";

    @Override
    public Result search(Query query, Execution execution) {

        // Query Properties to implement
        // hybrid.retrievalMethod
        // hybrid.rankingMethod
        // hybrid.rrf_k
        // hybrid.alpha
        // TODO: add score modifiers query_features
        // yql.tensor
        // yql.lexical
        // ranking methods are inherent to the SCHEMA, not the query! So I don't need to pass it

        // Retrieval methods: Disjunction, embedding_similarity, bm25
        // Ranking methods: RRF, normalize_linear, embedding_similarity, bm25

        // Questions:
            // can this 1 searcher handle all 12 combinations?
            // should we have different searchers per retrieval method? 12 searchers?
            // can retrieval and ranking methods be passed as parameters to this searcher?
        
        // Determine hybrid methods to use
        String retrieval_method = query.properties().
                getString("hybrid.retrievalMethod", "");
        
        String ranking_method = query.properties().
                getString("hybrid.rankingMethod", "");
        
        String yql_lexical = query.properties().
                getString("yql.lexical", "");

        String yql_tensor = query.properties().
                getString("yql.tensor", "");
        
        Integer rrf_k = query.properties().getInteger("hybrid.rrf_k", 60);
        Double alpha = query.properties().getDouble("hybrid.alpha", 0.5);

        if (retrieval_method == "disjunction") {
            // Declare result variables
            Result result_lexical, result_tensor;
            Query query_lexical = query.clone();
            query_lexical.properties().set("yql", yql_lexical);
            query_lexical.getRanking().setProfile(query.properties().getString("ranking.scoreModifiersLexical"));

            Query query_tensor = query.clone();
            query_tensor.properties().set("yql", yql_tensor);
            query_lexical.getRanking().setProfile(query.properties().getString("ranking.scoreModifiersTensor"));

            // Execute both searches async
            int timeout = 300 * 1000;       // TODO: make configurable
            AsyncExecution async_execution_lexical = new AsyncExecution(execution);
            Future<Result> future_lexical = async_execution_lexical.search(query_lexical);
            AsyncExecution async_execution_tensor = new AsyncExecution(execution);
            Future<Result> future_tensor = async_execution_tensor.search(query_tensor);
            try {
                result_lexical = future_lexical.get(timeout, TimeUnit.MILLISECONDS);
                result_tensor = future_tensor.get(timeout, TimeUnit.MILLISECONDS);
            } catch(TimeoutException | InterruptedException | ExecutionException e) {
                // TODO: Handle timeout better
                throw new RuntimeException(e.toString());
            }

            // TODO: Possible move this outside, when other retrieval methods are available.
            if (ranking_method == "rrf") {
                HitGroup fused_hit_list = rrf(result_tensor.hits(), result_lexical.hits(), rrf_k, alpha);
                return new Result(query, fused_hit_list);
            }
        }

        return new Result(query, new HitGroup());
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

        // Iterate through tensor hits list
        int rank = 1;
        for (Hit hit : hits_tensor) {
            reciprocal_rank = alpha * (1 / (rank + k));
            rrf_scores.put(hit.getId().toString(), reciprocal_rank);   // Store hit's score via its URI
            hit.setRelevance(reciprocal_rank);                 // Update score to be weighted RR (tensor)
            result.add(hit);
            rank++;
        }

        // Iterate through lexical hits list
        rank = 1;
        for (Hit hit : hits_lexical) {
            reciprocal_rank = (1-alpha) * (1 / (rank + k));

            // Check if score already exists. If so, add to it.
            existing_score = rrf_scores.get(hit.getId().toString());
            if (existing_score == null){
                existing_score = 0.0;
            }
            new_score = existing_score + reciprocal_rank;
            rrf_scores.put(hit.getId().toString(), new_score);
            hit.setRelevance(new_score);      // Update score to be weighted RR (lexical)
            result.add(hit);
            rank++;
        }

        // sort result
        result.sort();

        // Only return top hits (max length)
        Integer final_length = Math.max(hits_tensor.size(), hits_lexical.size());
        result.trim(0, final_length);

        return result;
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
}