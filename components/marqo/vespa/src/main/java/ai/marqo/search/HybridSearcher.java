package ai.marqo.search;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.google.common.annotations.VisibleForTesting;
import com.sun.jdi.InternalException;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.data.JsonProducer;
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
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.statistics.descriptive.DoubleStatistics;
import org.apache.commons.statistics.descriptive.Statistic;
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
    private static String QUERY_INPUT_CUSTOM_SCORE_MULT_WEIGHTS_GLOBAL =
            "marqo__custom_score_mult_weights_global";
    private static String QUERY_INPUT_CUSTOM_SCORE_ADD_WEIGHTS_GLOBAL =
            "marqo__custom_score_add_weights_global";

    private static String QUERY_INPUT_RECENCY_TIMESTAMP_KEY = "marqo__recency_timestamp_key";
    private static String QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE =
            "marqo__recency_should_apply_score";
    private static String QUERY_INPUT_RECENCY_APPLY_TO_TENSOR = "marqo__recency_apply_to_tensor";
    private static String QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL = "marqo__recency_apply_to_lexical";
    private static String MARQO_SEARCH_METHOD_LEXICAL = "lexical";
    private static String MARQO_SEARCH_METHOD_TENSOR = "tensor";
    private static String QUERY_RERANK_COUNT = "ranking.rerankCount";
    private List<String> STANDARD_SEARCH_TYPES = new ArrayList<>();

    // Thread-safe ObjectReader for parsing SortField JSON
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final ObjectReader SORT_FIELD_READER =
            OBJECT_MAPPER.readerFor(new TypeReference<List<SortField>>() {});
    private static final ObjectMapper MARQO_METADATA_FIELDS_MAPPER = new ObjectMapper();

    // A magic number used to represent missing sort field values in search results as we can only
    // return numeric values in match-features.
    // The value -1e50 is chosen as it is an extremely low number unlikely to occur in real data.
    private static final double MISSING_SORT_VALUE_SENTINEL = -1e50;

    private static final String MARQO_METADATA_FIELDS = "marqo__fields";
    private static final String FACETS_YQL_QUERY_DELIMITER = "\n---MARQO-YQL-QUERY-DELIMITER---\n";

    /** Hit field for the score before applying custom/global modifiers (RRF score). Exposed as _pre_rerank_score in API. */
    private static final String MARQO_PRE_RERANK_SCORE = "marqo__pre_rerank_score";

    /** Key under which Vespa attaches rank profile summary-features to a hit. */
    private static final String SUMMARY_FEATURES_FIELD = "summaryfeatures";

    /** Returns FeatureData for custom score (bm25, ranking_closeness_metric_*) from hit summaryfeatures. */
    private static FeatureData getSummaryFeaturesForHit(Hit hit) {
        Object o = hit.getField(SUMMARY_FEATURES_FIELD);
        return o instanceof FeatureData ? (FeatureData) o : null;
    }

    @VisibleForTesting
    @JsonInclude(Include.NON_NULL)
    record MarqoMetadataFields(
            Integer sortCandidates,
            Integer probeCandidates,
            Integer relevantCandidates,
            Integer postProcessCandidates)
            implements JsonProducer {

        @Override
        public StringBuilder writeJson(StringBuilder target) {
            try {
                target.append(MARQO_METADATA_FIELDS_MAPPER.writeValueAsString(this));
                return target;
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Parsed custom score rerank key. Matches Python ParsedCustomScoreKey.parse.
     * Key formats: {scoreType}_field_{fieldName} or {scoreType}_{sum|max|avg}.
     */
    @VisibleForTesting
    static final class CustomScoreKey {
        final String scoreType;
        final String fieldName; // null for aggregate
        final String aggregateType; // null for single-field

        CustomScoreKey(String scoreType, String fieldName, String aggregateType) {
            this.scoreType = scoreType;
            this.fieldName = fieldName;
            this.aggregateType = aggregateType;
        }

        /**
         * Parses a custom score rerank key (no prefix) into score type, field name and aggregate type.
         * Key formats: {scoreType}_field_{fieldName} or {scoreType}_{sum|max|avg}.
         *
         * @param key Key without prefix, e.g. "bm25_field_title", "bm25_sum",
         *     "closeness_retrieval_vector_field_variantImage"
         * @return Parsed key or null if unsupported/invalid
         */
        @VisibleForTesting
        static CustomScoreKey parseCustomScoreKey(String key) {
            if (key == null || key.isEmpty() || !key.contains("_")) {
                return null;
            }
            String[] supportedScoreTypes = {"bm25", "closeness_retrieval_vector"};
            String[] aggregateTypes = {"sum", "max", "avg"};
            for (String scoreType : supportedScoreTypes) {
                String prefix = scoreType + "_";
                if (!key.startsWith(prefix)) {
                    continue;
                }
                String rest = key.substring(prefix.length());
                if (Arrays.asList(aggregateTypes).contains(rest)) {
                    return new CustomScoreKey(scoreType, null, rest);
                }
                if (rest.startsWith("field_")) {
                    String fieldName = rest.substring(6);
                    if (fieldName.isEmpty()) {
                        return null;
                    }
                    return new CustomScoreKey(scoreType, fieldName, null);
                }
                return null;
            }
            return null;
        }

        /** Summary feature name for single-field BM25: bm25(marqo__lexical_<fieldName>). One per lexical field; no bm25(marqo__ranking_strings). */
        static String bm25SummaryFeatureName(String fieldName) {
            if (fieldName == null || fieldName.isEmpty()) {
                return null;
            }
            return "bm25(marqo__lexical_" + fieldName + ")";
        }
    }

    /**
     * Represents a field to sort by in the search results.
     * Uses Java record for immutability and conciseness.
     */
    private record SortField(
            @JsonProperty("field_name") String fieldName,
            @JsonProperty("order") SortOrder order,
            @JsonProperty("missing") MissingOrder missing) {}

    /**
     * Sort order enum for better type safety
     */
    private enum SortOrder {
        ASC,
        DESC;

        @JsonCreator
        public static SortOrder fromString(String value) {
            return SortOrder.valueOf(value.toUpperCase(Locale.ROOT));
        }
    }

    /**
     * Missing value handling enum
     */
    private enum MissingOrder {
        FIRST,
        LAST;

        @JsonCreator
        public static MissingOrder fromString(String value) {
            return MissingOrder.valueOf(value.toUpperCase(Locale.ROOT));
        }
    }

    private enum RelevanceCutoffMethod {
        RELATIVE_MAX_SCORE,
        MEAN_STD_DEV,
        GAP_DETECTION;

        @JsonCreator
        public static RelevanceCutoffMethod fromString(String value) {
            return RelevanceCutoffMethod.valueOf(value.toUpperCase(Locale.ROOT));
        }
    }

    enum ApplyInRetrieval {
        LEXICAL,
        TENSOR,
        BOTH;

        public static ApplyInRetrieval fromString(String value) {
            if (value == null) return null;
            return ApplyInRetrieval.valueOf(value.toUpperCase(Locale.ROOT));
        }
    }

    // Compile the regex pattern once and store it as a static final variable
    private static final Pattern DOC_ID_PATTERN =
            Pattern.compile("^index\\:[^\\s\\/]+\\/\\d+\\/(.+)$");
    private static final Pattern TARGET_HITS_PATTERN =
            Pattern.compile("(targetHits\\s*:\\s*)(\\d+)");
    private static final Pattern HNSW_EXPLORE_ADDITIONAL_HITS_PATTERN =
            Pattern.compile("(hnsw\\.exploreAdditionalHits\\s*:\\s*)(\\d+)");

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

        // Relevance Cut-off Parameters
        String relevanceCutoffMethod =
                query.properties().getString("marqo__hybrid.relevanceCutoff.method", null);
        Integer relevanceCutoffProbeDepth =
                query.properties().getInteger("marqo__hybrid.relevanceCutoff.probeDepth", null);
        Double relevanceCutoffParameter =
                readRelevanceCutoffParameter(query, relevanceCutoffMethod);
        Boolean relevanceCutoffAffectFacets =
                query.properties().getBoolean("marqo__hybrid.relevanceCutoff.affectFacets", false);
        Boolean relevanceCutoffOverrideSortCandidates =
                query.properties()
                        .getBoolean("marqo__hybrid.relevanceCutoff.overrideSortCandidates", false);
        String applyInRetrievalString =
                query.properties()
                        .getString("marqo__hybrid.relevanceCutoff.applyInRetrieval", null);
        ApplyInRetrieval applyInRetrieval = ApplyInRetrieval.fromString(applyInRetrievalString);
        Boolean relevanceCutoffOverrideLimitPlusOffset =
                query.properties()
                        .getBoolean("marqo__hybrid.relevanceCutoff.overrideLimitPlusOffset", false);

        // Sort by Parameters
        String sortByFields = query.properties().getString("marqo__hybrid.sortBy.fields", null);
        Integer sortBySortDepth =
                query.properties().getInteger("marqo__hybrid.sortBy.sortDepth", null);
        Integer sortByMinSortCandidates =
                query.properties().getInteger("marqo__hybrid.sortBy.minSortCandidates", null);

        // Collapse Parameters
        boolean collapse = query.properties().getString("collapsefield") != null;

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

        // --- Begin relevance cut-off handling ---
        // Execute probe lexical search for relevance cut-off if parameters are provided.
        // This must run BEFORE facets so we can apply max(relevantCandidates) to facets grouping.
        Integer relevantCandidates = null;
        Integer probeCandidates = null;
        if (relevanceCutoffMethod != null) {
            logIfVerbose("Executing probe lexical search for relevance cut-off", verbose);
            Query probeLexicalQuery =
                    createProbeLexialQuery(query, relevanceCutoffProbeDepth, verbose);
            Result probeLexicalResult = execution.search(probeLexicalQuery);
            probeCandidates = probeLexicalResult.hits().size();
            relevantCandidates =
                    detectCutoffCount(
                            probeLexicalResult.hits(),
                            relevanceCutoffMethod,
                            relevanceCutoffParameter,
                            verbose);
        }
        // --- End relevance cut-off handling ---

        // --- Update the query hits, offset, targetHits, and facets YQL
        boolean isRelevanceCutoffMethodEnabled = relevanceCutoffMethod != null;
        boolean isSortByEnabled = sortByFields != null;

        // When applyInRetrieval targets a specific leg, we clone the query and let
        // the clone receive all cutoff modifications (hits, offset, tensor YQL, facets).
        // The original query stays untouched so its hits won't truncate the final Result.
        // Target sub-query is created from the clone; non-target from the original.
        boolean isSelectiveCutoff =
                isRelevanceCutoffMethodEnabled
                        && applyInRetrieval != null
                        && applyInRetrieval != ApplyInRetrieval.BOTH;

        // Clone the query for relevance cutoff & sortBy manipulation
        Query cutoffSortByQuery = query.clone();
        if (isRelevanceCutoffMethodEnabled || isSortByEnabled) {
            cutoffSortByQuery =
                    updateQueryHitsOffsetsAndTargetHits(
                            cutoffSortByQuery,
                            relevantCandidates,
                            sortByMinSortCandidates,
                            isRelevanceCutoffMethodEnabled,
                            isSortByEnabled,
                            relevanceCutoffAffectFacets,
                            relevanceCutoffOverrideLimitPlusOffset,
                            verbose);
        }

        // Facet results will always be generated from a cutoff query to produce a conservative
        // count until we fix the
        // implementation in the future
        List<Future<Result>> futureFacets;
        if (isRelevanceCutoffMethodEnabled || isSortByEnabled) {
            futureFacets = getFacetsFutureList(cutoffSortByQuery, execution, verbose, collapse);
        } else {
            futureFacets = getFacetsFutureList(query, execution, verbose, collapse);
        }

        HitGroup hitsForPostProcessing;
        if (retrievalMethod.equals("disjunction")) {
            Result resultLexical, resultTensor;
            Query queryLexical, queryTensor;

            Query[] subQueries =
                    buildDisjunctionSubQueries(
                            isSelectiveCutoff,
                            applyInRetrieval,
                            query,
                            cutoffSortByQuery,
                            relevanceCutoffProbeDepth,
                            verbose);
            queryLexical = subQueries[0];
            queryTensor = subQueries[1];

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
            HitGroup combinedErrors =
                    collectErrorsFromResults(resultLexical, resultTensor, verbose);
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
                        rrf(
                                resultTensor.hits(),
                                resultLexical.hits(),
                                rrf_k,
                                alpha,
                                verbose,
                                collapse);
            } else {
                throw new RuntimeException(
                        "For retrievalMethod='disjunction', rankingMethod must be 'rrf'.");
            }

        } else if (STANDARD_SEARCH_TYPES.contains(retrievalMethod)) {
            if (STANDARD_SEARCH_TYPES.contains(rankingMethod)) {
                Query combinedQuery =
                        createSubQuery(cutoffSortByQuery, retrievalMethod, rankingMethod, verbose);
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

        // Fill summary-features before reranking when custom score uses ranking vector or BM25
        boolean hasRankingVector = query.properties().getBoolean("marqo__hasRankingVector", false);
        boolean hasRankingLexical =
                query.properties().getBoolean("marqo__hasRankingLexical", false);
        if (hasRankingVector || hasRankingLexical) {
            Result resultToFill = new Result(query, hitsForPostProcessing);
            execution.fill(resultToFill, "dummy-light-summary");
        }

        // Determine post-processing mode based on query parameters
        HitGroup processedHits;
        Integer sortCandidates = null;
        int postProcessCandidates;
        if (sortByFields != null) {
            // When overrideSortCandidates is set, trim hits to only relevant candidates
            // before sorting, so that non-relevant documents are excluded from sort results.
            if (relevanceCutoffOverrideSortCandidates
                    && relevantCandidates != null
                    && relevantCandidates < hitsForPostProcessing.size()) {

                int targetTrimSize =
                        (sortByMinSortCandidates != null)
                                ? Math.max(sortByMinSortCandidates, relevantCandidates)
                                : relevantCandidates;
                targetTrimSize = Math.min(targetTrimSize, hitsForPostProcessing.size());

                if (targetTrimSize < hitsForPostProcessing.size()) {
                    List<Hit> trimmed =
                            new ArrayList<>(
                                    hitsForPostProcessing.asList().subList(0, targetTrimSize));
                    hitsForPostProcessing = new HitGroup();
                    trimmed.forEach(hitsForPostProcessing::add);
                }
            }
            // If sortBy is set, we will sort the hits after post-processing
            processedHits =
                    postProcessBySort(
                            hitsForPostProcessing, sortByFields, sortBySortDepth, limit, offset);
            sortCandidates = hitsForPostProcessing.size();
            postProcessCandidates = hitsForPostProcessing.size();
        } else {
            // If sortBy is not set, we use the default post-processing
            processedHits =
                    postProcessResults(
                            hitsForPostProcessing,
                            query,
                            rerankDepthGlobal,
                            limit,
                            offset,
                            verbose);
            postProcessCandidates = hitsForPostProcessing.size();
        }

        if (!futureFacets.isEmpty()) {
            attachFacetsResult(futureFacets, timeout, processedHits, verbose);
        }

        // Extract recency multiplier from match features after post-processing (only if recency is
        // enabled)
        processedHits = extractRecencyScore(processedHits, query, verbose);
        MarqoMetadataFields marqoMetadataFields =
                new MarqoMetadataFields(
                        sortCandidates, probeCandidates, relevantCandidates, postProcessCandidates);

        processedHits.setField(MARQO_METADATA_FIELDS, marqoMetadataFields);
        return new Result(query, processedHits);
    }

    private void attachFacetsResult(
            List<Future<Result>> futureFacets,
            Integer timeout,
            HitGroup processedHits,
            boolean verbose) {
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

    @VisibleForTesting
    List<Future<Result>> getFacetsFutureList(
            Query query, Execution execution, boolean verbose, boolean collapse) {
        // Check for custom facets YQL properties - expect array of strings
        String[] facetsYqlQueries =
                query.properties()
                        .getString("marqo__yql.facets", "")
                        .split(FACETS_YQL_QUERY_DELIMITER);
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
                if (collapse) {
                    // Carrying collapsefield parameter to facets query will cause extra count since
                    // CollapseFieldSearch does extra searches
                    queryFacets.properties().set("collapsefield", null);
                }
                AsyncExecution asyncExecutionFacets = new AsyncExecution(execution);
                futureFacets.add(asyncExecutionFacets.search(queryFacets));
            }
        }
        return futureFacets;
    }

    /**
     * Builds lexical and tensor sub-queries for disjunction retrieval.
     *
     * @return a two-element array {@code [queryLexical, queryTensor]}
     */
    Query[] buildDisjunctionSubQueries(
            boolean isSelectiveCutoff,
            ApplyInRetrieval applyInRetrieval,
            Query originalQuery,
            Query cutoffQuery,
            Integer probeDepth,
            boolean verbose) {
        Query queryLexical, queryTensor;
        if (isSelectiveCutoff) {
            // Target from cutoffQuery (reduced), non-target from original (unreduced)
            if (applyInRetrieval == ApplyInRetrieval.LEXICAL) {
                throw new RuntimeException(
                        "applyInRetrieval='lexical' is not supported. This value is blocked at"
                                + " the API layer and should never reach this point.");
            } else {
                // Must be tensor cutoff
                queryLexical =
                        createSubQuery(
                                originalQuery,
                                MARQO_SEARCH_METHOD_LEXICAL,
                                MARQO_SEARCH_METHOD_LEXICAL,
                                verbose);
                queryLexical.setOffset(0);
                // The lexical leg is not subject to relevance cut-off here; expand its hits to
                // probeDepth so the post-process/sort pool remains stable regardless of
                // limit/offset.
                queryLexical.setHits(probeDepth);
                queryTensor =
                        createSubQuery(
                                cutoffQuery,
                                MARQO_SEARCH_METHOD_TENSOR,
                                MARQO_SEARCH_METHOD_TENSOR,
                                verbose);
            }
        } else {
            // Two cases both handled correctly by using cutoffQuery:
            // 1. No relevance cutoff: cutoffQuery is a plain clone of the original, so both legs
            //    are unmodified.
            // 2. applyInRetrieval=both: cutoffQuery is reduced and both legs should use the same
            //    reduced query.
            queryLexical =
                    createSubQuery(
                            cutoffQuery,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            verbose);
            queryTensor =
                    createSubQuery(
                            cutoffQuery,
                            MARQO_SEARCH_METHOD_TENSOR,
                            MARQO_SEARCH_METHOD_TENSOR,
                            verbose);
        }
        return new Query[] {queryLexical, queryTensor};
    }

    /**
     * Injects max(N) into the outermost all() of a facets grouping YQL expression.
     *
     * <p>Facets YQL has the form: {@code <select clause> | <grouping expression>}
     * where the grouping expression starts with {@code all(}. This method injects
     * {@code max(relevantCandidates)} right after the opening {@code all(} so that
     * Vespa only processes the top N documents (by relevance) for the aggregation.
     *
     * @param facetsYql The facets YQL string to modify.
     * @param relevantCandidates The number of relevant candidates to limit grouping to.
     * @param verbose Whether to log the modification.
     * @return The modified facets YQL with max(N) injected.
     */
    // Regex to find "| all(" followed by optional whitespace and an optional "max(M)" in the
    // outermost grouping expression. Uses lastIndexOf for the "|" split, then regex on the
    // grouping part.
    private static final Pattern OUTER_MAX_PATTERN = Pattern.compile("^all\\(\\s*max\\((\\d+)\\)");

    @VisibleForTesting
    String injectMaxHitsIntoFacetsGrouping(String facetsYql, int maxHits, boolean verbose) {
        if (facetsYql == null || facetsYql.isEmpty()) {
            logIfVerbose("Empty or null facets YQL, skipping max injection", verbose);
            return facetsYql == null ? "" : facetsYql;
        }

        // Find "| all(" — the separator between select clause and grouping.
        // We match on "| all(" rather than just "|" to be defensive against "|" appearing
        // in field names, search terms, or other parts of the YQL.
        // Use lastIndexOf because the grouping clause is always at the end.
        int pipeAllIndex = facetsYql.lastIndexOf("| all(");
        if (pipeAllIndex == -1) {
            logIfVerbose(
                    "No '| all(' found in facets YQL, skipping max injection: " + facetsYql,
                    verbose);
            return facetsYql;
        }

        String selectPart = facetsYql.substring(0, pipeAllIndex + 1); // includes the "|"
        String groupingPart = facetsYql.substring(pipeAllIndex + 1).trim(); // "all(..."

        // Check if there's already a max(M) in the outermost all().
        // Pattern: all( max(M) ... ) — with optional whitespace after "all("
        Matcher matcher = OUTER_MAX_PATTERN.matcher(groupingPart);
        String result;
        if (matcher.find()) {
            int existingMax = Integer.parseInt(matcher.group(1));
            if (maxHits < existingMax) {
                // Replace max(M) with max(N) since N is more restrictive
                result =
                        selectPart
                                + " "
                                + groupingPart.substring(0, matcher.start(1))
                                + maxHits
                                + groupingPart.substring(matcher.end(1));
                logIfVerbose(
                        String.format(
                                "Replaced max(%d) with max(%d) in facets grouping: %s",
                                existingMax, maxHits, result),
                        verbose);
            } else {
                // Existing max is already <= maxHits, skip
                logIfVerbose(
                        String.format(
                                "Existing max(%d) <= %d, skipping injection", existingMax, maxHits),
                        verbose);
                return facetsYql;
            }
        } else {
            // No max(M) present — wrap inner content with max(N) all(...)
            // e.g., all(group(1.1) each(...)) -> all(max(N) all(group(1.1) each(...)))
            // e.g., all( all(group(...) ...)) -> all(max(N) all(all(group(...) ...)))
            // Note: for facets, inner content already starts with all(), so no extra nesting.
            String innerContent = groupingPart.substring(4, groupingPart.length() - 1).trim();
            result = selectPart + " all(max(" + maxHits + ") all(" + innerContent + "))";
            logIfVerbose(
                    String.format("Inserted max(%d) into facets grouping: %s", maxHits, result),
                    verbose);
        }
        return result;
    }

    /**
     * Overload without facets parameters — delegates with false defaults.
     */
    public Query updateQueryHitsOffsetsAndTargetHits(
            Query query,
            Integer relevantCandidates,
            Integer sortByMinSortCandidates,
            boolean isRelevanceCutoffEnabled,
            boolean isSortByEnabled) {
        return updateQueryHitsOffsetsAndTargetHits(
                query,
                relevantCandidates,
                sortByMinSortCandidates,
                isRelevanceCutoffEnabled,
                isSortByEnabled,
                false,
                false,
                false);
    }

    /**
     * Updates the query hits, offsets, and targetHits based on the provided parameters.
     * Also updates the facets YQL: adjusts targetHits to match the main query and,
     * when facetsRelevantCandidates is provided, injects max(N) into facets grouping.
     * @param query the query to update.
     * @param relevantCandidates the number of relevant candidates found in the probe search.
     * @param sortByMinSortCandidates the minimum number of candidates required for sorting, from Marqo
     * @param isRelevanceCutoffEnabled whether relevance cutoff is enabled.
     * @param isSortByEnabled whether sorting is enabled.
     * @param affectFacets whether to also adjust facets YQL (targetHits and max(N) grouping).
     * @param overrideLimitPlusOffset when true, use max(relevantCandidates, limit+offset) instead of min.
     * @param verbose whether to log detailed information.
     * @return The updated query with new hits, offsets, and targetHits.
     */
    public Query updateQueryHitsOffsetsAndTargetHits(
            Query query,
            Integer relevantCandidates,
            Integer sortByMinSortCandidates,
            boolean isRelevanceCutoffEnabled,
            boolean isSortByEnabled,
            boolean affectFacets,
            boolean overrideLimitPlusOffset,
            boolean verbose) {

        // Validate input parameters
        if (!isRelevanceCutoffEnabled && !isSortByEnabled) {
            return query;
        }

        if (relevantCandidates == null && sortByMinSortCandidates == null) {
            throw new RuntimeException(
                    "Either relevantCandidates or sortByMinSortCandidates must be provided");
        }

        sortByMinSortCandidates = (sortByMinSortCandidates == null) ? -1 : sortByMinSortCandidates;
        relevantCandidates = (relevantCandidates == null) ? -1 : relevantCandidates;

        // Extract current tensor YQL for potential targetHits update
        String tensorYQL =
                query.properties().getString("marqo__yql." + MARQO_SEARCH_METHOD_TENSOR, "");

        Integer currentTensorTargetHits = null;
        Integer currentExploreAdditionalHits = null;
        int currentLimit = query.getHits();
        int currentOffset = query.getOffset();

        if (!tensorYQL.isEmpty()) {
            currentTensorTargetHits = extractCurrentTargetHits(tensorYQL);
            if (currentTensorTargetHits < (currentLimit + currentOffset)) {
                throw new RuntimeException(
                        "The targetHits in the tensor query should not be smaller than"
                                + " limit+offset");
            }
            currentExploreAdditionalHits = extractCurrentExploreAdditionalHits(tensorYQL);
        }

        Integer newHits = null;
        Integer newTensorTargetHits = null;

        if (isRelevanceCutoffEnabled && isSortByEnabled) {
            // Both sort and relevance cut-off enabled: use the higher one as new limits.
            // Note that sortByMinSortCandidates is guaranteed to be larger than limit+offset so no
            // check is required
            newHits = Math.max(relevantCandidates, sortByMinSortCandidates);
            if (currentTensorTargetHits != null) {
                if (overrideLimitPlusOffset) {
                    newTensorTargetHits = newHits;
                } else {
                    newTensorTargetHits = Math.max(newHits, currentTensorTargetHits);
                }
            }
        } else if (isRelevanceCutoffEnabled) {
            if (overrideLimitPlusOffset) {
                // Override mode: expand retrieval to max(relevantCandidates, limit+offset)
                // so all relevant documents are fetched even if they exceed limit+offset.
                newHits = relevantCandidates;
                if (currentTensorTargetHits != null) {
                    newTensorTargetHits = relevantCandidates;
                }
            } else {
                // Default: reduce to min(relevantCandidates, limit+offset)
                newHits = Math.min(relevantCandidates, (currentLimit + currentOffset));
                if (currentTensorTargetHits != null) {
                    newTensorTargetHits = Math.min(newHits, currentTensorTargetHits);
                }
            }
        } else {
            // Only sortByMinSortCandidates provided
            newHits = Math.max(sortByMinSortCandidates, (currentLimit + currentOffset));
            if (currentTensorTargetHits != null) {
                newTensorTargetHits = Math.max(newHits, currentTensorTargetHits);
            }
        }

        // Update query hits and offset
        query.setHits(newHits);
        query.setOffset(0);
        query.properties().set(QUERY_RERANK_COUNT, newHits);

        // Update tensor YQL targetHits if it exists
        /* Update to lexical YQL is not needed
        1. if the relevantCandidates < current lexical targetHits, fetching more is not harmful,
        2. if the relevantCandidates > current lexical targetHits, we should not increase it as it
        could potentially change the results set. We don't want to change the result set if
        relevance cut-off determines a higher hits number. This behavior is consistent with how we
        handle tensor targetHits.
         */
        if (currentTensorTargetHits != null
                && !Objects.equals(currentTensorTargetHits, newTensorTargetHits)) {
            int efSearch = currentTensorTargetHits + currentExploreAdditionalHits;
            String tensorYQLUpdated =
                    overwriteTargetHitsAndExploreAdditionalHits(
                            tensorYQL, newTensorTargetHits, efSearch);
            query.properties().set("marqo__yql." + MARQO_SEARCH_METHOD_TENSOR, tensorYQLUpdated);
        }

        // When affectFacets is opted in, update facets YQL:
        // - Adjust targetHits to newTensorTargetHits (consistent with main tensor query)
        // - Inject max(newHits) into grouping expressions
        String facetsYql = query.properties().getString("marqo__yql.facets", "");
        if (affectFacets && !facetsYql.isEmpty()) {
            String delimiter = FACETS_YQL_QUERY_DELIMITER;
            String[] queries = facetsYql.split(delimiter);
            for (int i = 0; i < queries.length; i++) {
                if (!queries[i].isEmpty()) {
                    if (currentTensorTargetHits != null
                            && !Objects.equals(currentTensorTargetHits, newTensorTargetHits)) {
                        int efSearch = currentTensorTargetHits + currentExploreAdditionalHits;
                        String tensorFacetYQLUpdated =
                                overwriteTargetHitsAndExploreAdditionalHits(
                                        queries[i], newTensorTargetHits, efSearch);
                        queries[i] = tensorFacetYQLUpdated;
                    }

                    queries[i] = injectMaxHitsIntoFacetsGrouping(queries[i], newHits, verbose);
                }
            }
            query.properties().set("marqo__yql.facets", String.join(delimiter, queries));
        }

        return query;
    }

    /**
     * Post processes the hits by sorting them based on the provided sortByFields.
     * @param hitsForPostProcessing the hits to be sorted.
     * @param sortByFields the JSON string representing the fields to sort by.
     * @param sortBySortDepth the depth to sort by, or null to sort all hits.
     * @param limit the maximum number of hits to return.
     * @param offset the offset for pagination.
     * @return a HitGroup containing the sorted hits.
     */
    HitGroup postProcessBySort(
            HitGroup hitsForPostProcessing,
            String sortByFields,
            Integer sortBySortDepth,
            Integer limit,
            Integer offset) {

        List<SortField> parsedSortByFields;

        try {
            parsedSortByFields = SORT_FIELD_READER.readValue(sortByFields);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(
                    "Invalid sort JSON format for marqo__hybrid.sortBy.fields", e);
        }

        // Validate sort fields requirements
        if (parsedSortByFields.isEmpty()) {
            throw new RuntimeException(
                    "sortBy fields cannot be empty. Must contain 1 to 3 sort fields.");
        }
        if (parsedSortByFields.size() > 3) {
            throw new RuntimeException(
                    "sortBy fields cannot contain more than 3 sort fields. Found: "
                            + parsedSortByFields.size());
        }

        // Validate that all required fields are provided
        for (int i = 0; i < parsedSortByFields.size(); i++) {
            SortField field = parsedSortByFields.get(i);
            if (field.fieldName() == null || field.fieldName().trim().isEmpty()) {
                throw new RuntimeException("fieldName is required for sort field at index " + i);
            }
            if (field.order() == null) {
                throw new RuntimeException("order is required for sort field at index " + i);
            }
            if (field.missing() == null) {
                throw new RuntimeException("missing is required for sort field at index " + i);
            }
        }

        // Validate sortBySortDepth requirements
        if (sortBySortDepth != null && sortBySortDepth < 1) {
            throw new RuntimeException(
                    "sortBySortDepth must be greater than or equal to 1. Found: "
                            + sortBySortDepth);
        }

        List<Hit> allHits = new ArrayList<>(hitsForPostProcessing.asList());
        int depth = (sortBySortDepth != null) ? sortBySortDepth : allHits.size();

        List<Hit> hitsToSort = new ArrayList<>(allHits.subList(0, Math.min(depth, allHits.size())));
        List<Hit> hitsAfterDepth =
                (depth < allHits.size())
                        ? new ArrayList<>(allHits.subList(depth, allHits.size()))
                        : new ArrayList<>();

        // Build comparator chain based on configured SortFields
        Comparator<Hit> sortComparator = null;
        for (int i = 0; i < parsedSortByFields.size(); i++) {
            final int idx = i;
            SortField sf = parsedSortByFields.get(i);
            Function<Hit, Double> keyExtractor =
                    hit -> {
                        FeatureData mf = (FeatureData) hit.getField("matchfeatures");
                        if (mf == null) return null;
                        double v = mf.getDouble("sort_field_value_" + idx);
                        return (v == MISSING_SORT_VALUE_SENTINEL) ? null : v;
                    };
            Comparator<Double> base = Comparator.naturalOrder();
            if (sf.order() == SortOrder.DESC) {
                base = base.reversed();
            }
            Comparator<Double> nullAware =
                    sf.missing() == MissingOrder.LAST
                            ? Comparator.nullsLast(base)
                            : Comparator.nullsFirst(base);
            Comparator<Hit> fieldComparator = Comparator.comparing(keyExtractor, nullAware);
            sortComparator =
                    (sortComparator == null)
                            ? fieldComparator
                            : sortComparator.thenComparing(fieldComparator);
        }

        // Append relevance as final tie-breaker (always descending)
        Comparator<Hit> relevanceTie =
                Comparator.comparingDouble((Hit h) -> h.getRelevance().getScore()).reversed();
        // always combine with relevance tie-break
        sortComparator =
                (sortComparator == null)
                        ? relevanceTie
                        : sortComparator.thenComparing(relevanceTie);
        hitsToSort.sort(sortComparator);

        // Merge sorted slice with the remainder
        List<Hit> combined = new ArrayList<>(hitsToSort);
        combined.addAll(hitsAfterDepth);

        // Reset relevance to reflect new positions
        for (int i = 0; i < combined.size(); i++) {
            combined.get(i).setRelevance(1.0 / (i + 1));
        }
        /*
         * TODO - check HitGroup.setOrdered and HitGroup HitSortOrderer
         *  for better performance and avoiding of sorting in the downstream code.
         */
        HitGroup result = new HitGroup();
        result.addAll(combined);

        result.trim(offset, limit);

        return result;
    }

    /**
     * Read the relevance cutoff parameter based on the relevance cutoff method.
     **/
    @VisibleForTesting
    Double readRelevanceCutoffParameter(Query query, String relevanceCutoffMethodString) {
        if (relevanceCutoffMethodString == null) {
            return null;
        }
        RelevanceCutoffMethod relevanceCutoffMethod;
        try {
            relevanceCutoffMethod = RelevanceCutoffMethod.fromString(relevanceCutoffMethodString);
        } catch (IllegalArgumentException e) {
            throw new RuntimeException(
                    "Unknown relevance cutoff method: " + relevanceCutoffMethodString);
        }

        switch (relevanceCutoffMethod) {
            case RELATIVE_MAX_SCORE -> {
                Double value =
                        query.properties()
                                .getDouble(
                                        "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor");
                if (value == null) {
                    throw new RuntimeException(
                            "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor is"
                                    + " missing");
                }
                return value;
            }
            case MEAN_STD_DEV -> {
                Double value =
                        query.properties()
                                .getDouble("marqo__hybrid.relevanceCutoff.parameters.stdDevFactor");
                if (value == null) {
                    throw new RuntimeException(
                            "marqo__hybrid.relevanceCutoff.parameters.stdDevFactor is missing");
                }
                return value;
            }
            case GAP_DETECTION -> {
                // no cutoff parameter for gap detection
                return null;
            }
        }
        return null;
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
            HitGroup hitsTensor,
            HitGroup hitsLexical,
            Integer k,
            Double alpha,
            boolean verbose,
            boolean collapse) {

        HashMap<String, Double> rrfScores = new HashMap<>();
        HashMap<String, String> docIdsToHitIds = new HashMap<>();
        HashMap<Double, String> collapseFieldHashToDocId = new HashMap<>();
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
                if (collapse) {
                    Double collapseFieldHash = extractCollapseFieldHash(hit);
                    if (collapseFieldHash != null) {
                        collapseFieldHashToDocId.put(collapseFieldHash, extractedDocId);
                    }
                }

                addHitToResult(
                        hit,
                        reciprocalRank,
                        rrfScores,
                        extractedDocId,
                        docIdsToHitIds,
                        result,
                        "marqo__raw_tensor_score");

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

                if (existingScore != null) {
                    // The document already exists in tensor result, so we fuse the scores
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

                } else if (!collapse) {
                    // If the document doesn't exist and there is no collapsing, add new hit to
                    // result list
                    logIfVerbose("No existing score found! Starting at 0.0.", verbose);
                    addHitToResult(
                            hit,
                            reciprocalRank,
                            rrfScores,
                            extractedDocId,
                            docIdsToHitIds,
                            result,
                            "marqo__raw_lexical_score");

                } else {
                    // Collapse field logic for lexical hits
                    Double collapseFieldHash = extractCollapseFieldHash(hit);
                    String extractedDocIdInTensor = collapseFieldHashToDocId.get(collapseFieldHash);

                    if (extractedDocIdInTensor == null) {
                        // No hit with the same collapse_field_hash exists in tensor result. Add it.
                        logIfVerbose(
                                String.format(
                                        "No existing doc with the same collapse field (with hash"
                                                + " %.7f) found! Add this doc with id: %s.",
                                        collapseFieldHash, extractedDocId),
                                verbose);
                        addHitToResult(
                                hit,
                                reciprocalRank,
                                rrfScores,
                                extractedDocId,
                                docIdsToHitIds,
                                result,
                                "marqo__raw_lexical_score");

                    } else if (!extractedDocId.equals(extractedDocIdInTensor)) {
                        // Different document with same collapse field hash
                        Double existingScoreInTensor = rrfScores.get(extractedDocIdInTensor);
                        logIfVerbose(
                                String.format(
                                        "Found hit with same collapse field (with hash %.7f) in"
                                                + " tensor. Existing score: %.7f. Id is: %s.",
                                        collapseFieldHash,
                                        existingScoreInTensor,
                                        extractedDocIdInTensor),
                                verbose);

                        if (reciprocalRank > existingScoreInTensor) {
                            // Discard the tensor hit, use this lexical hit
                            logIfVerbose(
                                    String.format(
                                            "Score is higher than the existing doc in tensor"
                                                + " result, replacing tensor hit. New score: %.7f,"
                                                + " New id is: %s.",
                                            reciprocalRank, extractedDocId),
                                    verbose);

                            rrfScores.remove(extractedDocIdInTensor);
                            result.remove(docIdsToHitIds.get(extractedDocIdInTensor));
                            docIdsToHitIds.remove(extractedDocIdInTensor);

                            addHitToResult(
                                    hit,
                                    reciprocalRank,
                                    rrfScores,
                                    extractedDocId,
                                    docIdsToHitIds,
                                    result,
                                    "marqo__raw_lexical_score");

                        } else {
                            // Same or lower rank in lexical result, discard the lexical hit
                            logIfVerbose(
                                    String.format(
                                            "Score is lower than or equal to the existing doc in"
                                                    + " tensor result, discarding lexical hit."
                                                    + " Discarded score is %.7f, id is: %s.",
                                            reciprocalRank, extractedDocId),
                                    verbose);
                        }
                    }
                    // If extractedDocId.equals(extractedDocIdInTensor), it should have been handled
                    // by the first branch
                }

                logIfVerbose(String.format("Modified lexical hit at rank: %d", rank), verbose);
                logIfVerbose(hit.toString(), verbose);

                rank++;
            }
        }

        return result;
    }

    /**
     * Helper method to add a hit to the result with proper scoring and mapping.
     */
    private static void addHitToResult(
            Hit hit,
            Double reciprocalRank,
            HashMap<String, Double> rrfScores,
            String extractedDocId,
            HashMap<String, String> docIdsToHitIds,
            HitGroup result,
            String rawScoreFieldName) {
        hit.setField(
                rawScoreFieldName,
                hit.getRelevance().getScore()); // Encode raw score for Marqo debugging purposes
        hit.setRelevance(reciprocalRank); // Update score to be weighted RR
        // Map hit's score to its shortened doc ID
        rrfScores.put(extractedDocId, reciprocalRank);
        // Map hit's full URI to its shortened doc ID
        docIdsToHitIds.put(extractedDocId, hit.getId().toString());
        result.add(hit);
    }

    /**
     * Extracts the collapse field hash from a hit's match features.
     * Returns null if the collapse field hash is not present or if match features are null.
     */
    @VisibleForTesting
    Double extractCollapseFieldHash(Hit hit) {
        Object matchFeaturesObj = hit.getField("matchfeatures");
        if (matchFeaturesObj == null) {
            return null;
        }

        if (matchFeaturesObj instanceof FeatureData) {
            FeatureData matchFeatures = (FeatureData) matchFeaturesObj;
            return matchFeatures.getDouble("collapse_field_hash");
        }

        return null;
    }

    HitGroup collectErrorsFromResults(Result resultLexical, Result resultTensor, boolean verbose) {
        // Return errors if either result list has an error. Make sure all errors are returned.
        HitGroup combinedErrors = new HitGroup();
        logIfVerbose(
                String.format("Tensor Errors found: %s", resultTensor.hits().getError()), verbose);
        logIfVerbose(
                String.format("Lexical Errors found: %s", resultLexical.hits().getError()),
                verbose);
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

        // Apply global score modifiers and rerank when regular global weights and/or custom score
        // rerank weights are present
        Tensor queryMultWeightsGlobal =
                extractTensorRankFeature(query, addQueryWrapper(QUERY_INPUT_MULT_WEIGHTS_GLOBAL));
        Tensor queryAddWeightsGlobal =
                extractTensorRankFeature(query, addQueryWrapper(QUERY_INPUT_ADD_WEIGHTS_GLOBAL));
        Tensor customScoreMult =
                extractTensorRankFeature(
                        query, addQueryWrapper(QUERY_INPUT_CUSTOM_SCORE_MULT_WEIGHTS_GLOBAL));
        Tensor customScoreAdd =
                extractTensorRankFeature(
                        query, addQueryWrapper(QUERY_INPUT_CUSTOM_SCORE_ADD_WEIGHTS_GLOBAL));
        boolean hasGlobalWeights =
                (queryMultWeightsGlobal != null && !queryMultWeightsGlobal.isEmpty())
                        || (queryAddWeightsGlobal != null && !queryAddWeightsGlobal.isEmpty());
        boolean hasCustomScoreWeights =
                (customScoreMult != null && !customScoreMult.isEmpty())
                        || (customScoreAdd != null && !customScoreAdd.isEmpty());

        if (hasGlobalWeights || hasCustomScoreWeights) {
            logIfVerbose("Applying global score modifiers and reranking.", verbose);
            resultToRerank = applyGlobalScoreModifiers(resultToRerank, query, verbose);
        } else if (query.properties().getBoolean("marqo__recency_apply_in_global_ranking_phase")) {
            // Apply recency score without global score modifiers
            double addToScoreWeight =
                    query.getRanking()
                            .getFeatures()
                            .getDouble(addQueryWrapper("marqo__recency_add_to_score_weight"))
                            .orElse(0.0);

            for (Hit hit : resultToRerank.asList()) {
                FeatureData matchFeatures = (FeatureData) hit.getField("matchfeatures");
                if (matchFeatures == null) continue;

                double recencyScore = matchFeatures.getDouble("recency_score");
                double originalScore = hit.getRelevance().getScore();
                double modifiedScore;

                if (addToScoreWeight > 0) {
                    // Additive mode
                    modifiedScore = originalScore + (recencyScore * addToScoreWeight);
                } else {
                    // Multiplicative mode (default)
                    modifiedScore = originalScore * recencyScore;
                }

                hit.setRelevance(modifiedScore);
                logIfVerbose(
                        String.format(
                                "Applied recency score %.4f to score %.4f -> %.4f for hit %s",
                                recencyScore, originalScore, modifiedScore, hit.getId()),
                        verbose);
            }

        } else {
            logIfVerbose(
                    "No global weights found. Skipping applying global score modifiers. Recency is"
                            + " disabled.",
                    verbose);
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
     * Creates a probe lexical query for relevance cut-off.
     * This query is used to determine the number of relevant candidates based on the specified
     * relevance cutoff method and parameter.
     *
     * @param query The original query to base the probe query on.
     * @param probeDepth The number of hits to retrieve in the probe query.
     * @param verbose Whether to log detailed information about the created query.
     * @return A new Query object configured for probe lexical search.
     */
    Query createProbeLexialQuery(Query query, Integer probeDepth, boolean verbose) {
        // Use dedicated probe lexical YQL when set (no custom-score extra rank terms); else main
        // lexical.
        String probeLexicalYql = query.properties().getString("marqo__yql.lexical.probe", null);
        Query probeLexicalQuery;
        if (probeLexicalYql != null && !probeLexicalYql.isEmpty()) {
            probeLexicalQuery =
                    createSubQuery(
                            query,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            verbose,
                            probeLexicalYql);
        } else {
            probeLexicalQuery =
                    createSubQuery(
                            query,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            MARQO_SEARCH_METHOD_LEXICAL,
                            verbose);
        }

        // Overwrite the lexical score modifiers in the probe query
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__mult_weights_lexical)", Tensor.from("tensor(p{}):{}"));
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__add_weights_lexical)", Tensor.from("tensor(p{}):{}"));

        // Turn off recency to make sure we get the raw relevance score
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__recency_should_calculate_score)", 0.0);
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__recency_should_apply_score)", 0.0);

        probeLexicalQuery.setHits(probeDepth);
        probeLexicalQuery.setOffset(0);
        probeLexicalQuery.properties().set(QUERY_RERANK_COUNT, probeDepth);

        String currentYql = probeLexicalQuery.properties().getString("yql");
        String updatedYql = overwriteTargetHitsIfPresent(currentYql, probeDepth);
        probeLexicalQuery.properties().set("yql", updatedYql);

        logIfVerbose(
                String.format(
                        "Created probe lexical query as: %s", probeLexicalQuery.toDetailString()),
                verbose);
        return probeLexicalQuery;
    }

    /**
     * Extracts mapped Tensor Address from cell then adds it as key to rank features, with cell value as the value.
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
     * Extracts the current targetHits value from YQL string, if multiple targetHits are present, return the first one.
     *
     * @param yql The YQL string containing targetHits
     * @return The current targetHits value as integer
     * @throws RuntimeException if targetHits is not found or invalid
     */
    @VisibleForTesting
    int extractCurrentTargetHits(String yql) {
        Matcher matcher = TARGET_HITS_PATTERN.matcher(yql);

        if (!matcher.find()) {
            throw new RuntimeException(
                    "YQL does not contain targetHits clause, cannot extract it.");
        }

        try {
            return Integer.parseInt(matcher.group(2));
        } catch (NumberFormatException e) {
            throw new RuntimeException("Invalid targetHits value in YQL: " + matcher.group(2), e);
        }
    }

    /**
     * Extracts the current exploreAdditionalHits value from YQL string, if multiple exploreAdditionalHits are present,
     * return the first one.
     *
     * @param yql The YQL string containing hnsw.exploreAdditionalHits
     * @return The current exploreAdditionalHits value as integer
     * @throws RuntimeException if hnsw.exploreAdditionalHits is not found or invalid
     */
    @VisibleForTesting
    int extractCurrentExploreAdditionalHits(String yql) {
        Matcher matcher = HNSW_EXPLORE_ADDITIONAL_HITS_PATTERN.matcher(yql);
        if (!matcher.find()) {
            throw new RuntimeException(
                    "YQL does not contain hnsw.exploreAdditionalHits clause, cannot extract it.");
        }
        try {
            return Integer.parseInt(matcher.group(2));
        } catch (NumberFormatException e) {
            throw new RuntimeException(
                    "Invalid exploreAdditionalHits value in YQL: " + matcher.group(2), e);
        }
    }

    /**
     * Overwrites the targetHits and hnsw.exploreAdditionalHits in the YQL string.
     * @param yql The original YQL string containing targetHits and hnsw.exploreAdditionalHits.
     * @param newTargetHits The new targetHits value to set in the YQL string.
     * @param efSearch The efSearch value, which is used to calculate the new hnsw.exploreAdditionalHits
     * @return Updated YQL string with new targetHits and hnsw.exploreAdditionalHits values.
     */
    @VisibleForTesting
    String overwriteTargetHitsAndExploreAdditionalHits(
            String yql, int newTargetHits, int efSearch) {
        // Count targetHits occurrences
        long targetHitsCount = TARGET_HITS_PATTERN.matcher(yql).results().count();

        if (targetHitsCount == 0) {
            throw new RuntimeException(
                    "YQL does not contain targetHits clause, cannot overwrite it.");
        }

        // Count hnsw.exploreAdditionalHits occurrences
        long hnswCount = HNSW_EXPLORE_ADDITIONAL_HITS_PATTERN.matcher(yql).results().count();

        if (hnswCount == 0) {
            throw new RuntimeException(
                    "YQL does not contain hnsw.exploreAdditionalHits clause, but targetHits is"
                            + " present. Both parameters must be present together.");
        }

        if (targetHitsCount != hnswCount) {
            throw new RuntimeException(
                    "YQL contains "
                            + targetHitsCount
                            + " targetHits occurrences but "
                            + hnswCount
                            + " hnsw.exploreAdditionalHits occurrences. Both must have the same"
                            + " count.");
        }
        // Overwrite targetHits
        String updatedYql = overwriteTargetHitsIfPresent(yql, newTargetHits);

        // Also update hnsw.exploreAdditionalHits to max(efSearch - normalizedTargetHits, 0)
        // Note: overwriteTargetHit normalizes 0 to 1, so we must use the same normalized value here
        int newExploreAdditionalHits = Math.max(efSearch - Math.max(newTargetHits, 1), 0);
        updatedYql =
                HNSW_EXPLORE_ADDITIONAL_HITS_PATTERN
                        .matcher(updatedYql)
                        .replaceAll("$1" + newExploreAdditionalHits);

        return updatedYql;
    }

    /**
     * Overwrites the targetHits in the YQL string if it is present.
     * @param yql The original YQL string containing targetHits.
     * @param newTargetHits The new targetHits value to set in the YQL string.
     * @return Updated YQL string with new targetHits value if present, otherwise returns original YQL.
     */
    String overwriteTargetHitsIfPresent(String yql, Integer newTargetHits) {
        // Validate input
        if (newTargetHits < 0) {
            throw new RuntimeException("targetHits value must be positive, got: " + newTargetHits);
        }

        if (newTargetHits == 0) {
            newTargetHits = 1; // The newTargetHits could be 0 if the relevantCandidates is 0
        }

        long targetHitsCount = TARGET_HITS_PATTERN.matcher(yql).results().count();
        if (targetHitsCount == 0) {
            return yql;
        }
        return TARGET_HITS_PATTERN.matcher(yql).replaceAll("$1" + newTargetHits);
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
     * @param query The original query to base the sub-query on.
     * @param retrievalMethod The retrieval method to use for the sub-query
     * @param rankingMethod The ranking method to use for the sub-query
     * @param verbose Whether to log detailed information about the created sub-query.
     * @param exactQuery An YQL string to use instead of the retrieval method's YQL.
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

        // Extract and set recency timestamp key tensor cells
        String recencyTimestampKeyFeature = addQueryWrapper(QUERY_INPUT_RECENCY_TIMESTAMP_KEY);
        logIfVerbose(
                "Attempting to extract recency tensor: " + recencyTimestampKeyFeature, verbose);
        Tensor recencyTimestampKey = extractTensorRankFeature(query, recencyTimestampKeyFeature);
        if (recencyTimestampKey != null) {
            logIfVerbose(
                    "Successfully extracted recency timestamp key tensor: " + recencyTimestampKey,
                    verbose);
            Iterator<Cell> recencyCells = recencyTimestampKey.cellIterator();
            recencyCells.forEachRemaining(
                    (cell) -> addFieldToRankFeatures(cell, queryNew, verbose));
        } else {
            logIfVerbose("Recency timestamp key tensor is null - not present in query", verbose);
        }

        // Check applyToSubqueries flags and override recency if this subquery type shouldn't get it
        boolean applyRecencyToTensor =
                query.properties().getBoolean(QUERY_INPUT_RECENCY_APPLY_TO_TENSOR, true);
        boolean applyRecencyToLexical =
                query.properties().getBoolean(QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL, true);

        boolean shouldDisableRecency = false;
        if (retrievalMethod.equals(MARQO_SEARCH_METHOD_TENSOR) && !applyRecencyToTensor) {
            shouldDisableRecency = true;
        } else if (retrievalMethod.equals(MARQO_SEARCH_METHOD_LEXICAL) && !applyRecencyToLexical) {
            shouldDisableRecency = true;
        }

        if (shouldDisableRecency) {
            logIfVerbose(
                    String.format(
                            "Disabling recency for %s subquery based on applyToSubqueries",
                            retrievalMethod),
                    verbose);
            queryNew.getRanking()
                    .getFeatures()
                    .put(addQueryWrapper(QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE), 0.0);
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
        Matcher matcher = DOC_ID_PATTERN.matcher(fullPath);

        // Check if the pattern matches and extract the document ID
        if (matcher.find()) {
            return matcher.group(1); // Return the captured group (document ID)
        } else {
            throw new InternalException(
                    "Vespa doc ID could not be extracted from the full hit ID: " + fullPath + ".");
        }
    }

    /**
     * Returns the set of match feature names from hit match features. FeatureData is not a Map;
     * it exposes featureNames(). We use that so bm25(*) and closeness(field,*) keys are available
     * for custom score extraction and aggregates.
     */
    private static Set<String> getMatchFeatureKeys(FeatureData matchFeatures) {
        if (matchFeatures == null) {
            return Collections.emptySet();
        }
        return new HashSet<>(matchFeatures.featureNames());
    }

    /**
     * Extracts the custom score value for one key. Custom score reranking uses only
     * summary-features (no fallback to match-features). For closeness_retrieval_vector the
     * summary feature name is ranking_closeness_metric_<field_name>. For bm25 we use
     * bm25(marqo__lexical_<field>) per lexical field; aggregate = sum/max/avg over those.
     */
    @VisibleForTesting
    static Double extractCustomScoreForHit(
            FeatureData matchFeatures,
            String key,
            CustomScoreKey parsed,
            Set<String> matchFeatureKeys,
            FeatureData summaryFeatures) {
        return new HybridSearcher()
                .extractCustomScoreForHit(
                        matchFeatures, key, parsed, matchFeatureKeys, summaryFeatures, null, false);
    }

    /**
     * Same as 5-arg but with optional keyForLog and verbose for logging (aggregation and read).
     */
    Double extractCustomScoreForHit(
            FeatureData matchFeatures,
            String key,
            CustomScoreKey parsed,
            Set<String> matchFeatureKeys,
            FeatureData summaryFeatures,
            String keyForLog,
            boolean verbose) {
        if (parsed == null) {
            return null;
        }
        if ("bm25".equals(parsed.scoreType)) {
            if (summaryFeatures == null) return null;
            if (parsed.aggregateType != null) {
                return aggregateFromSummaryFeatures(
                        summaryFeatures,
                        name -> name.startsWith("bm25(") && name.endsWith(")"),
                        parsed.aggregateType,
                        keyForLog,
                        "bm25Values",
                        false,
                        verbose);
            }
            String featName = CustomScoreKey.bm25SummaryFeatureName(parsed.fieldName);
            return getSingleFieldScoreWithLog(summaryFeatures, featName, keyForLog, verbose);
        }
        if ("closeness_retrieval_vector".equals(parsed.scoreType)) {
            if (summaryFeatures == null) return null;
            if (parsed.aggregateType != null) {
                // Same pattern as BM25: iterate summary-features. Treat null/NaN as 0 so all
                // index tensor fields contribute (schema lists all ranking_closeness_metric_*).
                return aggregateFromSummaryFeatures(
                        summaryFeatures,
                        name -> name.startsWith("ranking_closeness_metric_"),
                        parsed.aggregateType,
                        keyForLog,
                        "closenessValues",
                        true,
                        verbose);
            }
            String featName = "ranking_closeness_metric_" + parsed.fieldName;
            return getSingleFieldScoreWithLog(summaryFeatures, featName, keyForLog, verbose);
        }
        return null;
    }

    /**
     * Gets a double from FeatureData. Summary-features may be scalar (e.g. bm25, ranking_closeness_metric_*)
     * or tensor (e.g. tensor(float)(p{})); for tensors, returns the sum of all cells (single cell =
     * that value). Schema emits ranking_closeness_metric_* as scalar via reduce(..., sum).
     */
    private static Double getFeatureDouble(FeatureData data, String name) {
        if (data == null || name == null) return null;
        try {
            return data.getDouble(name);
        } catch (IllegalStateException e) {
            Tensor t = data.getTensor(name);
            if (t == null) return null;
            double sum = 0;
            for (Iterator<Cell> it = t.cellIterator(); it.hasNext(); ) {
                sum += it.next().getValue();
            }
            return Double.isFinite(sum) ? sum : null;
        }
    }

    /** Reads one summary-feature value and optionally logs; used for single-field custom score keys. */
    private Double getSingleFieldScoreWithLog(
            FeatureData summaryFeatures, String featureName, String keyForLog, boolean verbose) {
        Double score = featureName != null ? getFeatureDouble(summaryFeatures, featureName) : null;
        if (keyForLog != null && score != null) {
            logIfVerbose(
                    "[CustomScoreRerank] read from summary-features key="
                            + keyForLog
                            + " featureName="
                            + featureName
                            + " value="
                            + score,
                    verbose);
        }
        return score;
    }

    /**
     * Collects summary-feature values whose names pass the filter, aggregates them (sum/max/avg),
     * and optionally logs. Used for both BM25 and closeness_retrieval_vector aggregate keys.
     *
     * @param useZeroForMissing when true, treat null/NaN as 0.0 so every listed summary-feature
     *     contributes (e.g. closeness over all index tensor fields; schema lists all
     *     ranking_closeness_metric_*). When false, skip null/NaN (BM25 behavior).
     */
    private Double aggregateFromSummaryFeatures(
            FeatureData summaryFeatures,
            Predicate<String> nameFilter,
            String aggregateType,
            String keyForLog,
            String logLabel,
            boolean useZeroForMissing,
            boolean verbose) {
        List<Double> values = new ArrayList<>();
        for (String name : summaryFeatures.featureNames()) {
            if (nameFilter.test(name)) {
                Double v = getFeatureDouble(summaryFeatures, name);
                if (useZeroForMissing) {
                    values.add((v != null && !Double.isNaN(v)) ? v : 0.0);
                } else if (v != null && !Double.isNaN(v)) {
                    values.add(v);
                }
            }
        }
        Double result = aggregateValues(values, aggregateType);
        if (keyForLog != null && result != null) {
            logIfVerbose(
                    "[CustomScoreRerank] aggregation key="
                            + keyForLog
                            + " aggregateType="
                            + aggregateType
                            + " "
                            + logLabel
                            + "="
                            + values
                            + " result="
                            + result,
                    verbose);
        }
        return result;
    }

    private static Double aggregateValues(List<Double> values, String aggregateType) {
        if (values.isEmpty()) {
            return null;
        }
        return switch (aggregateType) {
            case "sum" -> values.stream().mapToDouble(Double::doubleValue).sum();
            case "max" -> values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
            case "avg" ->
                    values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            default -> null;
        };
    }

    /**
     * Normalizes a value by dividing by the maximum (value / max). The highest value maps to 1.0;
     * all others map to their proportion of the max. If max is zero, NaN, or negative, returns 1.0
     * to avoid division by zero. Used for both BM25 and closeness custom score keys.
     */
    @VisibleForTesting
    static double normalizeByMax(double value, double max) {
        if (max <= 0 || Double.isNaN(max) || !Double.isFinite(max)) {
            return 1.0;
        }
        return value / max;
    }

    /**
     * Returns weight * normalizedScore for one custom-score cell, or null if the score is missing
     * or the key is invalid. Scores are normalized by dividing by the max value across hits
     * (value / max) using maxPerKey. Custom score values are read only from summaryFeatures.
     * Tensor keys are always in canonical form (e.g. closeness_retrieval_vector_sum, bm25_sum)
     * without the marqo__score_ prefix, as set by Python when building the query.
     */
    private Double getWeightedNormalizedScoreForCell(
            Cell cell,
            FeatureData hitMatchFeatures,
            Set<String> matchFeatureKeys,
            Map<String, Double> maxPerKey,
            FeatureData summaryFeatures,
            Logger logger,
            boolean verbose) {
        String key = cell.getKey().label(0);
        CustomScoreKey parsed = CustomScoreKey.parseCustomScoreKey(key);
        if (parsed == null) return null;
        Double score =
                extractCustomScoreForHit(
                        hitMatchFeatures,
                        key,
                        parsed,
                        matchFeatureKeys,
                        summaryFeatures,
                        key,
                        verbose);
        if (score == null || Double.isNaN(score)) return null;
        double weight = cell.getValue().doubleValue();
        double normalizedScore = score;
        Double maxVal = maxPerKey != null ? maxPerKey.get(key) : null;
        if (maxVal != null) {
            normalizedScore = normalizeByMax(score, maxVal);
        }
        double modifierValue = weight * normalizedScore;
        logIfVerbose(
                "[CustomScoreRerank] apply modifier key="
                        + key
                        + " weight="
                        + weight
                        + " scoreValue="
                        + score
                        + " normalizedScore="
                        + normalizedScore
                        + " modifierValue="
                        + modifierValue,
                verbose);
        return modifierValue;
    }

    /**
     * Applies custom score rerank weights to add and mult modifiers: for each key in add weights,
     * adds (weight * normalizedScore) to addModifier; for each key in mult weights, multiplies
     * multModifier by (weight * normalizedScore). BM25 and closeness_retrieval_vector scores are
     * normalized by dividing by the max value across hits (value / max).
     */
    private void applyCustomScoreContributions(
            Double addModifier,
            Double multModifier,
            FeatureData hitMatchFeatures,
            Set<String> matchFeatureKeys,
            Tensor customAddWeights,
            Tensor customMultWeights,
            Map<String, Double> maxPerKey,
            double[] outAdd,
            double[] outMult,
            boolean verbose,
            FeatureData summaryFeatures) {
        double add = addModifier != null ? addModifier : 0.0;
        double mult = multModifier != null ? multModifier : 1.0;

        if (customAddWeights != null && !customAddWeights.isEmpty()) {
            for (Iterator<Cell> it = customAddWeights.cellIterator(); it.hasNext(); ) {
                Double contrib =
                        getWeightedNormalizedScoreForCell(
                                it.next(),
                                hitMatchFeatures,
                                matchFeatureKeys,
                                maxPerKey,
                                summaryFeatures,
                                logger,
                                verbose);
                if (contrib != null) add += contrib;
            }
        }

        if (customMultWeights != null && !customMultWeights.isEmpty()) {
            for (Iterator<Cell> it = customMultWeights.cellIterator(); it.hasNext(); ) {
                Double contrib =
                        getWeightedNormalizedScoreForCell(
                                it.next(),
                                hitMatchFeatures,
                                matchFeatureKeys,
                                maxPerKey,
                                summaryFeatures,
                                logger,
                                verbose);
                if (contrib != null) mult *= contrib;
            }
        }

        outAdd[0] = add;
        outMult[0] = mult;
    }

    /**
     * Compute per-key max value across hits for all custom score keys (BM25 and
     * closeness_retrieval_vector) present in add/mult weight tensors. Used for divide-by-max
     * normalization (value / max). Tensor keys are always in canonical form (e.g. bm25_sum,
     * closeness_retrieval_vector_sum). For aggregate keys, the value per hit is the aggregate
     * (sum/max/avg); max is taken over those values, so normalization is after aggregation.
     */
    @VisibleForTesting
    Map<String, Double> computeMaxPerKey(
            HitGroup hits, Tensor customAddWeights, Tensor customMultWeights) {
        Set<String> keys = new HashSet<>();
        if (customAddWeights != null) {
            for (Iterator<Cell> it = customAddWeights.cellIterator(); it.hasNext(); ) {
                String key = it.next().getKey().label(0);
                if (CustomScoreKey.parseCustomScoreKey(key) != null) keys.add(key);
            }
        }
        if (customMultWeights != null) {
            for (Iterator<Cell> it = customMultWeights.cellIterator(); it.hasNext(); ) {
                String key = it.next().getKey().label(0);
                if (CustomScoreKey.parseCustomScoreKey(key) != null) keys.add(key);
            }
        }
        Map<String, Double> result = new HashMap<>();
        for (String key : keys) {
            CustomScoreKey parsed = CustomScoreKey.parseCustomScoreKey(key);
            if (parsed == null) continue;
            double max = Double.NEGATIVE_INFINITY;
            for (Hit hit : hits) {
                FeatureData summaryFeatures = getSummaryFeaturesForHit(hit);
                if (summaryFeatures == null) continue;
                Double v =
                        extractCustomScoreForHit(
                                null,
                                key,
                                parsed,
                                Collections.emptySet(),
                                summaryFeatures,
                                null,
                                false);
                if (v != null && !Double.isNaN(v)) {
                    max = Math.max(max, v);
                }
            }
            if (Double.isFinite(max) && max > 0) {
                result.put(key, max);
            }
        }
        return result;
    }

    /**
     * Apply global score modifiers to the hit group. Modifies hit scores, does not add/remove hits.
     * @param hits The hit group to apply global score modifiers to.
     * @param query The query to check recency mode.
     * @param verbose Whether to log detailed information about the score modification process.
     */
    HitGroup applyGlobalScoreModifiers(HitGroup hits, Query query, boolean verbose) {
        FeatureData hitMatchFeatures;
        Double mult_modifier, add_modifier, original_score, modified_score, recencyScore;
        boolean exposePreRerankScore =
                query.properties().getBoolean("marqo__expose_pre_rerank_score", false);
        if (hits.size() == 0) {
            logIfVerbose("No hits to apply score modifiers to. Returning.", verbose);
            return hits;
        }

        Tensor customAddWeights =
                extractTensorRankFeature(
                        query, addQueryWrapper(QUERY_INPUT_CUSTOM_SCORE_ADD_WEIGHTS_GLOBAL));
        Tensor customMultWeights =
                extractTensorRankFeature(
                        query, addQueryWrapper(QUERY_INPUT_CUSTOM_SCORE_MULT_WEIGHTS_GLOBAL));
        boolean hasCustomScores =
                (customAddWeights != null && !customAddWeights.isEmpty())
                        || (customMultWeights != null && !customMultWeights.isEmpty());

        logIfVerbose(
                "[CustomScoreRerank] customAddWeights="
                        + (customAddWeights == null ? "null" : "size=" + customAddWeights.size())
                        + " customMultWeights="
                        + (customMultWeights == null ? "null" : "size=" + customMultWeights.size())
                        + " hasCustomScores="
                        + hasCustomScores,
                verbose);
        if (hasCustomScores && customAddWeights != null && !customAddWeights.isEmpty()) {
            for (Iterator<Cell> it = customAddWeights.cellIterator(); it.hasNext(); ) {
                Cell cell = it.next();
                String key = cell.getKey().label(0);
                double weight = cell.getValue().doubleValue();
                CustomScoreKey parsed = CustomScoreKey.parseCustomScoreKey(key);
                if (parsed != null) {
                    logIfVerbose(
                            "[CustomScoreRerank] unpack add_to_score key="
                                    + key
                                    + " scoreType="
                                    + parsed.scoreType
                                    + " field="
                                    + (parsed.fieldName != null ? parsed.fieldName : "n/a")
                                    + " aggregateType="
                                    + (parsed.aggregateType != null ? parsed.aggregateType : "n/a")
                                    + " weight="
                                    + weight,
                            verbose);
                }
            }
        }
        if (hasCustomScores && customMultWeights != null && !customMultWeights.isEmpty()) {
            for (Iterator<Cell> it = customMultWeights.cellIterator(); it.hasNext(); ) {
                Cell cell = it.next();
                String key = cell.getKey().label(0);
                double weight = cell.getValue().doubleValue();
                CustomScoreKey parsed = CustomScoreKey.parseCustomScoreKey(key);
                if (parsed != null) {
                    logIfVerbose(
                            "[CustomScoreRerank] unpack multiply_score_by key="
                                    + key
                                    + " scoreType="
                                    + parsed.scoreType
                                    + " field="
                                    + (parsed.fieldName != null ? parsed.fieldName : "n/a")
                                    + " aggregateType="
                                    + (parsed.aggregateType != null ? parsed.aggregateType : "n/a")
                                    + " weight="
                                    + weight,
                            verbose);
                }
            }
        }

        Set<String> allMatchFeatureKeys = new HashSet<>();
        Map<String, Double> maxPerKey = new HashMap<>();
        if (hasCustomScores) {
            Hit firstHit = hits.get(0);
            FeatureData firstMf = (FeatureData) firstHit.getField("matchfeatures");
            if (firstMf != null) {
                allMatchFeatureKeys = getMatchFeatureKeys(firstMf);
            }
            logIfVerbose(
                    "[CustomScoreRerank] match feature keys from first hit (count="
                            + allMatchFeatureKeys.size()
                            + "): "
                            + allMatchFeatureKeys,
                    verbose);
            /* Compute max score for each key for divide-by-max normalization */
            maxPerKey = computeMaxPerKey(hits, customAddWeights, customMultWeights);
        }

        boolean applyRecency =
                query.properties().getBoolean("marqo__recency_apply_in_global_ranking_phase");

        double addToScoreWeight =
                query.getRanking()
                        .getFeatures()
                        .getDouble(addQueryWrapper("marqo__recency_add_to_score_weight"))
                        .orElse(0.0);

        int hitIndex = 0;
        for (Hit hit : hits) {
            logIfVerbose("Applying score modifiers to hit: " + hit.getId(), verbose);
            hitMatchFeatures = (FeatureData) hit.getField("matchfeatures");
            if (hitMatchFeatures != null) {
                mult_modifier = hitMatchFeatures.getDouble("global_mult_modifier");
                add_modifier = hitMatchFeatures.getDouble("global_add_modifier");
                recencyScore = applyRecency ? hitMatchFeatures.getDouble("recency_score") : 1.0;

                if (mult_modifier != null && add_modifier != null) {
                    double effectiveMult = mult_modifier;
                    double effectiveAdd = add_modifier;
                    if (hasCustomScores) {
                        Set<String> hitMatchFeatureKeys = getMatchFeatureKeys(hitMatchFeatures);
                        if (hitMatchFeatureKeys.isEmpty()) {
                            hitMatchFeatureKeys = allMatchFeatureKeys;
                        }
                        FeatureData summaryFeatures = getSummaryFeaturesForHit(hit);
                        double[] outAdd = new double[1];
                        double[] outMult = new double[1];
                        applyCustomScoreContributions(
                                add_modifier,
                                mult_modifier,
                                hitMatchFeatures,
                                hitMatchFeatureKeys,
                                customAddWeights,
                                customMultWeights,
                                maxPerKey,
                                outAdd,
                                outMult,
                                verbose,
                                summaryFeatures);
                        effectiveAdd = outAdd[0];
                        effectiveMult = outMult[0];
                        if (hitIndex == 0) {
                            logIfVerbose(
                                    String.format(
                                            "[CustomScoreRerank] first hit add_modifier=%.5f"
                                                    + " outAdd=%.5f outMult=%.5f maxPerKeySize=%d",
                                            add_modifier, outAdd[0], outMult[0], maxPerKey.size()),
                                    verbose);
                        }
                    }

                    original_score = hit.getRelevance().getScore();
                    // Expose pre-rerank score only when explicitly requested, so old Marqo versions
                    // that don't know this field can still parse results after a rollback.
                    if (exposePreRerankScore) {
                        hit.setField(MARQO_PRE_RERANK_SCORE, original_score);
                    }
                    double baseScore = original_score * effectiveMult + effectiveAdd;

                    if (!applyRecency) {
                        modified_score = baseScore;
                    } else if (addToScoreWeight > 0) {
                        modified_score =
                                baseScore
                                        + (recencyScore != null ? recencyScore : 1.0)
                                                * addToScoreWeight;
                    } else {
                        modified_score = baseScore * (recencyScore != null ? recencyScore : 1.0);
                    }

                    logIfVerbose(
                            String.format(
                                    "Original score: %.7f, mult modifier: %.5f, add modifier: %.5f,"
                                        + " recency score: %.5f, addToScoreWeight: %.5f, Modified"
                                        + " score: %.7f",
                                    original_score,
                                    effectiveMult,
                                    effectiveAdd,
                                    recencyScore != null ? recencyScore : 1.0,
                                    addToScoreWeight,
                                    modified_score),
                            verbose);
                    if (hasCustomScores && hit.getRelevance().getScore() != modified_score) {
                        logIfVerbose(
                                String.format(
                                        "[CustomScoreRerank] hit=%s original=%.7f effectiveAdd=%.5f"
                                                + " effectiveMult=%.5f modified=%.7f",
                                        hit.getId(),
                                        original_score,
                                        effectiveAdd,
                                        effectiveMult,
                                        modified_score),
                                verbose);
                    }
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
            hitIndex++;
        }
        return hits;
    }

    /**
     * Extracts recency score from match features and sets it as a field on each hit.
     * This runs independently of global score modifiers, but only when recency is enabled.
     *
     * @param hits The hits to process
     * @param query The query to check if recency is enabled
     * @param verbose Whether to log verbose messages
     * @return The processed hits with recency_score field set
     */
    HitGroup extractRecencyScore(HitGroup hits, Query query, boolean verbose) {
        boolean recencyEnabled = query.properties().getBoolean("marqo__recency_enabled");
        if (!recencyEnabled) {
            logIfVerbose("Recency is not enabled. Skipping recency score extraction.", verbose);
            return hits;
        }

        if (hits.size() == 0) {
            logIfVerbose("No hits to extract recency score from. Returning.", verbose);
            return hits;
        }

        logIfVerbose("Recency is enabled. Extracting recency score from match features.", verbose);

        for (Hit hit : hits) {
            // Extract match features
            FeatureData hitMatchFeatures = (FeatureData) hit.getField("matchfeatures");
            if (hitMatchFeatures != null) {
                // Extract recency score if present and set as field
                Double recency_score = hitMatchFeatures.getDouble("recency_score");
                if (recency_score != null) {
                    hit.setField("marqo__recency_score", recency_score);
                    logIfVerbose(
                            String.format(
                                    "Extracted recency score for hit %s: %.5f",
                                    hit.getId(), recency_score),
                            verbose);
                }
            }
        }

        return hits;
    }

    /**
     * Detects the cutoff count for relevance filtering based on the specified method
     * @param probeCandidates The lexical search results to analyze
     * @param cutoffMethodString The method to use for cutoff detection
     * @param relevanceCutoffParameter The parameter for the cutoff method. It is unified across all methods.
     * @param verbose Whether to log verbose information
     * @return The number of relevant results to keep
     */
    @VisibleForTesting
    int detectCutoffCount(
            HitGroup probeCandidates,
            String cutoffMethodString,
            Double relevanceCutoffParameter,
            boolean verbose) {
        List<Hit> lexicalHits = new ArrayList<>(probeCandidates.asList());

        if (lexicalHits.isEmpty()) {
            logIfVerbose("No lexical hits found in probe pool, returning 0", verbose);
            return 0;
        }

        RelevanceCutoffMethod cutoffMethod;
        try {
            cutoffMethod = RelevanceCutoffMethod.fromString(cutoffMethodString);
        } catch (IllegalArgumentException e) {
            throw new RuntimeException("Unknown relevance cutoff method: " + cutoffMethodString);
        }

        double[] probeLexicalScores =
                lexicalHits.stream().mapToDouble(hit -> hit.getRelevance().getScore()).toArray();

        switch (cutoffMethod) {
            case GAP_DETECTION -> {
                logIfVerbose("Using gapDetection method for relevance cutoff", verbose);
                // Find the elbow point in the lexical hits
                double maxDelta = -1.0;
                int bestIndex = lexicalHits.size(); // default: keep all
                for (int i = 0; i < lexicalHits.size() - 1; i++) {
                    double score1 = probeLexicalScores[i];
                    double score2 = probeLexicalScores[i + 1];
                    double delta = score1 - score2;

                    if (delta > maxDelta) {
                        maxDelta = delta;
                        bestIndex = i + 1;
                    }
                }
                return bestIndex;
            }
            case MEAN_STD_DEV -> {
                logIfVerbose("Using normalFit method for relevance cutoff", verbose);
                DoubleStatistics stats =
                        DoubleStatistics.of(
                                EnumSet.of(Statistic.MEAN, Statistic.STANDARD_DEVIATION),
                                probeLexicalScores);
                double mean = stats.getAsDouble(Statistic.MEAN);
                double stdDev = stats.getAsDouble(Statistic.STANDARD_DEVIATION);
                double threshold = mean + (relevanceCutoffParameter * stdDev);
                return countGreaterOrEqual(probeLexicalScores, threshold);
            }
            case RELATIVE_MAX_SCORE -> {
                logIfVerbose("Using softCodedScoreCut method for relevance cutoff", verbose);
                double topScore = lexicalHits.get(0).getRelevance().getScore();
                double dynamicThreshold = topScore * relevanceCutoffParameter;
                return countGreaterOrEqual(probeLexicalScores, dynamicThreshold);
            }
            default -> {
                throw new RuntimeException("Unknown relevance cutoff method: " + cutoffMethod);
            }
        }
    }

    /**
     * Returns the number of elements in a descending-sorted array
     * that are greater than or equal to the given threshold.
     *
     * @param descSorted A descending-sorted array of scores.
     * @param threshold The threshold value to compare against.
     */
    @VisibleForTesting
    static int countGreaterOrEqual(double[] descSorted, double threshold) {
        int low = 0, high = descSorted.length;
        while (low < high) {
            int mid = (low + high) >>> 1;
            if (descSorted[mid] >= threshold) {
                // move low up
                low = mid + 1;
            } else {
                // shrink high down
                high = mid;
            }
        }
        return low;
    }
}
