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
    private static String MARQO_SEARCH_METHOD_LEXICAL = "lexical";
    private static String MARQO_SEARCH_METHOD_TENSOR = "tensor";
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

    @VisibleForTesting
    @JsonInclude(Include.NON_NULL)
    record MarqoMetadataFields(
            Integer sortCandidates, Integer probeCandidates, Integer relevantCandidates)
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

        // trimAndFuse: retrieve [offset, offset+limit) from both lexical and tensor, fuse, apply
        // GSM, trim to [0, limit)
        // fuseAndTrim: retrieve [0, offset+limit) from both lexical and tensor, fuse, apply GSM,
        // trim to [offset, offset+limit)
        // fuseAndExclude: retrieve [0, offset+limit) from both lexical and tensor, fuse; fuse again
        // with [0, offset) to deduce last pages, exclude last pages, apply GSM, trim to [0, limit)
        // fuseAndExcludeWithExtraTensorSearch: same as above, but use result of an extra tensor
        // search [0, offset) when deducing last page
        // (TBD) excludeAndFuseWithState: retrieve [0, offset+limit) from both lexical and tensor,
        // remove all docs in state from both result, fuse, apply GSM, trim to [0, limit)
        String paginationMode =
                query.properties().getString("marqo__hybrid.paginationMode", "trimAndFuse");
        boolean shouldRetrieveTensorLastPage =
                "fuseAndExcludeWithExtraTensorSearch".equals(paginationMode);
        boolean shouldExcludeLastPages = paginationMode.startsWith("fuseAndExclude");

        // Relevance Cut-off Parameters
        String relevanceCutoffMethod =
                query.properties().getString("marqo__hybrid.relevanceCutoff.method", null);
        Integer relevanceCutoffProbeDepth =
                query.properties().getInteger("marqo__hybrid.relevanceCutoff.probeDepth", null);
        Double relevanceCutoffParameter =
                readRelevanceCutoffParameter(query, relevanceCutoffMethod);

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

        List<Future<Result>> futureFacets =
                getFacetsFutureList(query, execution, verbose, collapse);

        // --- Begin relevance cut-off handling ---
        // Execute probe lexical search for relevance cut-off if parameters are provided
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

        // --- Update the query hits, offset and targetHits, if sort or relevance cut-off is used
        boolean isRelevanceCutoffMethodEnabled = relevanceCutoffMethod != null;
        boolean isSortByEnabled = sortByFields != null;
        boolean isDisjunctionSearch = retrievalMethod.equals("disjunction");

        query =
                updateQueryHitsOffsetsAndTargetHits(
                        query,
                        relevantCandidates,
                        sortByMinSortCandidates,
                        isRelevanceCutoffMethodEnabled,
                        isSortByEnabled,
                        isDisjunctionSearch && !"trimAndFuse".equals(paginationMode));

        HitGroup hitsForPostProcessing;
        if (isDisjunctionSearch) {
            Result resultLexical, resultTensor, resultTensorLastPage = null;
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

            Future<Result> futureTensorLastPage = null;
            if (shouldRetrieveTensorLastPage && offset > 0) {
                String tensorYQL = queryTensor.properties().getString("yql", "");
                int currentTensorTargetHits = extractCurrentTargetHits(tensorYQL);

                // TODO ideally, we should pass in the targetHit override, and whether it's fixed
                if (currentTensorTargetHits > offset + limit) {
                    // the targetHit is probably overridden, the tensor search result should be
                    // stable
                    logIfVerbose(
                            String.format(
                                    "TargetHits=%d, larger than (offset+limit)=%d. It's probably"
                                            + " overridden, no need to do another tensor retrieval",
                                    currentTensorTargetHits, offset + limit),
                            verbose);
                } else {
                    int currentExploreAdditionalHits =
                            extractCurrentExploreAdditionalHits(tensorYQL);
                    int efSearch = currentTensorTargetHits + currentExploreAdditionalHits;
                    String tensorYQLUpdated = overwriteTargetHits(tensorYQL, offset, efSearch);

                    Query queryTensorLastPage = queryTensor.clone();
                    queryTensorLastPage.setHits(offset);
                    queryTensorLastPage.properties().set("yql", tensorYQLUpdated);

                    logIfVerbose(
                            String.format(
                                    "Get last page tensor result with limit %d and YQL: %s",
                                    offset, tensorYQLUpdated),
                            verbose);
                    futureTensorLastPage =
                            new AsyncExecution(execution).search(queryTensorLastPage);
                }
            }

            try {
                resultLexical = futureLexical.get(timeout, TimeUnit.MILLISECONDS);
                resultTensor = futureTensor.get(timeout, TimeUnit.MILLISECONDS);

                if (shouldRetrieveTensorLastPage && futureTensorLastPage != null) {
                    resultTensorLastPage = futureTensorLastPage.get(timeout, TimeUnit.MILLISECONDS);
                }
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

            String resultSummary =
                    String.format(
                            "LEXICAL RESULTS: %s || TENSOR RESULTS: %s",
                            resultLexical, resultTensor);
            if (resultTensorLastPage != null) {
                resultSummary += " TENSOR RESULTS LAST PAGE: " + resultTensorLastPage;
            }
            logIfVerbose(resultSummary, verbose);

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

                if (shouldExcludeLastPages && offset > 0) {
                    // Simulate previous page result.
                    // TODO Also consider the case of pinned docs and excluded docs

                    HitGroup tensorHitsPreviousPages;

                    if (resultTensorLastPage != null) {
                        logIfVerbose(
                                String.format(
                                        "Offset is %d, Got previous page result from extra tensor"
                                                + " search: ",
                                        offset),
                                verbose);
                        tensorHitsPreviousPages = resultTensorLastPage.hits();
                        logHitGroup(tensorHitsPreviousPages, verbose);
                    } else {
                        logIfVerbose(
                                String.format(
                                        "Offset is %d, Simulate previous page result: ", offset),
                                verbose);

                        tensorHitsPreviousPages = resultTensor.hits().clone();
                        tensorHitsPreviousPages.trim(0, offset);
                        logIfVerbose(
                                String.format("Tensor Hit Group is trimmed to %d", offset),
                                verbose);
                        logHitGroup(tensorHitsPreviousPages, verbose);
                    }

                    HitGroup lexicalHitsPreviousPages = resultLexical.hits().clone();
                    lexicalHitsPreviousPages.trim(0, offset);
                    logIfVerbose(
                            String.format("Lexical Hit Group is trimmed to %d", offset), verbose);
                    logHitGroup(lexicalHitsPreviousPages, verbose);

                    // Apply RRF and global ranking on simulated previous page result
                    HitGroup candidateHitsOfPreviousPages =
                            rrf(
                                    tensorHitsPreviousPages,
                                    lexicalHitsPreviousPages,
                                    rrf_k,
                                    alpha,
                                    verbose,
                                    collapse);
                    logIfVerbose("Simulated previous page candidates (after rrf): ", verbose);
                    logHitGroup(candidateHitsOfPreviousPages, verbose);

                    Integer rerankDepthLastPage = null;
                    if (rerankDepthGlobal != null) {
                        // deduce the rerankDepthGlobal used by last page
                        // TODO this is a best effort, since rerankDepthGlobal can be fixed, or
                        // multiply (offset+limit) by a factor. We might need a smarter way to do
                        // this.
                        rerankDepthLastPage = Math.max(0, offset - limit) + rerankDepthGlobal;
                    }

                    HitGroup hitsOfPreviousPages =
                            postProcessResults(
                                    candidateHitsOfPreviousPages,
                                    query,
                                    rerankDepthLastPage,
                                    offset,
                                    0,
                                    false,
                                    verbose);
                    logIfVerbose(
                            "Simulated previous page result (after global ranking and trimming): ",
                            verbose);
                    logHitGroup(hitsOfPreviousPages, verbose);

                    // Remove previous page results
                    // TODO handle collapse fields
                    Set<String> docIdsOfPreviousPages = new HashSet<>();
                    for (Hit hit : hitsOfPreviousPages) {
                        // id from different replicas are different, we need to trim replica to get
                        // the internal id
                        String extractedDocId = extractDocIdFromHitId(hit.getId().toString());
                        docIdsOfPreviousPages.add(extractedDocId);
                    }

                    if (!docIdsOfPreviousPages.isEmpty()) {
                        HitGroup hitsForPostProcessingWithoutPreviousPages = new HitGroup();
                        for (Hit hit : hitsForPostProcessing.asList()) {
                            String extractedDocId = extractDocIdFromHitId(hit.getId().toString());
                            if (!docIdsOfPreviousPages.contains(extractedDocId)) {
                                hitsForPostProcessingWithoutPreviousPages.add(hit);
                            }
                        }
                        // TODO check if there's any other metadata to copy over
                        hitsForPostProcessing = hitsForPostProcessingWithoutPreviousPages;
                    }

                    logIfVerbose(
                            "hitsForPostProcessing after removing previous page result: ", verbose);
                    logHitGroup(hitsForPostProcessing, verbose);
                }

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

        // Determine post-processing mode based on query parameters
        HitGroup processedHits;
        Integer sortCandidates = null;
        if (sortByFields != null) {
            // If sortBy is set, we will sort the hits after post-processing
            processedHits =
                    postProcessBySort(
                            hitsForPostProcessing, sortByFields, sortBySortDepth, limit, offset);
            sortCandidates = hitsForPostProcessing.size();
        } else {
            // If sortBy is not set, we use the default post-processing
            // when we do disjunction search or use relevance cutoff, the offset is set to 0, so we
            // need to trim previous pages
            boolean needToTrimPreviousPages =
                    isDisjunctionSearch && "fuseAndTrim".equals(paginationMode)
                            || isRelevanceCutoffMethodEnabled;
            processedHits =
                    postProcessResults(
                            hitsForPostProcessing,
                            query,
                            rerankDepthGlobal,
                            limit,
                            offset,
                            needToTrimPreviousPages,
                            verbose);
        }

        if (!futureFacets.isEmpty()) {
            attachFacetsResult(futureFacets, timeout, processedHits, verbose);
        }

        MarqoMetadataFields marqoMetadataFields =
                new MarqoMetadataFields(sortCandidates, probeCandidates, relevantCandidates);

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
     * Updates the query hits, offsets, and targetHits based on the provided parameters.
     *
     * @param query                    the query to update.
     * @param relevantCandidates       the number of relevant candidates found in the probe search.
     * @param sortByMinSortCandidates  the minimum number of candidates required for sorting, from Marqo
     * @param isRelevanceCutoffEnabled whether relevance cutoff is enabled.
     * @param isSortByEnabled          whether sorting is enabled.
     * @param shouldResetOffsetLimit   whether we need to set offset to 0 and limit to offset+limit
     * @return The updated query with new hits, offsets, and targetHits.
     */
    public Query updateQueryHitsOffsetsAndTargetHits(
            Query query,
            Integer relevantCandidates,
            Integer sortByMinSortCandidates,
            boolean isRelevanceCutoffEnabled,
            boolean isSortByEnabled,
            boolean shouldResetOffsetLimit) {

        // Validate input parameters
        if (!isRelevanceCutoffEnabled && !isSortByEnabled) {
            if (shouldResetOffsetLimit) {
                query.setHits(query.getOffset() + query.getHits());
                query.setOffset(0);
            }

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
                newTensorTargetHits = Math.max(newHits, currentTensorTargetHits);
            }
        } else if (isRelevanceCutoffEnabled) {
            // Only relevance cut-off enabled:
            // - If relevantCandidates < limit+offset: reduce to relevantCandidates
            // - If relevantCandidates >= limit+offset: keep existing behavior (limit+offset)
            newHits = Math.min(relevantCandidates, (currentLimit + currentOffset));
            if (currentTensorTargetHits != null) {
                newTensorTargetHits = Math.min(newHits, currentTensorTargetHits);
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

        // Update tensor YQL targetHits if it exists
        if (currentTensorTargetHits != null
                && !Objects.equals(currentTensorTargetHits, newTensorTargetHits)) {
            int efSearch = currentTensorTargetHits + currentExploreAdditionalHits;
            String tensorYQLUpdated = overwriteTargetHits(tensorYQL, newTensorTargetHits, efSearch);
            query.properties().set("marqo__yql." + MARQO_SEARCH_METHOD_TENSOR, tensorYQLUpdated);
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
            boolean needToTrimPreviousPages,
            boolean verbose) {

        // Step 1: Split original hits into 2 lists: result to rerank and excess hits
        // Excess hits will not be reranked, and will be added back after reranking the other
        // results
        HitGroup resultToRerank = new HitGroup();
        HitGroup excessHits = new HitGroup();

        int idx = 0;
        // If rerank count global is not set, rerank all hits
        if (rerankDepthGlobal == null) {
            rerankDepthGlobal = hitsForPostProcessing.size();
        } else if (needToTrimPreviousPages) {
            // When previous pages are also in search result, we need to increase the original
            // rerankDepthGlobal to make sure global score modifiers are applied to current page
            rerankDepthGlobal = rerankDepthGlobal + offset;
        }
        // Calculate total hits needed: when needToTrimPreviousPages is true, we need offset + limit
        // hits
        int totalHitsNeeded = needToTrimPreviousPages ? offset + limit : limit;

        for (Hit hit : hitsForPostProcessing) {
            if (idx < rerankDepthGlobal) {
                resultToRerank.add(hit);
            } else if (idx < totalHitsNeeded) {
                // Total hits to process should be offset + limit when trimming previous pages
                excessHits.add(hit);
            } else {
                // Ignore all hits after totalHitsNeeded
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

        // Step 2: Apply global score modifiers and rerank
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

        // Step 3: Sort  TODO should only sort after applying global score modifiers
        resultToRerank.sort();

        logIfVerbose("Reranked result list (SORTED): ", verbose);
        logHitGroup(resultToRerank, verbose);

        // Step 4: Add excess hits if needed
        if (totalHitsNeeded > rerankDepthGlobal) {
            // Add excess hits to the end of reranked results then sort
            logIfVerbose(
                    String.format(
                            "Adding %d excess hits to the end of reranked results and sorting.",
                            excessHits.size()),
                    verbose);
            resultToRerank.addAll(excessHits.asList());
        }

        // Step 5: Final trim to limit (pagination was already applied if needed)
        if (needToTrimPreviousPages) {
            logIfVerbose(
                    String.format(
                            "Final trimming result list from offset %s and limit: %d",
                            offset, limit),
                    verbose);
            resultToRerank.trim(offset, limit);
        } else {
            logIfVerbose(String.format("Final trimming result list to limit: %d", limit), verbose);
            resultToRerank.trim(0, limit);
        }

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
        Query probeLexicalQuery =
                createSubQuery(
                        query, MARQO_SEARCH_METHOD_LEXICAL, MARQO_SEARCH_METHOD_LEXICAL, verbose);

        // Overwrite the lexical score modifiers in the probe query
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__mult_weights_lexical)", Tensor.from("tensor(p{}):{}"));
        probeLexicalQuery
                .getRanking()
                .getFeatures()
                .put("query(marqo__add_weights_lexical)", Tensor.from("tensor(p{}):{}"));

        probeLexicalQuery.setHits(probeDepth);
        probeLexicalQuery.setOffset(0);

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
    String overwriteTargetHits(String yql, int newTargetHits, int efSearch) {
        // Validate input
        if (newTargetHits < 0) {
            throw new RuntimeException("targetHits value must be positive, got: " + newTargetHits);
        }

        if (newTargetHits == 0) {
            // If targetHits is set to 0, we can set it to 1 to avoid error from Vespa
            newTargetHits = 1;
        }

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

        // Replace all targetHits occurrences
        String updatedYql = TARGET_HITS_PATTERN.matcher(yql).replaceAll("$1" + newTargetHits);

        // Also update hnsw.exploreAdditionalHits to max(efSearch - newTargetHits, 0)
        int newExploreAdditionalHits = Math.max(efSearch - newTargetHits, 0);
        updatedYql =
                HNSW_EXPLORE_ADDITIONAL_HITS_PATTERN
                        .matcher(updatedYql)
                        .replaceAll("$1" + newExploreAdditionalHits);

        return updatedYql;
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
     * Apply global score modifiers to the hit group. Modifies hit scores, does not add/remove hits.
     * @param hits The hit group to apply global score modifiers to.
     * @param verbose Whether to log detailed information about the score modification process.
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
