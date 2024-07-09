package ai.marqo.search;

import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import java.util.Optional;

class MarqoQuery {
    private static final String QUERY_INPUT_FIELDS_TO_RANK = "marqo__fields_to_rank";

    private RetrievalMethod retrievalMethod;
    private RankingMethod rankingMethod;
    private int rrfK;
    private double alpha;
    private int timeoutMillis;
    private Query query;

    private MarqoQuery(
            RetrievalMethod retrievalMethod,
            RankingMethod rankingMethod,
            int rrfK,
            double alpha,
            int timeoutMillis,
            Query query) {
        this.retrievalMethod = retrievalMethod;
        this.rankingMethod = rankingMethod;
        this.rrfK = rrfK;
        this.alpha = alpha;
        this.timeoutMillis = timeoutMillis;
        this.query = query;
    }

    public static MarqoQuery from(Query query) {
        MarqoQuery marqoQuery =
                new MarqoQuery(
                        RetrievalMethod.valueOf(
                                query.properties().getString("marqo__hybrid.retrievalMethod")),
                        RankingMethod.valueOf(
                                query.properties().getString("marqo__hybrid.rankingMethod")),
                        query.properties().getInteger("marqo__hybrid.rrf_k", 60),
                        query.properties().getDouble("marqo__hybrid.alpha", 0.5),
                        query.properties().getInteger("timeout", 1000),
                        query);
        query.trace(marqoQuery, 1);
        marqoQuery.validate();
        return marqoQuery;
    }

    public Query createSubQuery() {
        return createSubQuery(retrievalMethod, rankingMethod);
    }

    /**
     * Creates custom sub-query from the original query.
     * Clone original query, Update the following:
     * 'yql' (based on RETRIEVAL method)
     * 'ranking.profile'    (based on RANKING method)
     * 'ranking.features'
     *      fields to search  (based on ??? method)
     *      score modifiers (based on RANKING method)
     * @param retrievalMethod
     * @param rankingMethod
     */
    public Query createSubQuery(RetrievalMethod retrievalMethod, RankingMethod rankingMethod) {
        query.trace(
                String.format(
                        "Creating subquery with retrieval: %s, ranking: %s",
                        retrievalMethod, rankingMethod),
                2);

        // Extract relevant properties
        // YQL uses RETRIEVAL method
        String yqlNew = query.properties().getString("marqo__yql." + retrievalMethod, "");
        // Rank Profile uses RETRIEVAL + RANKING method
        String rankProfileNew =
                query.properties()
                        .getString("marqo__ranking." + retrievalMethod + "." + rankingMethod, "");

        // Log fetched properties
        query.trace(String.format("YQL %s found: %s", retrievalMethod, yqlNew), 2);
        query.trace(
                String.format(
                        "Rank Profile %s.%s found: %s",
                        retrievalMethod, rankingMethod, rankProfileNew),
                2);

        // Create New Subquery
        Query queryNew = query.clone();
        queryNew.properties().set("yql", yqlNew);

        // Set fields to rank (extract using RANKING method)
        String featureNameFieldsToRank =
                addQueryWrapper(QUERY_INPUT_FIELDS_TO_RANK + "_" + rankingMethod);
        query.trace("Using fields to rank from " + featureNameFieldsToRank, 2);

        extractTensorRankFeature(query, featureNameFieldsToRank)
                .cellIterator()
                .forEachRemaining((cell) -> addFieldToRankFeatures(cell, queryNew));

        // Set rank profile (using RANKING method)
        queryNew.getRanking().setProfile(rankProfileNew);

        query.trace(new VespaQueryDetails(queryNew), 2);

        return queryNew;
    }

    private void addFieldToRankFeatures(Tensor.Cell cell, Query query) {
        TensorAddress cellKey = cell.getKey();
        for (int i = 0; i < cellKey.size(); i++) {
            query.getRanking()
                    .getFeatures()
                    .put(addQueryWrapper(cellKey.label(i)), cell.getValue());
        }
    }

    private Tensor extractTensorRankFeature(Query query, String featureName) {
        Optional<Tensor> optionalTensor = query.getRanking().getFeatures().getTensor(featureName);
        return optionalTensor.orElseThrow(
                () ->
                        new IllegalArgumentException(
                                "Rank Feature: " + featureName + " not found in query!"));
    }

    private String addQueryWrapper(String str) {
        return "query(" + str + ")";
    }

    private void validate() {
        if (!retrievalMethod.getSupportedRankingMethods().contains(rankingMethod)) {
            throw new IllegalArgumentException(
                    String.format(
                            "Retrieval method '%s' does not support '%s' ranking method. It only"
                                    + " supports %s",
                            retrievalMethod,
                            rankingMethod,
                            retrievalMethod.getSupportedRankingMethods()));
        }
    }

    public RetrievalMethod getRetrievalMethod() {
        return retrievalMethod;
    }

    public RankingMethod getRankingMethod() {
        return rankingMethod;
    }

    public int getRrfK() {
        return rrfK;
    }

    public double getAlpha() {
        return alpha;
    }

    public int getTimeoutMillis() {
        return timeoutMillis;
    }

    @Override
    public String toString() {
        return "MarqoQuery{"
                + "retrievalMethod="
                + retrievalMethod
                + ", rankingMethod="
                + rankingMethod
                + ", rrfK="
                + rrfK
                + ", alpha="
                + alpha
                + ", timeoutMillis="
                + timeoutMillis
                + ", query="
                + query.toDetailString()
                + '}';
    }
}
