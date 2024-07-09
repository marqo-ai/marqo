package ai.marqo.search;

import com.yahoo.search.Query;

public class VespaQueryDetails {
    private Query query;

    public VespaQueryDetails(Query query) {
        this.query = query;
    }

    @Override
    public String toString() {
        return "VespaQueryDetails{"
                + "query="
                + query.toDetailString()
                + "queryString="
                + query.getModel().getQueryString()
                + "yql="
                + query.properties().getString("yql", "")
                + "rankingFeatures="
                + query.getRanking().getFeatures().toString()
                + "rankProfile="
                + query.getRanking().getProfile()
                + '}';
    }
}
