package ai.marqo.search;

import java.util.Set;

public enum RetrievalMethod {
    disjunction(RankingMethod.rrf),
    lexical(RankingMethod.lexical, RankingMethod.tensor),
    tensor(RankingMethod.lexical, RankingMethod.tensor);

    private final Set<RankingMethod> supportedRankingMethods;

    RetrievalMethod(RankingMethod... supportedRankingMethods) {
        this.supportedRankingMethods = Set.of(supportedRankingMethods);
    }

    public Set<RankingMethod> getSupportedRankingMethods() {
        return supportedRankingMethods;
    }
}
