package ai.marqo.search;

import com.yahoo.search.result.HitGroup;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NamedHitGroup {
    private String name;
    private HitGroup hits;

    public NamedHitGroup(String name, HitGroup hits) {
        this.name = name;
        this.hits = hits;
    }

    @Override
    public String toString() {
        String hitsStr =
                IntStream.range(0, hits.size())
                        .mapToObj(this::formatHit)
                        .collect(Collectors.joining(","));
        return "HitGroup(" + name + ") with size " + hits.size() + ":[" + hitsStr + "]";
    }

    private String formatHit(int i) {
        return String.format(
                "(IDX: %d, ID: %s, REL: %.7f)",
                i, hits.get(i).getId(), hits.get(i).getRelevance().getScore());
    }
}
