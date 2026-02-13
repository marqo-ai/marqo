[OBSOLETE. Refer to custom_score_rerank_plan_final.md]
# Ranking Vector Reranking via Rank Profile Functions (Revised Plan)

## 0. Goal

Replace the current `rank() + nearestNeighbor + closeness()` approach for **ranking vector closeness** (semi-structured indexes) with:

- Per-field ranking functions in the Vespa rank profile that compute similarity to a separate ranking vector using tensor math.
- All new signals exposed as **summary-features**, fetched via `.fill()` in the global custom searcher.
- The YQL query **unchanged**, except that `rank()` is only used when BM25 rerankers are present.

Distance metrics are defined **per index**, not per field, as one of:
- `euclidean`
- `angular`
- `dotproduct`
- `prenormalized-angular`
- `hamming`
- `geodegrees` (geo; **not supported** for ranking vectors)

All ranking scores must be normalized to \([0, 1]\) where 1 = closest.

---

## 1. YQL / rank() Changes

### 1.1 Remove extra nearestNeighbor for ranking vector

**Current behavior (simplified):**

```yql
select * from index where rank(
  (
    {label:"retrieval_field1", targetHits:100} nearestNeighbor(emb_field1, marqo__retrieval_query_embedding) or
    {label:"retrieval_field2", targetHits:100} nearestNeighbor(emb_field2, marqo__retrieval_query_embedding) or
    userQuery()
  ),
  {label:"ranking_field1", targetHits:1} nearestNeighbor(emb_field1, marqo__query_embedding),
  {label:"ranking_field2", targetHits:1} nearestNeighbor(emb_field2, marqo__query_embedding),
  {field: marqo__ranking_strings}userQuery()   -- BM25 rerankers
)
```

**New behavior:**

- **Never** generate `nearestNeighbor(..., marqo__query_embedding)` in the query.
- `rank()` is used **only** to attach BM25 rerankers when they are present.

**Case A – BM25 rerankers present**

```yql
select * from index where rank(
  (
    {label:"retrieval_field1", targetHits:100} nearestNeighbor(emb_field1, marqo__retrieval_query_embedding) or
    {label:"retrieval_field2", targetHits:100} nearestNeighbor(emb_field2, marqo__retrieval_query_embedding) or
    userQuery()
  ),
  {field: marqo__ranking_strings}userQuery()
)
```

**Case B – only ranking vector closeness (no BM25 rerankers)**

```yql
select * from index where
  {label:"retrieval_field1", targetHits:100} nearestNeighbor(emb_field1, marqo__retrieval_query_embedding) or
  {label:"retrieval_field2", targetHits:100} nearestNeighbor(emb_field2, marqo__retrieval_query_embedding) or
  userQuery()
```

**Implementation sketch (Python):**

```python
def _build_yql(self, has_ranking_vector: bool, has_bm25_rerankers: bool) -> str:
    retrieval_terms = self._get_retrieval_tensor_terms(marqo_query) + self._get_lexical_terms(marqo_query)

    if len(retrieval_terms) == 0:
        raise ValueError("No retrieval terms")
    elif len(retrieval_terms) == 1:
        base_query = retrieval_terms[0]
    else:
        base_query = f"({' or '.join(retrieval_terms)})"

    if has_bm25_rerankers:
        bm25_rerank_term = "{field: marqo__ranking_strings}userQuery()"
        return f"rank({base_query}, {bm25_rerank_term})"
    else:
        # No rank() needed when only ranking vector is present
        return base_query
```

---

## 2. Rank Profile Changes (Semi-Structured Indexes)

### 2.1 Remove closeness(label, ranking_...) from features

**Current:**

```vespa
match-features {
    closeness(label, retrieval_field1)
    closeness(label, retrieval_field2)
    closeness(label, ranking_field1)
    closeness(label, ranking_field2)
}
```

**New:**

- **Remove** all ranking-vector closeness and BM25 from `match-features`. Custom score reranking uses **only** `summary-features`.
- Introduce **per-field ranking functions** named by the **tensor field name** (the Marqo field name, e.g. `tensor_field_a`), and expose them as `summary-features`. No query input for field order is needed: the searcher looks up `ranking_closeness_metric_<field_name>` by the field name in the custom score key.
- BM25 for reranking is also exposed only in `summary-features`, **one per lexical field**: `bm25(marqo__lexical_<field1>)`, `bm25(marqo__lexical_<field2>)`, etc. There is **no** `bm25(marqo__ranking_strings)` in summary-features.

```vespa
summary-features {
    ranking_closeness_metric_tensor_field_a
    ranking_closeness_metric_tensor_field_b
    bm25(marqo__lexical_text_field1)
    bm25(marqo__lexical_text_field2)
}
```
(One entry per tensor field and one per lexical field; no aggregate BM25 feature.)

> Note: We use `summary-features` only for custom score reranking. The custom searcher calls `.fill(result, "summaryfeatures")` before reranking. There is **no** `marqo__tensor_field_order` query input; the summary feature name is `ranking_closeness_metric_` + &lt;Marqo tensor field name&gt;.

### 2.2 Add per-field ranking score functions

**Goal:** For each tensor field (with Marqo field name `field_name`, e.g. `tensor_field_a`), define a function **`ranking_closeness_metric_<field_name>()`** that computes similarity to `marqo__query_embedding` and normalizes it to \([0,1]\) where 1 = closest. The function and summary-feature are named by the **actual tensor field name** so the searcher can look up `ranking_closeness_metric_` + field name from the custom score key without any query input for field order.

Assumptions:
- Tensor fields: `marqo_index.tensor_fields` with `.embeddings_field_name`
- Vector dimension: `dim = marqo_index.model.get_dimension()`
- Index distance metric: `marqo_index.distance_metric` \(Angular, PrenormalizedAngular, Euclidean, DotProduct, Hamming\)

**Schema generator helpers (Python):**

```python
def _get_ranking_score_expression(field_name: str,
                                  distance_metric: DistanceMetric,
                                  dim: int) -> str:
    """Return ranking expression normalized to [0,1] where 1 = closest.

    For DotProduct, the expression is unbounded; we normalize in the Java searcher.
    """
    if distance_metric in (DistanceMetric.Angular, DistanceMetric.PrenormalizedAngular):
        # cosine_similarity ∈ [-1, 1] → normalize to [0, 1]
        return (
            f"(1.0 + cosine_similarity("
            f"attribute({field_name}), "
            f"query(marqo__query_embedding), "
            f"x)) / 2.0"
        )

    elif distance_metric == DistanceMetric.Euclidean:
        # euclidean_distance ∈ [0, ∞) → 1 / (1 + d) ∈ (0, 1]
        return (
            f"1.0 / (1.0 + euclidean_distance("
            f"attribute({field_name}), "
            f"query(marqo__query_embedding), "
            f"x))"
        )

    elif distance_metric == DistanceMetric.DotProduct:
        # Unbounded; normalize later per query in Java
        return (
            f"reduce("
            f"attribute({field_name}) * "
            f"query(marqo__query_embedding), "
            f"sum, x)"
        )

    elif distance_metric == DistanceMetric.Hamming:
        # hamming ∈ [0, 8*dim] → convert to [0,1], 1 = identical
        return (
            f"1.0 - (hamming("
            f"attribute({field_name}), "
            f"query(marqo__query_embedding)) "
            f"/ (8.0 * {dim}))"
        )

    elif distance_metric == DistanceMetric.Geodegrees:
        # Not meaningful for embedding similarity
        raise ValueError(
            "Geodegrees distance metric is not supported for ranking vectors. "
            "It is only applicable to geographic position fields, not embedding tensors."
        )

    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")


def _generate_ranking_score_functions(self, marqo_index: SemiStructuredMarqoIndex) -> List[str]:
    functions: List[str] = []
    distance_metric = marqo_index.distance_metric
    dim = marqo_index.model.get_dimension()

    for tensor_field in marqo_index.tensor_fields:
        # Use Marqo field name so searcher can look up ranking_closeness_metric_<name> without field order
        field_name = tensor_field.embeddings_field_name  # for expression (Vespa attribute name)
        marqo_field_name = tensor_field.name             # for function/summary-feature name
        expr = _get_ranking_score_expression(field_name, distance_metric, dim)
        functions.append(
            "function ranking_closeness_metric_" + marqo_field_name + "() {\n"
            "    expression: " + expr + "\n"
            "}"
        )

    return functions
```

**Rank profile template (semi-structured, simplified):**

```vespa
rank-profile hybrid_with_ranking_features inherits default {
    inputs {
        query(marqo__query_embedding) tensor<float>(x[{DIM}])
        query(marqo__query_embedding) tensor<float>(x[{DIM}])
    }

    # Retrieval scoring (existing behavior)
    function embedding_score() {
        expression: max(
            if(query(marqo__embeddings_field1) > 0, closeness(field, marqo__embeddings_field1), 0),
            if(query(marqo__embeddings_field2) > 0, closeness(field, marqo__embeddings_field2), 0)
        )
    }

    first-phase {
        expression: embedding_score()
    }

    # Per-field ranking vector scores (normalized when possible). Named by field for direct lookup.
    function ranking_closeness_metric_embeddings_field1() {
        expression: (1.0 + cosine_similarity(attribute(marqo__embeddings_field1),
                                            query(marqo__query_embedding), x)) / 2.0
    }

    function ranking_closeness_metric_embeddings_field2() {
        expression: (1.0 + cosine_similarity(attribute(marqo__embeddings_field2),
                                            query(marqo__query_embedding), x)) / 2.0
    }

    summary-features {
        ranking_closeness_metric_embeddings_field1
        ranking_closeness_metric_embeddings_field2
        bm25(marqo__lexical_text_field)
    }
}
```

> Note: For Euclidean / Hamming, `ranking_closeness_metric_<field>()` is already normalized to [0,1] in the function. For DotProduct, the function returns an unbounded score; normalization happens in the Java searcher.

---

## 3. Searcher Changes (Java)

### 3.1 Fill summary-features before reranking

**File:** `HybridSearcher.java`

**New flow (simplified):**

```java
@Override
public Result search(Query query, Execution execution) {
    // 1) Run underlying Vespa search (tensor + lexical fusion)
    Result result = execution.search(query);
    HitGroup hits = result.hits();

    boolean hasRankingVector = query.properties().getBoolean("marqo.hasRankingVector", false);
    boolean hasRankingLexical = query.properties().getBoolean("marqo.hasRankingLexical", false);

    // 2) If we need ranking features, fill summary-features first
    if (hasRankingVector || hasRankingLexical) {
        execution.fill(result, "summaryfeatures");
    }

    // 3) Rerank using summary-features
    if (hasRankingVector || hasRankingLexical) {
        reRankByRankingSignals(hits, query);
    }

    // 4) Fill full document summaries as usual
    execution.fill(result);
    return result;
}
```

### 3.2 Reranking using per-field scores (summary-features only)

- The searcher **only** uses summary-features for custom score reranking (no fallback to match-features).
- For **closeness_retrieval_vector**: single-field key → summary feature `ranking_closeness_metric_<field_name>` (Marqo field name from the key). Aggregate key (sum/max/avg) → iterate summary feature names that start with `ranking_closeness_metric_`, collect values, then aggregate.
- For **bm25**: read from summary-features only; single field → `bm25(marqo__lexical_<field>)`, aggregate (sum/max/avg) → aggregate over all `bm25(marqo__lexical_*)` values in summary-features.
- No query input `marqo__tensor_field_order` is needed because summary features are named by field name.


## 4. Tests

### 4.1 Unit tests – YQL

- Ensure no EXTRA `nearestNeighbor(..., marqo__query_embedding)` appears in generated YQL if closeness custom score 
  is requested (after the first rank 
  term,
  if rank() is used).
- Ensure `rank()` is only emitted when BM25 rerankers are present.

### 4.2 Unit tests – schema

- One `ranking_closeness_metric_<field_name>()` per tensor field (named by Marqo field name).
- Expressions differ by distance_metric; keep them readable in the template (multi-line where helpful).
- `summary-features` contains ranking scores and BM25; **no** closeness/bm25 in match-features for custom score reranking.

```python
def test_ranking_functions_per_field():
    schema = generate_schema(
        tensor_fields=["title_embeddings", "description_embeddings"],
        distance_metric=DistanceMetric.PrenormalizedAngular,
    )
    assert "function ranking_closeness_metric_title_embeddings()" in schema
    assert "function ranking_closeness_metric_description_embeddings()" in schema
    assert "cosine_similarity(attribute(marqo__embeddings_title_embeddings)" in schema


def test_distance_metric_expressions():
    expr = _get_ranking_score_expression("emb", DistanceMetric.Angular, 384)
    assert "cosine_similarity" in expr and "/ 2.0" in expr

    expr = _get_ranking_score_expression("emb", DistanceMetric.Euclidean, 384)
    assert "euclidean_distance" in expr and "1.0 / (1.0 +" in expr

    expr = _get_ranking_score_expression("emb", DistanceMetric.DotProduct, 384)
    assert "reduce(" in expr and "sum, x" in expr

    expr = _get_ranking_score_expression("emb", DistanceMetric.Hamming, 384)
    assert "hamming(" in expr and "/ (8.0 * 384)" in expr

    with pytest.raises(ValueError):
        _get_ranking_score_expression("emb", DistanceMetric.Geodegrees, 384)


def test_summary_features_not_match_features():
    schema = generate_schema(has_ranking_vector=True)
    assert "summary-features {" in schema
    assert "ranking_closeness_metric_" in schema
    assert "closeness(field," not in schema  # no ranking closeness in match-features
```

### 4.3 Integration tests

- Existing hybrid search behavior should remain similar:
  - Retrieval unchanged
  - Reranking now uses normalized summary-features instead of closeness() from match-features.
- Assert that end-to-end requests succeed and ordering differences (if any) are minor and explainable by monotonic normalization.
