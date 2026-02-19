# Custom Score Global Reranking – Final Plan

**Combined from:** `custom_score_rerank_feature_plan.md` and `custom_score_rerank_revised_plan.md`.  
**Constraints for this feature:**
1. We will **not** implement ranking query in this feature: no `label` in YQL or in `closeness()` calls.
2. There is **only one** query input for embeddings: **`marqo__query_embedding`**. We do not separate retrieval vs ranking vector.
3. Scores for custom reranking come from **summary-features** and **`ranking_closeness_metric_*`** functions (not from `closeness()` or extra `nearestNeighbor` terms).

---

# Coding Rules

1. Only test on semi-structured indexes. Do not use structured index.
2. Don't assume the Java tests pass. Run `mvn spotless:apply`, `mvn test`, `mvn clean package`, and rerun vespa full-start to actually test your changes.

---

# Summary

This document details the approach for the custom score global reranking feature. It allows fine-grained control over hybrid search by exposing specific scores (BM25 or vector closeness) via global score modifiers, per field or as aggregates, for all hits. Scores are computed in the rank profile and exposed as **summary-features**; the custom searcher fills and reads them (no match-features for custom score reranking).

---

# Problem Statement

Currently, lexical and tensor retrievers only rank via aggregate scores across all fields (SUM for BM25, MAX for closeness). There is no way to use these scores in the global phase. We want **bm25** and **vector closeness** to be available as **score modifiers in the global score modifier stage**, with per-field or aggregate weighting.

---

# Glossary

1. **bm25** – Okapi BM25 over an indexed string field.
2. **closeness** – In Vespa, a match feature used with `nearestNeighbor`. For **custom score reranking** we do **not** use `closeness()` in match-features; instead we use rank-profile functions **`ranking_closeness_metric_<field_name>()`** that compute similarity to `marqo__query_embedding` and expose them as **summary-features**.
3. **rank()** – Vespa query operator. We use it **only** when BM25 rerankers are present, to attach the BM25 term; we do **not** add extra `nearestNeighbor` terms for ranking.
4. **marqo__query_embedding** – The **single** query input for embeddings. It is used for both retrieval (in the main `nearestNeighbor` search) and for the ranking score functions (similarity to this vector). There is no separate “ranking vector” or “retrieval vector” in this feature.

---

# Tenets

- **Customer Obsession:** Search must be more relevant and configurable; solution must be usable, performant, and reliable.
- **Reliability:** All other search features must continue to work.
- **Performance:** Extra rank features and reranking must not cost excessive resources (NFR: CPU and p50 latency increase by no more than 10%).

---

# Functional Requirements

- **FR-1:** User can rerank by fine-grained scores: bm25 or vector closeness, per single field or aggregate (sum/max/avg).
- **FR-2:** Multiple scores can be weighted and combined via global `score_modifiers`.
- **FR-3:** Custom scores work together with normal global `score_modifiers`.
- **FR-4:** Pre-rerank score exposed in results for debugging where applicable.

---

# Out Of Scope (this feature)

1. **Ranking query** – No separate “ranking vector” or labels in YQL; no `label` in `nearestNeighbor` or `closeness()`.
2. **rerankStart** – Rerank results from a certain index onwards. We currently only support reranking from the first result.

---

# API Design

API remains unchanged. Custom score reranking uses `scoreModifiers` with a `field_name` prefixed with `marqo__score_`. Format:

```text
# Single field
"marqo__score_{SCORE_TYPE}_field_{FIELD_NAME}"

# Aggregate
"marqo__score_{SCORE_TYPE}_{AGGREGATE_TYPE}"

# Supported for this feature:
SCORE_TYPE in ("bm25", "closeness_retrieval_vector")
AGGREGATE_TYPE in ("sum", "max", "avg")
```

Example:

```python
"scoreModifiers": {
  "add_to_score": [
      {"field_name": "marqo__score_bm25_field_variantTitle", "weight": 1},
      {"field_name": "marqo__score_bm25_max", "weight": 1},
      {"field_name": "marqo__score_closeness_retrieval_vector_field_variantImage", "weight": 1},
      {"field_name": "marqo__score_closeness_retrieval_vector_avg", "weight": 1},
  ],
  "multiply_score_by": [
      {"field_name": "marqo__score_bm25_sum", "weight": 1},
  ]
}
```

*Note: `closeness_ranking_vector` is out of scope for this plan (no ranking query).*

---

# Architecture (High Level)

1. **Rank profile:** Define per-field functions `ranking_closeness_metric_<field_name>()` (similarity to `marqo__query_embedding`) and expose them and per-lexical BM25 in **summary-features** only. No custom-score closeness or BM25 in match-features.
2. **YQL:** Use `rank()` only when BM25 rerankers are present (to attach the extra BM25 terms). Do **not** add any 
   extra `nearestNeighbor(..., marqo__query_embedding)` terms. Do **not** use `label` in YQL.
3. **Custom searcher:** Before reranking, call `execution.fill(result, "summaryfeatures")` when custom score keys are present; then read scores **only** from summary-features (no fallback to match-features) and apply global score modifiers.

---

# Low Level Design

## 1. YQL / rank() (Python)

- **Single embedding input:** All tensor search uses **`marqo__query_embedding`**. No separate retrieval vs ranking vector; no labels.
- **rank():** Use **only** when BM25 rerankers are present. Then: `rank(retrieval_query, extra_bm25_terms)`.
- **No extra nearestNeighbor for ranking:** Do not add any `nearestNeighbor(..., marqo__query_embedding)` (or any ranking vector) as extra rank terms. Closeness for reranking comes from summary-features computed by the rank profile, not from extra YQL.

**Case A – BM25 rerankers present**
- Tensor retriever will have nearest neighbor searches in the retrieval term but another one (in this case the contains 
  term for the variantTitle field custom score) as the 2nd term.

```yql
select * from index where rank(
  ( {targetHits:100} nearestNeighbor(emb_field1, marqo__query_embedding) or
    {targetHits:100} nearestNeighbor(emb_field2, marqo__query_embedding) or
  ),
  marqo__lexical_variantTitle contains 'sometext'
)
```

- Lexical retriever will have the normal lexical term in the retrieval term but will have another one as the 2nd term

```yql
select * from index where rank(
  marqo__lexical_variantDescription contains 'sometext',
  marqo__lexical_variantTitle contains 'sometext'
)
```

**Case B – only closeness custom score (no BM25 rerankers)**

```yql
select * from index where
  {targetHits:100} nearestNeighbor(emb_field1, marqo__query_embedding) or
  {targetHits:100} nearestNeighbor(emb_field2, marqo__query_embedding)
```

(No `rank()` and no labels.)

### Fine-grained explanation of generating YQL for extra bm25 term:
1. Final YQL must follow these rules:
    1. If mode is hybrid search AND RRF AND at least 1 valid custom_score_reranker is set:
        1. For the tensor retriever it should look like:`rank(tensor_term, bm25_custom_rerank_term)`
        2. For the lexical retriever it should look like: `rank(lexical_term, bm25_custom_rerank_term)`
2. How do we get `bm25_custom_rerank_term`?
    1. Modify `_get_lexical_search_term` such that you can pass it 
       1. `_is_ranking_term` boolean ( give this parameter to `_generate_or_terms` and `_get_lexical_contains_term` as well
       2. For `_generate_or_terms`, if it’s a ranking term, don’t put `targetHits`. It’s unneeded since a ranking term doesn’t retrieve hits
       3. For `_get_lexical_contains_term`, no need to check only 1 of query and attributes_to_search are defined. Just check if `_is_ranking_term`, and if it is, use `attributes_to_search`.
       4. `_get_lexical_contains_term` should be the only method that actually makes a `contains` statement, so we have no 
                 duplication of code. 
       5. Modify `_get_lexical_contains_term` such that:
          - It can also accept an input: `attributes_to_search` where you can manually give it a list of attributes 
             to include a `contains` term for. It should also be able to contain a value (maybe `*`) indicating that you should use `default contains` (this is for sum/max/avg). Only ONE of `attributes_to_search` or 
                        `query` can be set. It can’t be both or neither. Document this behavior in the docstring.
       6. Now we make a method `_get_fields_to_bm25_rerank_by`. It accepts 
                    `custom_score_keys: Set[str] = set()` as input, checks all the bm25 related keys, and returns a list of all involved fields for bm25. If there are any aggregates (sum/avg/max), just return a list with `*`. `_get_lexical_contains_term` should interpret this as the sign to use `default contains`.

6. So the flow would be:
    1. Get `custom_score_rerank` from `self._get_hybrid_score_modifiers`
    2. Use `custom_score_rerank` to get `custom_score_keys`
    3. Use `_get_fields_to_bm25_rerank_by` to get the fields
    4. Use `_get_lexical_search_term` with those fields to create the search terms
    5. Put them together with rank so you will have a whole new `lexical_term` and `tensor_term`


---

## 2. Rank Profile (Semi-Structured Schema)

### 2.1 Query input

- **One embedding input:** `query(marqo__query_embedding) tensor<float>(x[dim])`. Used for both retrieval (in YQL) and for the ranking score functions below.
- Custom score weights: `marqo__custom_score_mult_weights_global`, `marqo__custom_score_add_weights_global`.

### 2.2 No match-features for custom score

- Do **not** add `closeness(label, ...)` or extra BM25 to match-features for custom score reranking. Custom score reranking uses **summary-features** only.

### 2.3 Per-field ranking functions (summary-features)

- For each **tensor field** (Marqo field name e.g. `tensor_field_a`): define a function **`ranking_closeness_metric_<field_name>()`** that computes similarity to **`query(marqo__query_embedding)`** and normalizes to [0,1] where 1 = closest (where possible; DotProduct may be normalized in Java).
- For each **lexical field**: BM25 is exposed in summary-features as **`bm25(marqo__lexical_<field>)`**. One per lexical field;

**Summary-features (example):**

```vespa
summary-features {
    ranking_closeness_metric_tensor_field_a
    ranking_closeness_metric_tensor_field_b
    bm25(marqo__lexical_text_field1)
    bm25(marqo__lexical_text_field2)
}
```

- **No** `marqo__tensor_field_order` query input; the searcher looks up `ranking_closeness_metric_` + &lt;field name&gt; from the custom score key.

**Distance metrics:** Support angular, prenormalized-angular, euclidean, dotproduct, hamming (per index). Geodegrees not supported for ranking. Expressions (conceptually):

- Angular / Prenormalized-angular: `(1.0 + cosine_similarity(attribute(emb_field), query(marqo__query_embedding), x)) / 2.0`
- Euclidean: `1.0 / (1.0 + euclidean_distance(attribute(emb_field), query(marqo__query_embedding), x))`
- DotProduct: `reduce(attribute(emb_field) * query(marqo__query_embedding), sum, x)` — raw in schema; **normalized in custom searcher** to [0,1] with 1=closest via min-max across hits **only when index distance metric is dot product** (Python sets `marqo__custom_score_closeness_distance_metric` to the index’s distance metric value; searcher checks for `"dotproduct"`; angular/euclidean/hamming are already [0,1] in the rank profile so are not min-max normalized in the searcher).
- Hamming: `1.0 - (hamming(...) / (8.0 * dim))`

### 2.4 Document summary for fill

- Define a document-summary (e.g. `summaryfeatures`) that can be requested when filling so that the custom searcher receives summary-features (e.g. `execution.fill(result, "summaryfeatures")`).

---

## 3. Python Query Properties

- When custom score keys indicate **closeness** reranking: set `marqo__hasRankingVector` so the searcher knows to fill and use summary-features for closeness.
- When custom score keys indicate **BM25** reranking: set `marqo__hasRankingLexical`.
- Do **not** set `marqo__tensor_field_order`; summary features are named by field.

---

## 4. Searcher (Java)

### 4.1 Fill before reranking

- If `marqo__hasRankingVector` or `marqo__hasRankingLexical`: build result from hits and call **`execution.fill(result, "summaryfeatures")`** before applying custom score modifiers.

### 4.2 Reading scores (summary-features only)

- **No fallback to match-features.** Custom score values are read **only** from summary-features.
- **closeness_retrieval_vector:**  
  - Single field: get summary feature **`ranking_closeness_metric_<field_name>`** (field name from the key).  
  - Aggregate: collect all summary feature names starting with `ranking_closeness_metric_`, read values, then sum/max/avg.
- **bm25:**  
  - Single field: get **`bm25(marqo__lexical_<field>)`**.  
  - Aggregate: collect all `bm25(marqo__lexical_*)` from summary-features and aggregate.
- Rank features may be returned as tensors (e.g. single-cell); the searcher must handle both scalar and tensor (e.g. sum of tensor cells) when reading a double.

### 4.3 Applying modifiers

- Use the same logic as in the feature plan: for each key in add weights, add (weight × normalized score) to the 
  `global_add_modifier`; for each key in mult weights, multiply the `global_mult_modifier` by (weight × normalized 
  score). BM25 scores are min-max normalized across hits when needed.
- The `global_add_modifier` and `global_mult_modifier` are already extracted from match-features anyway so simply 
  execute the above step before applying the modifiers to the final score.

### 4.4 Return _pre_rerank_score per hit
- For each hit, SAVE the score applying the modifiers in a field, and let marqo interpret it and output it in the 
  final result. This will let us see the pure RRF score before the global modifiers, which is useful for debugging.
- We already do this for _lexical_score and _tensor_score, so use a similar approach.

### 4.5 Logs
- To make it easy to trace exactly what's happening in the custom searcher, please emit a clear log message every time:
  - We read and unpack the custom score requests from the query properties, including the score type, field, 
    aggregate type, keys and weights for each.
  - We read a custom score from summary-features, including the key and value.
  - We perform an aggregation (e.g. sum/max/avg) across multiple fields for a custom score, including the keys, individual values, and resulting aggregate value.
  - We apply a custom score modifier, including the key, weight, score value, and resulting modifier value.
  - Make it clear enough that it's easy for a human to understand the whole story of what happened when reading the 
    logs.

---

# Testing

- **Unit (Python):** Conversion of score modifier field names to query inputs; YQL has no extra `nearestNeighbor` for ranking and no labels; `rank()` only when BM25 rerankers present.
- **Unit (Schema):** One `ranking_closeness_metric_<field_name>()` per tensor field; summary-features include these and per-lexical BM25; no closeness/BM25 in match-features for custom score.
- **Unit (Java):** Weights and keys parsed correctly; scores read from summary-features only; tensor/scalar handling for feature values.
- **Integration:** Single-field and aggregate bm25 and closeness_retrieval_vector; combination with global score modifiers; no use of labels or ranking query.

- Unit Tests
    - Python search functions
        - Test conversion of input score modifier field names to `tensor` type query inputs
        - Test that correct `rank()` statements are constructed with custom scores
        - Test that facets query stays the same
    - Schema
        - Test schemas have new match features and query inputs
    - Java searcher
        - Test that weights are extracted per hit, custom scores are extracted, applied to global score modifiers
- Integration Tests
    - Successful use cases - Custom Scores should work as expected in the following scenarios:
        - single field bm25
        - single field closeness retrieval vector
        - aggregated bm25
        - aggregated closeness retrieval vector
        - multiple custom scores in one query
        - multiple custom scores with regular global score modifiers
    - Failure cases
        - with sort_by (validation error)
        - requesting scores of non-existent fields
        - requesting non-existent scores
    - Interactions with other features (should still work as intended)
        - with rerankDepthTensor
          - Should not affect each other. Since rerankDepthTensor works on targetHits of the tensor retriever, 
            while custom score reranking only happens in global.
        - with rerankDepth
          - They should work together! Show that custom score rerankers ONLY affect the global results up to 
            rerankDepth. If there are 5 final hits, but rerankDepth is 3, show that only the top 3 hits are affected 
            by the custom score reranking and the bottom 2 are not.
        - with facets
          - Should be unaffected. Facets are a separate query that work in parallel. Show the facets query is 
            completely unaffected by the changes for custom score reranking. show that if you use them both, facets 
            still returns the correct answer.
        - with pagination
          - Should be unaffected. Pagination is a separate step that happens after reranking. Show that if you use pagination 
            with custom score reranking, you get the correct paginated results.
        - with collapse_fields
          - Collapsing happens during fusion portion. Reranking will happen to fused list. Should not affect each 
            other. Show this in a test.
        - with relevance_cutoff
          - Custom scores do not change or use targetHits, so they should be unaffected.
          - In your test, make sure probeLexicalQuery is not changed. This is the lexical query used to determine 
            target hits. 
        - **sort_by**
          - Since this is another global reranker, cannot be used at the same time as ranking query. 
          - API validation: we should error out if both `sort_by` and `rankingQueryTensor` / `rankingContext` are defined.
        - with recency boost
          - Should work independently of this feature. Since it already works with existing global score modifier application, nothing should change.

## Integration Test Structure
Tests are only useful if they definitively show the core properties of the feature work. I will detail how integration tests should look in order to prove this.
At its core, each test must show:
(1) Custom score rerankers modified the final score, and by a certain amount. This means if the custom score 
modifier is `marqo__score_bm25_field_my_field`, the doc with the highest BM25 score for `my_field` should end with a 
higher score than a doc with a lower BM25 score for that field.
(2) This score modification reliably and deterministically changes the order of results, based on how well they 
match the custom score field/aggregate used.
(3) The final _score will be different from the _pre_rerank_score, and you should know exactly by how much and why 
based on the custom score modifier and the scores for each hit.

## Instructions
Our query will be `tuxedo`.
Each integration test should use this index:
 - model is "open_clip/ViT-B-16-SigLIP-512/webli". This is because we know they exact closeness of different terms 
   to our goal term, which is `tuxedo`.
 - We should have 4 fields to work with. lex_retrieval_field, lex_ranking_field, tens_retrieval_field, tens_ranking_field
 - We define the documents such that RRF results are in deterministic order. Base order is deterministic, and so is 
   order with custom score reranking.

Closeness (prenormalized-angular) to term 'tuxedo' with model 'open_clip/ViT-B-16-SigLIP-512/webli'
tuxedo -> 1.0
black tuxedo -> 0.9290061705548538
black tie -> 0.9105825129267998
suit -> 0.901995477338818
shorts -> 0.8311302085908341
backpack -> 0.8264572877032847
floral dress -> 0.825913938286213
suede shoes -> 0.8106355248508749
rainbow tie -> 0.7955299917394352
unrelated -> 0.5882339267201514

Doc order:
(1) In BOTH tensor and lexical
(2) In ONLY tensor (medium strength)
(3) In ONLY lexical (medium strength)
(4) In ONLY tensor (lower strength)
(5) In ONLY lexical (lower strength)

Base RRF order: 1,2,3,4,5
add_to_score with bm25 lex_ranking_field will REVERSE the order: 5,4,3,2,1. 
add_to_score with tensor_ranking_field to REVERSE the order: 5,4,3,2,1.
For multiply_score_by, the doc in both lexical and tensor has original score too high, so no matter the multiplier, the customs cores can't make it fully reverse. But still test that it affects the scores.

```python
docs = [
    {
        # (1) In BOTH tensor and lexical
        "_id": "doc1",
        "lex_retrieval_field": "tuxedo tuxedo tuxedo",  # VERY HIGH lexical score
        "tensor_retrieval_field": "tuxedo",             # VERY HIGH tensor score

        "lex_ranking_field": "tuxedo",                  # lowest bm25 score for global reranking
        "tensor_ranking_field": "unrelated"             # lowest closeness score for global reranking
    },
    {
        # (2) In ONLY tensor (medium strength)
        "_id": "doc2",
        "lex_retrieval_field": "no match",               # no lexical match
        "tensor_retrieval_field": "suit",                # MEDIUM tensor score

        "lex_ranking_field": "tuxedo tuxedo",           # 2nd lowest bm25 score for global reranking
        "tensor_ranking_field": "rainbow tie"           # 2nd lowest closeness score for global reranking
    },
    {
        # (3) In ONLY lexical (medium strength)
        "_id": "doc3",
        "lex_retrieval_field": "tuxedo tuxedo",         # MEDIUM lexical score

        "lex_ranking_field": "tuxedo tuxedo tuxedo",    # 3rd lowest bm25 score for global reranking
        "tensor_ranking_field": "shorts"                # 3rd lowest closeness score for global reranking
    },
    {
        # (4) In ONLY tensor (lower strength)
        "_id": "doc4",
        "lex_retrieval_field": "no match",              # no lexical match
        "tensor_retrieval_field": "shorts",             # LOWER tensor score (but it's still clothes)

        "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo", # 4th lowest bm25 score for global reranking
        "tensor_ranking_field": "suit"                      # 4rd lowest closeness score for global reranking
    },
    {
        # (5) In ONLY lexical (lower strength)
        "_id": "doc5",
        "lex_retrieval_field": "tuxedo",                # LOW lexical score

        "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo",  # highest bm25 score for global reranking
        "tensor_ranking_field": "tuxedo"                            # closeness bm25 score for global reranking
    },
]

tensor_fields=["tensor_retrieval_field", "tensor_ranking_field"]
```

Now for these results to be deterministic, we must set alpha only slightly above 0.5 to make sure the tensor results 
will be 
interleaved before lexical results, then we only make the retrieval fields searchable.
```python
# Search config for this to work
hybrid_parameters = {
    "alpha": 0.5001,   # To guarantee tensor results will always be slightly ahead in interleaving
    "searchableAttributesTensor": ["tensor_retrieval_field"],
    "searchableAttributesLexical": ["lex_retrieval_field"],
}
```

Now if we want to test, we simply change what we put inside the `score_modifiers` parameter.
For example, for `test_rrf_with_bm25_single_field_modifies_scores`, use this:
```python
score_modifiers = {
    "add_to_score": [
        {"field_name": "marqo__score_bm25_field_lex_ranking_field", "weight": 1},
    ]
}
```
For `test_rrf_with_closeness_retrieval_vector_single_field_modifies_scores` use this:
```python
score_modifiers = {
    "add_to_score": [
        {"field_name": "marqo__score_closeness_retrieval_vector_field_tensor_ranking_field", "weight": 1},
    ]
}
```
With this method, you can test that it changes order AND changes the score in the same test. No need for separate tests.
For multiply_score_by, it will be harder to change order, so you can check score for that one.

## Comprehensive Testing
1. Test all aggregate types. sum/max/avg.
    - To get the base closeness and bm25 scores, 
2. Test all distance metric types (angular, prenormalized-angular, euclidean, dotproduct, hamming) for closeness.
  - For this, you would have to create separate indexes with different distance metrics, but you can use the same documents and queries.

---

# References

- Vespa rank() and YQL: [rank()](https://docs.vespa.ai/en/reference/querying/yql.html), [nearestNeighbor](https://docs.vespa.ai/en/reference/querying/yql.html#nearestneighbor).
- Vespa ranking: [ranking expressions](https://docs.vespa.ai/en/ranking/ranking-expressions-features.html), [document summaries](https://docs.vespa.ai/en/querying/document-summaries.html).
- Original plans: `custom_score_rerank_feature_plan.md`, `custom_score_rerank_revised_plan.md` (unchanged).
