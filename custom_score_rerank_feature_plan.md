Stakeholder: @Jesse Clark 

Author: @Joshua Kim 

# Coding Rules
1. Only test on semi-structured indexes. Do not use structured index.
2. Don't assume the Java tests pass. Run mvn spotless:apply, mvn test, mvn clean package, and rerun vespa full-start to 
actually test your changes.
3. When testing multiple different cases, use self.subTest with a list of cases. This makes it easier to see which 
   subTest failed.
# Summary

This document details the approach and intended solution for the custom score global reranking feature. This feature will allow us even greater control over hybrid search by making specific scores (bm25 or closeness) accessible via global score modifiers per field for all hits. Allowing for access to and weighting of all these score knobs will get us closer to an optimal search configuration. This document is geared toward a more technical audience.

# Problem Statement

Currently in both our lexical and tensor retrievers, we only rank via the aggregate of their scores across all fields (SUM for bm25, MAX for closeness). Additionally, we have no way to rank with these scores in the global phase. Only in the individual retrievers. Detailed table of supported reranking methods phase is in [Appendix A](https://www.notion.so/Custom-Score-Global-Reranking-Feature-2f875d43da4c8087a820ffbde1b819be?pvs=21).

For better relevance fine-tuning, we want to use bm25 and closeness scores as **score modifiers in the global score modifier stage**. It should be weighted and able to be combined with other fields, to be reranked in one step.

# Glossary

1. `bm25` - Calculates the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking function over the given [indexed string field](https://docs.vespa.ai/en/reference/schemas/schemas.html#indexing-index).
2. `closeness` - Match feature used with the [nearestNeighbor](https://docs.vespa.ai/en/reference/querying/yql.html#nearestneighbor) query operator. A number which is close to 1 when a point vector in the document is close to a matching point vector in the query.
3. `rank() query operator` - Operator that allows for calculating of separate match features after retrieval. The first, and only the first, argument of the *rank()* function determines whether a document is a match, but all arguments are used for calculating rank features. The `rank` operator is useful for boosting documents based on the presence of certain terms without impacting matching or retrieval logic.
4. `retrieval vector` - vector used in the main `nearestNeighbor` search in the tensor retriever of hybrid search. This is used to retrieve the candidate tensor hits for ranking. Before implementation of ranking query, this is **also** ranking vector.
5. `ranking vector` - separate vector used for reranking but not retrieval of candidate hits. Simply used for calculating closeness and reranking.

# Tenets

**Customer Obsession: ****We prioritise making our search more relevant for our customers by creating the most robust and customizable configuration possible. This solution must not only meets functional requirements but also exceeds expectations in terms of usability, performance, and reliability.

**Reliability:** The system should be highly reliable as this ties into the **Customer Obsession** tenet. All other search features should still work as intended.

**Performance:** The feature should still be performant, with extraction of extra rank features and reranking not costing too much extra resources.

# Functional Requirements

- ***FR-1**: The user must be able to rerank all results by the accessible fine-grained scores: bm25 or vector closeness / on a single-field or aggregate*
- ***FR-2**: Multiple scores should be able to be weighted and combined via global `score_modifiers`*
- ***FR-3**: Custom scores should work in combination with normal global `score_modifiers`*
- ***FR-4**: Pre-rerank score should be exposed in results for debugging purposes*

# Non-Functional Requirements

- ***NFR-1**: CPU Usage increase by no more than 10% using custom scores*
- ***NFR-2**: p50 Latency should increase by no more than 10% with custom scores*

# Out Of Scope

1. Implementation of Ranking Query ([https://www.notion.so/Ranking-Query-Feature-24175d43da4c80b4b0cee3ada6d4cc0c](https://www.notion.so/Ranking-Query-Feature-24175d43da4c80b4b0cee3ada6d4cc0c?pvs=21))
    1. Should be implemented right after this feature
2. `rerankStart` parameter
    1. To rerank results starting from a certain index onwards in the global phase. Slack message here: https://marqo-ai.slack.com/archives/C0ABS5KMW1W/p1769728928924559

# Success Criteria

The feature is successful if custom scores can be used to modify in hybrid search to modify RRF hits’ scores, and all functional and non-functional requirements are met.

# API Design

API would technically remain unchanged. The way to access custom score reranking would be to use `scoreModifiers` at the query level, adding a `field_name` prefixed with `__marqo_score`. Since no index field name can start with `__marqo` , this ensures that the field name does not overlap any actual index field name.

We will follow the format below for `field_name`:

```sql
# For a specific field
f"__marqo_score_{SCORE_TYPE}_field_{FIELD_NAME}"

# For an aggregate
f"__marqo_score_{SCORE_TYPE}_{AGGREGATE_TYPE}"

# Currently supported types:
assert SCORE_TYPE in ("bm25", "closeness_ranking_vector", "closeness_retrieval_vector")
assert AGGREGATE_TYPE in ("sum", "max", "avg")
```

Here is how these would look for all score types, both individual fields and aggregates:

```python
"scoreModifiers": {
      "add_to_score": [
          {
			      # BM25 -> "variantTitle" field
	          "field_name": "__marqo_score_bm25_field_variantTitle",
	          "weight": 1,
          },
          {
			      # BM25 MAX 
			      "field_name": "__marqo_score_bm25_max",
	          "weight": 1,
          },
          {
			      # closeness to ranking vector -> "variantImage" field
	          "field_name": "__marqo_score_closeness_ranking_vector_field_variantImage",
	          "weight": 1,
          },
          {
			      # closeness to ranking vector SUM
			      "field_name": "__marqo_score_closeness_ranking_vector_sum",
	          "weight": 1,
          },
          {
			      # closeness to retrieval vector -> "variantImage" field
	          "field_name": "__marqo_score_closeness_retrieval_vector_field_variantImage",
	          "weight": 1,
          },
          {
			      # closeness to retrieval vector AVG
			      "field_name": "__marqo_score_closeness_retrieval_vector_avg",
	          "weight": 1,
          },
      ],
      "multiply_score_by": [
          {
			      # BM25 SUM
			      "field_name": "__marqo_score_bm25_sum",
	          "weight": 1,
          },
      ]
  },
```

# Architecture

1. Fetch all scores as match-features in match-phase for both retrievers
2. In custom searcher: Extract scores and combine with global score modifiers

### Scores as match features

To support this, we must include all these scores for all relevant fields as match features. `bm25` score must be a match feature for lexical fields and `closeness()` to the retrieval and ranking vector. Example:

```python
match-features {
    # Retrieval closeness
    closeness(label, retrieval_tensor_field1)
    closeness(label, retrieval_tensor_field2)

    # Ranking closeness
    closeness(label, ranking_tensor_field1)
    closeness(label, ranking_tensor_field2)

    # bm25
    bm25(lexical_field1)
    bm25(lexical_field2)
}
```

*Note: Until the [Ranking Query](https://www.notion.so/Design-Doc-30cca11efeed40e0a08dee4fbba4f85f?pvs=21) feature is implemented, only closeness to the retrieval vector will be available.*

 

Explained in detail in Low Level Design portion.

# Data Storage / Modeling

N/A

# Low Level Design

### A. Python Search Logic Changes

From the input score modifiers, we must do the following:

1. Identify the custom scores  (those with prefix `__marqo_score`), convert them and pass them as new dictionary query inputs: `marqo__custom_score_mult_weights_global` and `marqo__custom_score_add_weights_global` (in `vespa_index.py`)
    1. Those inputs would look like this:

```python
marqo__custom_score_mult_weights_global = {
	bm25_field_variantTitle: 1,
	closeness_ranking_vector_field_variantImage: 1
}

marqo__custom_score_add_weights_global = {
	bm25_max: 2,
	closeness_retrieval_vector_sum: 3
}
```

*Note: These would not be used in `marqo__mult_weights_global` and `marqo__add_weights_global`*

ii. Modify the yql queries to fetch all needed scores as match features using `rank()`. For every key in `marqo__custom_score_mult_weights_global` and `marqo__custom_score_add_weights_global`, we translate it to its corresponding yql statement and add it as an argument in `rank()`. 

Example with the inputs above:

```python
# bm25 turns into the corresponding _get_lexical_search_term
bm25_field_variantTitle 
-> marqo__lexical_variantTitle contains 'sometext'

# closeness against ranking vector turns into a nearestNeighbor term 
closeness_ranking_vector_field_variantImage
-> {label:'ranking_variantImage'} nearestNeighbor(marqo__embeddings_variantImage, marqo__ranking_query_embedding

# bm25 aggregate turns into the corresponding _get_lexical_search_term (for all applicable fields)
bm25_max
-> default contains 'sometext'

# closeness aggregate turns into nearestNeighbor terms (for all applicable fields)
closeness_retrieval_vector_sum
-> {label:'retrieval_variantImage'} nearestNeighbor(marqo__embeddings_variantImage, marqo__retrieval_query_embedding, {label:'retrieval_tensor_field2'} nearestNeighbor(marqo__embeddings_tensor_field2, marqo__retrieval_query_embedding
```

### A1. Overlapping match features

Some queries will overlap each other (will get the same scores). To minimize the number of rank terms, we remove any redundant queries. For example, we have:

- `marqo__lexical_variantTitle contains 'sometext'` which gets bm25 score for `variantTitle`
- `default contains 'sometext'` which gets bm25 scores for all lexically searchable fields.

In this case, we just keep `default contains 'sometext'` which will get a superset of match features.

For the tensor retriever, we would also remove the rank() queries for `closeness_retrieval_vector_sum`, since they are already calculated in the first term (retrieval portion). 

We can create a method `reduce_to_least_required_terms` to remove these redundant terms.

Putting it all together, the final YQL query with all of these as arguments for `rank()` would look like:

```sql
select * from index where rank(
	# FIRST TERM: standard tensor retrieval for 2 fields
	{label:'retrieval_variantImage', targetHits:100} nearestNeighbor(marqo__embeddings_tensor_field1, marqo__retrieval_query_embedding) or {label:'retrieval_tensor_field2', targetHits:100} nearestNeighbor(marqo__embeddings_tensor_field2,
   # FOLLOWING TERMS: collect needed match features
   # For closeness_ranking_vector_field_variantImage
   {label:'ranking_variantImage'} nearestNeighbor(marqo__embeddings_variantImage, marqo__ranking_query_embedding,
   # For bm25_max
   default contains 'sometext',
   {label:'retrieval_variantImage'} nearestNeighbor(marqo__embeddings_variantImage, marqo__retrieval_query_embedding, 
   {label:'retrieval_tensor_field2'} nearestNeighbor(marqo__embeddings_tensor_field2, marqo__retrieval_query_embedding
   # Removed terms for bm25_field_variantTitle and closeness_retrieval_vector_sum
)
```

Implementation for A:

1. Final YQL must follow these rules:
    1. If mode is hybrid search AND RRF AND at least 1 valid custom_score_reranker is set:
        1. For the tensor retriever it should look like:`rank(tensor_term, bm25_custom_rerank_term, nearest_neighbor_custom_rerank_term_field_1, nearest_neighbor_custom_rerank_term_field_2, nearest_neighbor_custom_rerank_term_field_3, ...)`
        2. For the lexical retriever it should look like: `rank(lexical_term, bm25_custom_rerank_term, nearest_neighbor_custom_rerank_term_field_1, nearest_neighbor_custom_rerank_term_field_2, nearest_neighbor_custom_rerank_term_field_3, ...)`
2. How do we get `bm25_custom_rerank_term`?
    1. `_get_lexical_contains_term` should be the only method that actually makes a `contains` statement, so we have no duplication of code. 
    2. Modify `_get_lexical_contains_term` such that:
        1. It can also accept an input: `attributes_to_search` where you can manually give it a list of attributes to include a `contains` term for. It should also be able to contain a value (maybe `*`) indicating that you should use `default contains` (this is for sum/max/avg). Only ONE of `attributes_to_search` or 
        `query` can be set. It can’t be both or neither. Document this behavior in the docstring.
    3. Now we make a method `_get_fields_to_bm25_rerank_by`. It accepts 
    `custom_score_keys: Set[str] = set()` as input, checks all the bm25 related keys, and returns a list of all involved fields for bm25. If there are any aggregates (sum/avg/max), just return a list with `*`. `_get_lexical_contains_term` should interpret this as the sign to use `default contains`.
3. How do we get `nearest_neighbor_custom_rerank_term_field_1`, 2, 3…?
    1. Modify `_get_individual_field_tensor_search_terms` to accept a list of attributes
    2. Make method `_get_fields_to_closeness_rerank_by` which accepts `custom_score_keys`
    3. If there is an aggregate for closeness, we must pass `_get_individual_field_tensor_search_terms` all possible tensor fields in the index.
4. Once we have all the extra ranking terms, we can put them together at the end in the format described above. 
5. So the flow would be:
    1. Get `custom_score_rerank` from `self._get_hybrid_score_modifiers`
    2. Use `custom_score_rerank` to get `custom_score_keys`
    3. Use `_get_fields_to_bm25_rerank_by` and `_get_fields_to_closeness_rerank_by` to get the fields
    4. Use `_get_lexical_contains_term` and `_get_individual_field_tensor_search_terms` with those fields to create the search terms
    5. Put them together with rank so you will have a whole new `lexical_term` and `tensor_term`

### B. Schema Changes

To support this feature, index rank profiles need:

i. New query inputs of type `tensor` for the custom score fields and their weights:

- `marqo__custom_score_mult_weights_global`
- `marqo__custom_score_add_weights_global`

ii. New match features must be added:

- `closeness` to retrieval vector per field
- `closeness` to ranking vector per field
- `bm25` score per field

```sql
rank-profile  base_rank_profile {
	match-features {
	    # Retrieval closeness
	    closeness(label, retrieval_tensor_field1)
	    closeness(label, retrieval_tensor_field2)
	
	    # Ranking closeness
	    closeness(label, ranking_tensor_field1)
	    closeness(label, ranking_tensor_field2)
	
	    # bm25
	    bm25(lexical_field1)
	    bm25(lexical_field2)
	}
}
```

### C. Java Custom Searcher Changes

After fusion, per hit in `postProcessResults`, we will do the following:

1. Extract `marqo__custom_score_add_weights_global`. For each key in this dictionary, extract the corresponding score from the hit.
    1. Break the key into its components. Here are some examples:
        1. If key is `bm25_field_variantTitle`:
            1. score type: `bm25` (singular field)
            2. field name to extract: `variantTitle`
        2. If key is `closeness_ranking_vector_sum`:
            1. score type: `closeness_ranking_vector` (aggregate)
            2. aggregate type: `sum`
    2. If the key has `field`, the field name to extract will be after it
    3. Otherwise, if it ends in `sum`, `max`, `avg`, it is an aggregate, and scores from all fields must be extracted. The operation must be performed on those scores.
    4. If score type is `bm25`, normalize using min-max normalization (becomes value between 0 and 1):

```python
bm25_normalized = (bm25 - min) / (max - min)
```

        e. Multiply this normalized score to its weight

  f. Add the weighted custom score to `global_add_modifier`

1. Extract `marqo__custom_score_mult_weights_global`. For each key in this dictionary, extract the corresponding score from the hit.
    1. Repeat same steps from 1a-1e
    2. Multiply `global_mult_modifier` to the weighted custom score

By doing this, applying global score modifiers will now also include the custom scores in the global phase. Reranking now occurs on both normal score modifiers and custom scores.

# Dependencies

N/A. Everything used in this design is already present in Marqo mainline.

# Engineering Excellence

## Consistency and Integrity

- Race condition:
    - Document A is both a lexical and tensor match
    - Document A is simultaneously being updated and searched with hybrid search
    - Update happens in between the lexical and tensor searches (possible because they are asynchronous)
    - Custom score (`bm25` or `closeness`) may be different in the lexical and tensor search
    - The custom scores from the **tensor** search will be followed

## Reliability & Resilience

- From the global score modifier field names, if a field name
- For the Vespa query,

## Scalability

N/A

## Observability

N/A

## Security

N/A. Nothing security related will change with this feature.

## Testing

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
        - with rerankDepth
        - with facets
        - with pagination
        - with collapse_fields
        - with relevance_cutoff
        - with recency boost

# Key Risks

Since all requested scores are retrieved as match features per hit and manipulated in the global phase, possible performance issues could arise if too many extra scores are requested. 

A POC and performance testing would let us find an upper limit for requested scores. We need to see how increasing the number of scores requested affects:

- memory usage
- latency

per request.

# Future Improvements (Optional)

- Integration of more ranking functions aside from bm25 and closeness
- Implementation of ranking query
- Implementation of `rerankStart`

# Cost Analysis

We can use a clone FN index in staging for performance testing.

The cost of this index is USD$42/day. Performance testing will take roughly a week.  So USD$200 in total.

# Release / Roll-out

Estimated development time: 3 weeks

2 PRs

PR 1: Vespa searcher & schema changes 

PR 2: Internal Marqo Python search logic changes

**Backwards Compatibility:**

- Rank profiles will only be changed by **adding** new match features and query inputs. Nothing will be removed, so no old features should be affected.

**Rollback**

- For this feature, it will be in a minor release and we can roll out it out to smaller customers or in staging for the science team to test and tweak before rolling out to major customers (Fashionnova).
- Upgrading a customer to the new version will require a marqo version upgrade then a schema update.
- Rolling back a customer will just require rolling back their marqo version then updating their schema.

# Impact on other components

1. **rerankDepthTensor**
    - This only affects `targetHits` of the tensor retriever. Should not affect ranking query.
2. **rerankDepth**
    - Only `rerankDepth` results will be reranked by ranking query.
3. **facets**
    - These are separate queries run in parallel to the main queries.
    - We need to make sure facets query does NOT include **ranking vector** portion, only the **retriever vector**.
4. **pagination**
    - After @Yihan Zhao ‘s fix, it should still happen at the end of the custom searcher, **after** reranking. Should not affect.
5. **collapse_fields**
    - Collapsing happens during fusion portion. Reranking will happen to fused list. Should not affect.
6. **relevance_cutoff**
    - Ranking query does not use `targetHits`, therefore is unaffected.
    - We need to make sure `probeLexicalQuery` is not changed.
7. **sort_by**
    1. Since this is another global reranker, cannot be used at the same time as ranking query.
    2. API validation: we should error out if both `sort_by` and `rankingQueryTensor` / `rankingContext` are defined.
8. **recency_boost**
    1. Should work independently of this feature. Since it already works with existing global score modifier application, nothing should change.

# Alternative Solutions Considered

**Option 1**. Solution discussed above - Include custom score field names in `scoreModifiers`.

**Option 2.** Include custom scores in a separate parameter: `rerankingCustomScores`:

- Pros: score type, aggregate type, field can be separated. More intuitive to read and extract
- Cons: Extra parameter to maintain, not intuitive that its weights combine with `scoreModifiers`.

```python
"rerankingCustomScores": {
      "add_to_score": [
          {
			      # BM25 -> "variantTitle" field
	          "score_type": "bm25",
	          "aggregate_type": "field",
	          "field_name": "variantTitle"
	          "weight": 1,
          },
          {
			      # BM25 MAX
			      "score_type": "bm25",
	          "aggregate_type": "max",
	          "weight": 1,
          },
          {
			      # closeness to ranking vector -> "variantImage" field
		        "score_type": "closeness_ranking_vector",
	          "aggregate_type": "field",
	          "field_name": "variantImage",
	          "weight": 1,
          },
          {
			      # closeness to ranking vector SUM
			      "score_type": "closeness_ranking_vector",
	          "aggregate_type": "sum",
	          "weight": 1,
          },
          {
			      # closeness to retrieval vector -> "variantImage" field
	          "score_type": "closeness_retrieval_vector",
	          "aggregate_type": "field",
	          "field_name": "variantImage",
	          "weight": 1,
          },
          {
			      # closeness to retrieval vector AVG
			      "score_type": "closeness_retrieval_vector",
	          "aggregate_type": "avg",
	          "weight": 1,
          },
      ]
  },
```

# FAQs

- When would this ranking occur?
    - This is for the global phase, after fusion is done. We also have need for reranking in the 2nd phase (per retriever), but it’s outside the scope of this project.

# References

1. Vespa closeness definition - [https://docs.vespa.ai/en/reference/ranking/rank-features.html#closeness(name)](https://docs.vespa.ai/en/reference/ranking/rank-features.html#closeness(name))
2. Multiple Nearest Neighbor searches in the same query: http://docs.vespa.ai/en/nearest-neighbor-search-guide.html#multiple-nearest-neighbor-search-operators-in-the-same-query
3. rank() query operator: https://docs.vespa.ai/en/reference/querying/yql.html
4. Using `rank()` examples: https://blog.vespa.ai/redefining-hybrid-search-possibilities-with-vespa/?_gl=1*sespsg*_gcl_au*MTIxMTgyODA3Mi4xNzY2OTgyNjcx

# Appendix

### **Appendix A: Availability of reranking methods for RRF results per phase**

|  | first-phase (node level, all candidate hits) | second-phase (node level, top N hits) | global-phase (container level) |
| --- | --- | --- | --- |
| sort_by | ❌ | ❌ | ✅ |
| recency_boost | ✅ | ❌ | ✅ |
| ranking by `bm25` (specific field) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `bm25` (SUM) | ✅ (lexical retriever) | ❌ | TO BE IMPLEMENTED |
| ranking by `bm25` (AVG) | POSSIBLE in rank profile, not implemented yet | ❌ | TO BE IMPLEMENTED |
| ranking by `bm25` (MAX) | POSSIBLE in rank profile, not implemented yet | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `retrieval_vector` (specific field) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `retrieval_vector` (SUM) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `retrieval_vector` (AVG) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `retrieval_vector` (MAX) | ✅ (tensor retriever) | for lexical-tensor  | TO BE IMPLEMENTED |
| ranking by `closeness` to `ranking_vector` (specific field) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `ranking_vector` (SUM) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `ranking_vector` (AVG) | ❌ | ❌ | TO BE IMPLEMENTED |
| ranking by `closeness` to `ranking_vector` (MAX) | Skipping for now? | ❌ | TO BE IMPLEMENTED |
| score_modifiers | ✅ | when `secondPhaseModifier` set to `true` ; or for lexical-tensor | ✅ |

# Design Review Follow-ups

Fill in any action items and next steps derived from the review.

- First phase: We’re ok with calculating bm25 scores for tensor matches,
    - Implement full feature, so it’s symmetric
    - Have feature flag to turn off calculating closeness for lexical hits, in case performance is bad
- Everyone in production only uses 1 tensor field - MAIN USE CASE
- Can rank() be applied only to a subset of matches? Check Vespa documentation if it can be applied at different phases
- Verify that score only becomes available when it’s included in the YQL query, not just added to match features
- To support ranking on fields that aren’t in retrieval would be ~1 extra day of work
- Deliver: End of next week (feb 13). Develop on 2.24 release branch
- Refresher
    - Every string field is assumed to be lexically searchable in an index when added