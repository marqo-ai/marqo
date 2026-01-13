# Investigation: Zero Hits with `sortBy` + `relevanceCutoff` in Pure Tensor Search

## Issue Summary

When using pure tensor search (`retrievalMethod: "tensor"`, `rankingMethod: "tensor"`) with weighted `queryTensor` dictionaries (e.g., `{"office": 1, "low quality": -0.5}`) and combining `sortBy` + `relevanceCutoff`, the search returns 0 results.

## Root Cause

The `relevanceCutoff` feature relies on a **lexical probe search** to determine the number of relevant candidates. With pure tensor search and no lexical query component, the probe search has nothing to match against.

### Detailed Flow

1. **Query Construction (Python)**
   - When using `queryTensor` with a weighted dictionary, no lexical query text is provided
   - `_get_lexical_search_term()` returns `'false'` when `or_phrases` and `and_phrases` are empty
   - The lexical YQL becomes: `select * from schema where (false)`

2. **Probe Search (Vespa Searcher)**
   - `HybridSearcher.java:207-223` creates a probe lexical query using `marqo__yql.lexical`
   - The `where (false)` clause matches **zero documents**
   - `probeCandidates` = 0, `relevantCandidates` = 0

3. **Query Modification**
   - `HybridSearcher.java:481-488` sets `newHits = Math.min(relevantCandidates, limit+offset)` = 0
   - `targetHits` in tensor YQL is reduced to 1 (minimum normalization)

4. **Result**: Search returns 0 or at most 1 result

## Key Files

| File | Location | Role |
|------|----------|------|
| `HybridSearcher.java` | `vespa/src/main/java/ai/marqo/search/HybridSearcher.java:207-223` | Probe search always uses lexical |
| `_get_lexical_search_term` | `core/unstructured_vespa_index/unstructured_vespa_index.py:243-244` | Returns `'false'` for empty queries |
| `hybrid_search.py` | `core/search/hybrid_search.py:315-319` | No lexical phrases for tensor-only search |

## Impacted Scenarios

| Retrieval | Ranking | Has Lexical Query? | `relevanceCutoff` Works? |
|-----------|---------|-------------------|--------------------------|
| tensor | tensor | No (queryTensor only) | **Broken** |
| tensor | tensor | Yes (q or queryLexical) | Works |
| tensor | lexical | Yes (requires lexical) | Works |
| lexical | tensor | Yes (requires lexical) | Works |
| disjunction | rrf | Yes (requires lexical) | Works |

## Recommended Fixes

### Option 1: Validation (Safest, Quick)

Add validation in `api_models.py` to prevent using `relevanceCutoff` with pure tensor searches that lack a lexical query:

```python
@root_validator(pre=False)
def _validate_relevance_cutoff_requires_lexical_query(cls, values):
    """Validate that relevanceCutoff requires a lexical query component"""
    relevance_cutoff = values.get('relevance_cutoff')
    hybrid_params = values.get('hybridParameters')
    q = values.get('q')

    if relevance_cutoff is not None and hybrid_params is not None:
        # Pure tensor search without lexical query
        if (hybrid_params.retrievalMethod == 'tensor' and
            hybrid_params.rankingMethod == 'tensor' and
            q is None and
            hybrid_params.queryLexical is None):
            raise ValueError(
                "relevanceCutoff cannot be used with pure tensor search (retrievalMethod='tensor', "
                "rankingMethod='tensor') when no lexical query is provided. Either provide 'q' or "
                "'hybridParameters.queryLexical', or use 'sortBy' without 'relevanceCutoff'."
            )
    return values
```

### Option 2: Tensor Probe Search (More Work, Better UX)

Modify `HybridSearcher.java` to use tensor search for the probe when no lexical query is available:

1. Create a `createProbeTensorQuery()` method
2. Detect when lexical YQL is `where (false)`
3. Use tensor results for relevance cutoff calculation

### Option 3: Skip Cutoff for Pure Tensor

If no lexical query is available, skip the relevance cutoff entirely and fall back to standard behavior.

## Workaround for Users

Until a fix is implemented, users can:

1. **Use `sortBy` without `relevanceCutoff`** - sorting will still work
2. **Provide a lexical query** - even a broad one like `"*"` might work
3. **Use RRF retrieval method** - which requires both lexical and tensor queries

## References

- Slack thread: https://marqo-ai.slack.com/archives/C092710EZ37/p1768341430366019
- Related code:
  - `components/marqo/vespa/src/main/java/ai/marqo/search/HybridSearcher.java`
  - `components/marqo/src/marqo/core/search/hybrid_search.py`
  - `components/marqo/src/marqo/tensor_search/models/api_models.py`
