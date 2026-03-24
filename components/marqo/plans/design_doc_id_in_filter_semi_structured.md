## Summary

Enable the `IN` filter operator for the `_id` field on **semi-structured indexes**. This allows users to pass a list of document IDs to restrict search results to a specific set of documents (a "wishlist" pattern). The `IN` operator already works for structured indexes; this extends support to semi-structured indexes, which are the default index type.

**Scope:** `_id` field only. General field IN support for semi-structured indexes is deferred due to a Vespa limitation ([vespa-engine/vespa#30711](https://github.com/vespa-engine/vespa/issues/30711)).

---

## Motivation / Problem Statement

A customer wants to input a list of product `_id` values to filter search results (a "wishlist" feature). Required list size: up to **1,000 IDs**. Max we’ve seen: **7,670 IDs**.

Currently, semi-structured indexes reject IN filters with:

```
InvalidArgumentError("The 'IN' filter keyword is not yet supported for unstructured indexes")
```

The only alternative is chaining `_id:X OR _id:Y OR ...`, which hits a **Vespa StackOverflow at ~38 IDs** due to recursive ANTLR parsing of deeply nested OR trees.

---

## Background & Investigation

### Why was this not implemented initially?

In March 2024, the team attempted to use the `in` operator inside `sameElement()` queries on composite map fields:

```
marqo__short_string_fields contains sameElement(key contains "field", value in ('v1', 'v2'))
```

This failed with: *"The in operator is only supported for integer and string fields. The field value is not of these types"*

The error message is **misleading**. The real issue is that **Vespa does not support `in` inside `sameElement()`**. Per [Vespa query language docs](https://docs.vespa.ai/en/reference/query-language-reference.html), the only operators allowed inside `sameElement()` are: `and`, `equiv`, `near`, `onear`, `or`, `rank`, and `phrase`. This is tracked as [vespa-engine/vespa#30711](https://github.com/vespa-engine/vespa/issues/30711), still **open** with milestone "Later" as of July 2025.

### Why can we do it now?

The `_id` field (`marqo__id`) is **not** stored in the composite `sameElement` map fields. It is a **direct top-level attribute** field in the Vespa schema:

```
field marqo__id type string {
    indexing: attribute | summary
    attribute: fast-search
    rank: filter
}
```

The `in` operator works on direct attribute string fields with no issues.

### Empirical Testing Results (Local M2 Mac) (Vespa 8.513.17)

| Approach | Works? | Limit Found |
| --- | --- | --- |
| `_id:X OR _id:Y OR ...` (OR chaining) | Yes | **~38 IDs** (StackOverflow) |
| `marqo__id in ("id1", "id2", ...)` (native IN) | Yes | **100,000+** (no limit found) |
| `sameElement(key, value in (...))` | **No** | N/A (Vespa rejects) |

The native `in` operator on `marqo__id` fully satisfies the 1,000+ requirements

---

### Functional Requirements (7)

- FR-1–FR-5: Core behavior — _id IN (...) works on semi-structured, across all search methods,
composes with AND/OR/NOT
- FR-6: Non-_id fields still raise InvalidArgumentError
- FR-7: Special character escaping

### Non-Functional Requirements (5)

- NFR-1: 1,000+ IDs (target 10,000+)
- NFR-2: No latency regression for non-IN queries
- NFR-3: No schema/deployment changes
- NFR-4: Backwards compatible
- NFR-5: No new infrastructure dependencies

## Architecture

### Current State

Semi-structured index filter generation lives in `semi_structured_vespa_index.py` inside the `_get_filter_term()` method. This method walks the parsed filter AST and generates Vespa YQL filter clauses. The `InTerm` branch currently raises `InvalidArgumentError`.

The filter parser (`search_filter.py`) already parses `InTerm` nodes for all index types. Structured indexes already convert `InTerm` to `field in (v1, v2, ...)` YQL.

### Proposed Change

Replace the `raise InvalidArgumentError` at `semi_structured_vespa_index.py:1098` with a handler:

- **If field is `_id`:** generate `marqo__id in ("id1", "id2", ...)`
- **Otherwise:** raise `InvalidArgumentError` with an updated, more specific message

### Code Changes

**File:** `src/marqo/core/semi_structured_vespa_index/semi_structured_vespa_index.py`

Add a `generate_in_filter_string` function inside `_get_filter_term()`, alongside the existing `generate_equality_filter_string` and `generate_range_filter_string`:

```python
def generate_in_filter_string(node: search_filter.InTerm) -> str:
    if node.field == MARQO_DOC_ID:
        escaped_values = ', '.join(
            f'"{self.escape(v)}"' for v in node.value_list
        )
        return f'{VESPA_FIELD_ID} in ({escaped_values})'
    else:
        raise InvalidArgumentError(
            "The 'IN' filter keyword is only supported for the '_id' field "
            "on semi-structured indexes."
        )
```

Wire it into `tree_to_filter_string`:

```python
elif isinstance(node, search_filter.InTerm):
    return generate_in_filter_string(node)
```

**Files changed:**

| File | Change |
| --- | --- |
| `src/marqo/core/semi_structured_vespa_index/semi_structured_vespa_index.py` | Replace `raise` with `generate_in_filter_string` handler |
| `tests/integ_tests/tensor_search/search/test_search_combined.py` | Update error message assertion, add `_id` IN success tests for unstructured |
| New: `tests/integ_tests/tensor_search/search/test_id_in_filter_semi_structured.py` | Dedicated integration test with 1,000 documents |

---

## Feature Interactions

### Collapse Fields

**No issue.** Filters (including IN) are combined with collapse filters via AND in the WHERE clause (`semi_structured_vespa_index.py:504-506`). Collapse itself already uses `field in (...)` syntax internally for its sort-by follow-up queries (`collapse_search.py:239-242`). The `_id` IN filter reduces the candidate pool *before* collapse grouping occurs — only wishlist documents are considered for collapse groups.

### Facets

**No issue.** Facet queries apply the same `filter_term` to all facet subqueries (`semi_structured_vespa_index.py:792-796`). The `exclude_terms` mechanism (which regenerates filters excluding specific terms for facet counting) works at the filter tree node level and handles `InTerm` nodes transparently — if an `InTerm` is in the exclude list, it is skipped during filter string generation (`semi_structured_vespa_index.py:1055-1057`).

### Hybrid Search (Tensor + Lexical)

**No issue.** The `filter_term` is computed once and appended to both the tensor YQL and lexical YQL subqueries identically (`semi_structured_vespa_index.py:586-587`). The IN filter restricts both retrieval branches equally.

### Score Modifiers / Custom Score Reranking

**No interaction.** Filters operate in the WHERE clause to reduce candidates. Score modifiers and custom score reranking operate in the ranking phase on the already-filtered result set. They are independent mechanisms.

### Relevance Cutoff

**No issue.** The probe query used for relevance cutoff threshold calculation uses the same `filter_term` (`semi_structured_vespa_index.py:512-514`). The IN filter is included in the probe, ensuring the threshold is computed only over wishlist documents.

### Recency Scoring

**No interaction.** Recency parameters are passed as query features to the rank profile and are independent of filter clauses.

---

## Edge Cases / Error Handling

| Scenario | Behavior | Detail |
| --- | --- | --- |
| `_id IN (id1, id2)` on semi-structured | Generates `marqo__id in ("id1", "id2")` | Happy path |
| `color IN (red, blue)` on semi-structured | `InvalidArgumentError` | Updated message: "only supported for the `_id` field on semi-structured indexes" |
| `_id IN ()` (empty list) | Parses as `InTerm` with `value_list = ['']` (one empty string element) | The filter parser does not distinguish an empty IN list from one containing a single empty string. This is identical to structured index behavior. Vespa receives `marqo__id in ("")` and returns 0 hits (no document has an empty `_id`). |
| `_id IN (id with "quotes")` | Correctly escaped via `self.escape()` | The escape method (defined in `vespa_index.py:303-316`) prefixes `"` and `\\` with a backslash. So `_id IN (he"llo)` generates `marqo__id in ("he\\"llo")`. This is the same escaping used by all existing filter types. |
| `NOT _id IN (id1, id2)` | Generates `!(marqo__id in ("id1", "id2"))` | Exclusion — returns all docs *except* id1 and id2 |
| `_id IN (id1) AND color:red` | Both clauses combined with AND | The IN clause restricts to `id1`, the equality clause further filters by `color` |

---

## Engineering Excellence

### Performance

The native Vespa `in` operator on an attribute field with `fast-search` is highly optimized. Tested to 100,000+ values with no degradation. No additional queries, network calls, or post-processing are introduced.

### Backwards Compatibility

Fully backwards compatible. The only behavioral change is that a previously-erroring filter string (`_id IN (...)` on semi-structured) now succeeds. All other filter behavior is unchanged.

---

## Observability

N/A — No new metrics, logs, or monitoring needed. Filter queries are already logged by the existing `QueryLogger`. Failed queries already surface through standard error handling.

---

## Security

N/A — No new attack surface. Document ID values in filter strings are escaped using the existing `self.escape()` method, which handles `"` and `\\` characters. No user input is interpolated into queries without escaping.

---

## Testing Plan

### Unit Tests

Add to `tests/unit_tests/marqo/core/semi_structured_vespa_index/`:

- `_id IN (id1, id2, id3)` → generates `marqo__id in ("id1", "id2", "id3")`
- `_id IN (id_with_"quotes")` → proper escaping: `marqo__id in ("id_with_\\"quotes")`
- `_id IN (single_id)` → no trailing comma: `marqo__id in ("single_id")`
- `NOT _id IN (id1, id2)` → generates `!(marqo__id in ("id1", "id2"))`
- `_id IN (id1) AND color:red` → combined filter with AND
- `color IN (red, blue)` → raises `InvalidArgumentError`
- `int_field IN (1, 2)` → raises `InvalidArgumentError`

### Integration Tests

**New file:** `tests/integ_tests/tensor_search/search/test_id_in_filter_semi_structured.py`

- Create semi-structured index with `random` model (no real inference)
- Add **1,000 documents**
- `_id IN (subset)` with TENSOR search → verify only specified docs returned
- `_id IN (subset)` with LEXICAL search → verify correct results
- `_id IN (subset)` with HYBRID search → verify correct results
- `NOT _id IN (subset)` → verify exclusion
- `_id IN (id1) AND field:value` → combined filter
- `_id IN (all_1000_ids)` → large list, no error
- `_id IN ()` (empty) → 0 hits, no error

**Update existing:** `tests/integ_tests/tensor_search/search/test_search_combined.py`

- Update `test_filter_unstructured_index_in_keyword_fails` — `_id IN (...)` should now **succeed**, while general field IN should still fail with updated error message

### API Tests

- End-to-end test: create semi-structured index, add docs, search with `_id IN (...)` via HTTP
- Verify interaction with collapse: `_id IN (...)` + collapse parameter
- Verify interaction with facets: `_id IN (...)` + facets parameter

---

## Rollout

- No schema changes required
- No Vespa searcher (Java) changes — pure Python-side filter generation
- No migration needed — works on all existing semi-structured indexes
- Feature is available immediately on deployment with no feature flag

---

## Open Questions / Future Work

1. **General field IN for semi-structured** — blocked by [vespa-engine/vespa#30711](https://github.com/vespa-engine/vespa/issues/30711). The only correct workaround (OR of `sameElement` clauses) hits the StackOverflow limit at ~38 values. This can be revisited if Vespa adds `in` support inside `sameElement`.
2. **Empty IN list handling** — The parser currently treats `_id IN ()` as a single empty string value rather than a truly empty list. This is consistent with structured index behavior but could be tightened with explicit validation if desired.