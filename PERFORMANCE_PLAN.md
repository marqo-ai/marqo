# Marqo Performance Improvement Plan — Codebase Changes

**Scope:** Marqo Python API + Vespa Java components only. No infrastructure (DynamoDB, Cloudflare, EKS) or Go code.

## Context & Key Findings

Marqo's current architecture runs a dedicated fleet of Python API pods per customer index. This plan covers the concrete code changes to transform it into a shared, multi-tenant API layer with less work per request.

**Key findings from codebase exploration:**

1. **RRF/hybrid search is already in Vespa** — `HybridSearcher.java` handles score fusion, re-ranking, collapse, sort-by, relevance cutoff. The high-level plan's top W2 priority is done.
2. **Config is a singleton** created at startup from env vars (`config.py`). One `VespaClient`, one `InferenceClient` for all requests.
3. **No auth exists in the codebase** — handled externally by Cloudflare Workers / RP sidecar.
4. **Document validation is Python-side** — field names, types, ranges, sizes validated in `add_documents_handler.py` before Vespa feed. This is the main candidate for Vespa migration.
5. **Routing config will come on the request** — a dataplane component (e.g. Cloudflare Workers, sidecar) will attach routing headers (Vespa endpoint, inference endpoint) per request. No backing store needed in Marqo.

**Execution order:** Workstream 2 first (reduces API surface), then Workstream 1 (multi-tenant), then Workstream 3 (Go prep).

---

## Workstream 2: Push Logic to Vespa (Do First)

### 2A. Instrument API Layer Timing

Add granular timing to identify where milliseconds are actually spent, enabling data-driven prioritization of what to move to Vespa.

**Files to modify:**
- `components/marqo/src/marqo/tensor_search/tensor_search.py` — Add timing around:
  - Query validation block (`search()` lines ~384-466)
  - Query construction / YQL generation
  - Vespa call round-trip
  - Response gathering and shaping (`gather_documents_from_response()`)
- `components/marqo/src/marqo/core/search/hybrid_search.py` — Add timing around:
  - Vectorization pipeline in `execute_search()`
  - Query building (MarqoHybridQuery construction)
  - Each Vespa round-trip (lexical, tensor, combined)
- `components/marqo/src/marqo/core/vespa_index/add_documents_handler.py` — Already has coarse timing; add per-step timing for:
  - `_validate_doc()` per batch
  - `_handle_field()` per batch
  - `_convert_to_vespa_docs()` (serialization)
  - Individual inference modality calls in `_vectorise_tensor_fields()`

**Pattern:** Use existing `RequestMetricsStore.for_request().time()` context manager. Emit via `StatsDMiddleware`.

### 2B. Move Document Validation to Vespa DocProc

Document validation (types, ranges, field names, sizes) runs in Python for every document. Moving this to a Vespa Document Processor chain eliminates Python CPU work on the hot ingestion path.

**New Java files:**
- `components/marqo/vespa/src/main/java/ai/marqo/docproc/MarqoDocumentValidator.java`
  - Vespa `DocumentProcessor` implementation
  - Validates field names (no `__` prefix, no reserved names like `_id`)
  - Validates data types match schema field types
  - Validates numerical ranges (int32/int64/float/double bounds)
  - Validates document size (configurable max bytes)
  - Returns `Progress.FAILED` with structured error on validation failure
  - Reads validation config from Vespa config definition
- `components/marqo/vespa/src/test/java/ai/marqo/docproc/MarqoDocumentValidatorTest.java`

**Vespa config/registration:**
- `components/marqo/vespa/src/main/resources/configdefinitions/marqo-validation.def` — Config definition:
  ```
  maxDocBytes int default=100000
  maxFieldNameLength int default=512
  reservedFieldPrefixes[] string
  ```
- `components/marqo/src/marqo/core/index_management/vespa_application_package.py` — Add DocProc chain to `services.xml` generation:
  ```xml
  <document-processing>
    <chain id="marqo-validation" inherits="indexing">
      <documentprocessor id="ai.marqo.docproc.MarqoDocumentValidator"/>
    </chain>
  </document-processing>
  ```

**Python-side changes (feature-flagged):**
- `components/marqo/src/marqo/tensor_search/enums.py` — Add `MARQO_ENABLE_VESPA_DOC_VALIDATION = "MARQO_ENABLE_VESPA_DOC_VALIDATION"`
- `components/marqo/src/marqo/api/configs.py` — Default `"FALSE"`
- `components/marqo/src/marqo/core/vespa_index/add_documents_handler.py`:
  - When flag enabled, `_validate_doc()` becomes a no-op (skip Python validation)
  - `_handle_field()` skips type checking (Vespa DocProc handles it)
  - Still perform duplicate ID detection in Python (ordering guarantee)
  - Map Vespa DocProc error responses back to `AddDocumentsError` in `_handle_vespa_response()`

**Error mapping:** Vespa DocProc failures return in the feed response. Modify `_handle_vespa_response()` to parse validation-specific error messages and map them to the same error format Python currently returns (preserving API contract).

### 2C. Additional Vespa Optimizations (Based on Instrumentation)

After 2A instrumentation shows where time is spent, evaluate these candidates:

**Query parameter defaults in Vespa** — If setting `approximate`, `efSearch`, and limit defaults in Python adds measurable overhead (unlikely), move to a Vespa query profile. Files: schema templates (`.sd.jinja2`) and `vespa_application_package.py`.

**Response shaping in Vespa** — If `gather_documents_from_response()` is expensive, implement field filtering in a Vespa Searcher's `fill()` phase. **Likely defer** — this is fast in Python and trivial in Go.

**Vectorization batching** — Not a Vespa change but: if instrumentation shows inference round-trips dominate, optimize batching in `_vectorise_tensor_fields()` to reduce the number of inference calls.

---

## Workstream 1: Multi-Tenant API Layer

### Design Principle: Routing on the Request

The dataplane component (Cloudflare Workers, sidecar proxy) attaches routing information to each request via HTTP headers. Marqo reads these headers and routes accordingly. No config service or backing store needed in Marqo.

**Request headers (set by dataplane):**
```
X-Marqo-Vespa-Config-Url: http://vespa-config.index-abc.svc:19071
X-Marqo-Vespa-Query-Url: http://vespa-query.index-abc.svc:8080
X-Marqo-Vespa-Document-Url: http://vespa-doc.index-abc.svc:8080
X-Marqo-Inference-Url: http://inference.index-abc.svc:8884
X-Marqo-Content-Cluster-Name: content_default
X-Marqo-Rate-Limit-Rps: 100
```

When these headers are absent, fall back to env var config (backward compatible single-tenant mode).

### 1A. Routing Header Model & Parser

**New files:**
- `components/marqo/src/marqo/core/routing/__init__.py`
- `components/marqo/src/marqo/core/routing/routing_headers.py`:
  ```python
  @dataclass
  class RoutingHeaders:
      vespa_config_url: Optional[str]
      vespa_query_url: Optional[str]
      vespa_document_url: Optional[str]
      inference_url: Optional[str]
      content_cluster_name: Optional[str]
      rate_limit_rps: Optional[int]

      @classmethod
      def from_request(cls, request: Request) -> Optional['RoutingHeaders']:
          """Extract routing headers. Returns None if no routing headers present."""
          ...

      def is_present(self) -> bool:
          """True if any routing header was provided."""
          ...
  ```

**Unit tests:** `components/marqo/tests/unit_tests/core/routing/test_routing_headers.py`

### 1B. Multi-Endpoint Client Pool

VespaClient and InferenceClient are already parameterized by URL. We need a pool that reuses client instances across requests to the same endpoint.

**New files:**
- `components/marqo/src/marqo/vespa/vespa_client_pool.py`:
  ```python
  class VespaClientPool:
      """Cache of VespaClient instances keyed by (config_url, query_url, document_url).

      Thread-safe. Evicts idle clients after configurable timeout.
      Falls back to a default client when no routing headers present.
      """
      def __init__(self, default_client: VespaClient, max_idle_seconds: int = 300):
          ...

      def get_client(self, routing: Optional[RoutingHeaders] = None) -> VespaClient:
          """Return the appropriate VespaClient for the given routing.
          Returns default_client when routing is None."""
          ...
  ```
- `components/marqo/src/marqo/core/inference/inference_client/inference_client_pool.py`:
  ```python
  class InferenceClientPool:
      """Same pattern for InferenceClient keyed by base_url."""
      ...
  ```

**Unit tests:**
- `components/marqo/tests/unit_tests/vespa/test_vespa_client_pool.py`
- `components/marqo/tests/unit_tests/core/inference/test_inference_client_pool.py`

### 1C. Thread Routing Through Request Path

The core change: each request resolves its own Vespa/inference client from routing headers.

**Files to modify:**

1. **`components/marqo/src/marqo/config.py`** — Add client pools to Config:
   ```python
   class Config:
       def __init__(self, vespa_client, inference, ...):
           ...
           self.vespa_client_pool = VespaClientPool(default_client=vespa_client)
           self.inference_pool = InferenceClientPool(default_client=inference)

       def get_vespa_client(self, routing: Optional[RoutingHeaders] = None) -> VespaClient:
           return self.vespa_client_pool.get_client(routing)

       def get_inference(self, routing: Optional[RoutingHeaders] = None) -> Inference:
           return self.inference_pool.get_client(routing)
   ```

2. **`components/marqo/src/marqo/tensor_search/api.py`** — Extract routing headers in each endpoint:
   ```python
   @app.post("/indexes/{index_name}/search")
   async def search(request: Request, index_name: str, ...):
       routing = RoutingHeaders.from_request(request)
       # Pass routing through to business logic
       ...
   ```

   Key endpoints to modify: `search`, `add_or_replace_documents`, `update_documents`, `get_document`, `get_documents`, `delete_documents`, `recommend`, `embed`, `typeahead`.

3. **Core layer files** — Add `routing: Optional[RoutingHeaders] = None` parameter:
   - `components/marqo/src/marqo/tensor_search/tensor_search.py` — `search()`, `add_documents()`, etc. use `config.get_vespa_client(routing)` instead of `config.vespa_client`
   - `components/marqo/src/marqo/core/document/document.py` — `add_documents()`, `get_documents()`, `delete_documents()`
   - `components/marqo/src/marqo/core/search/hybrid_search.py` — `execute_search()`
   - `components/marqo/src/marqo/core/search/recommender.py`
   - `components/marqo/src/marqo/core/embed/embed.py`
   - `components/marqo/src/marqo/core/typeahead/typeahead.py`
   - `components/marqo/src/marqo/core/index_management/index_management.py` — `get_index()`, `get_all_indexes()`

4. **`components/marqo/src/marqo/tensor_search/index_meta_cache.py`** — In multi-tenant mode:
   - `get_index()` needs to use the routed Vespa client to fetch index metadata
   - The background refresh thread fetches from the default Vespa endpoint (indexes visible to this pod)
   - Per-request cache misses fetch from the routed endpoint
   - Add `routing` parameter to `get_index()`

**Backward compatibility:** When `RoutingHeaders.from_request()` returns `None`, all methods fall back to `config.vespa_client` / `config.inference` (the env-var-configured defaults). Zero behavior change for existing single-tenant deployments.

### 1D. Per-Request Authentication

**New files:**
- `components/marqo/src/marqo/api/auth.py`:
  ```python
  def validate_auth(request: Request, routing: Optional[RoutingHeaders]) -> None:
      """Validate API key from Authorization header against routing config.

      No-op when routing headers are absent (single-tenant mode — auth handled externally).
      When routing is present, the dataplane has already authenticated;
      this is a secondary validation if api_key_hash is provided in headers.
      """
      ...
  ```

  Note: Since auth is currently handled by the dataplane (Cloudflare Workers / RP sidecar), and routing comes from the dataplane, the Marqo-side auth is a defense-in-depth check, not the primary gate. Keep it simple.

- `components/marqo/src/marqo/core/exceptions.py` — Add `AuthenticationError`
- `components/marqo/src/marqo/api/exceptions.py` — Map to 401/403 responses

**Files to modify:**
- `components/marqo/src/marqo/tensor_search/api.py` — Add auth validation call in relevant endpoints, add exception handler

### 1E. Request Isolation

**New files:**
- `components/marqo/src/marqo/api/rate_limiter.py`:
  ```python
  class PerIndexRateLimiter:
      """Token-bucket rate limiter keyed by index_name.

      Rate limit comes from X-Marqo-Rate-Limit-Rps header.
      When header absent, no rate limiting (single-tenant mode).
      """
      def check(self, index_name: str, routing: Optional[RoutingHeaders]) -> None:
          """Raises RateLimitExceeded if over limit."""
          ...
  ```
- `components/marqo/src/marqo/api/circuit_breaker.py`:
  ```python
  class PerEndpointCircuitBreaker:
      """Circuit breaker keyed by Vespa/inference endpoint URL.

      Opens after N consecutive failures, half-opens after cooldown.
      Prevents cascading failures when one index's backend is down.
      """
      def check(self, endpoint_url: str) -> None:
          """Raises CircuitOpen if endpoint is in open state."""
          ...

      def record_success(self, endpoint_url: str) -> None: ...
      def record_failure(self, endpoint_url: str) -> None: ...
  ```

**Files to modify:**
- `components/marqo/src/marqo/config.py` — Add rate limiter and circuit breaker instances
- `components/marqo/src/marqo/tensor_search/api.py` — Check rate limit early in endpoints, wrap Vespa/inference calls with circuit breaker
- `components/marqo/src/marqo/core/exceptions.py` — Add `RateLimitExceededError`, `CircuitBreakerOpenError`
- `components/marqo/src/marqo/api/exceptions.py` — Map to 429 / 503

---

## Workstream 3: Go Rewrite Preparation

### 3A. Formalize OpenAPI Spec

**New files:**
- `components/marqo/openapi/marqo-api.yaml` — Complete OpenAPI 3.1 spec

**Approach:**
1. Start from FastAPI's auto-generated `/openapi.json` (run the server, capture it)
2. Clean up: add descriptions, examples, formalize error response schemas
3. Add response schemas for all error types (validation, not found, rate limit, etc.)
4. Version as v1

**Source files to reference:**
- `components/marqo/src/marqo/tensor_search/api.py` — Endpoint definitions
- `components/marqo/src/marqo/api/models/` — Request/response Pydantic models
- `components/marqo/src/marqo/tensor_search/models/api_models.py` — `SearchQuery` and related
- `components/marqo/src/marqo/api/exceptions.py` — Error format

### 3B. Contract Test Suite

HTTP-level tests that validate API behavior without importing Marqo internals. These become the migration safety net.

**New files:**
- `components/marqo/tests/contract_tests/conftest.py` — Configurable base URL fixture
- `components/marqo/tests/contract_tests/test_search_contract.py` — Search endpoint: basic tensor, lexical, hybrid, error cases
- `components/marqo/tests/contract_tests/test_documents_contract.py` — CRUD: add, get, delete, batch, error cases
- `components/marqo/tests/contract_tests/test_index_contract.py` — Create, get, delete, list indexes
- `components/marqo/tests/contract_tests/test_error_contract.py` — Validate error response shape is consistent

**Approach:** Use `requests` or `httpx` library. No Marqo imports. Test response shapes (field names, types), status codes, and error formats. These exact same tests run against both Python and future Go service.

### 3C. Response Snapshot Tests

**New files:**
- `components/marqo/tests/contract_tests/snapshots/` — Golden JSON files per endpoint
- `components/marqo/tests/contract_tests/test_response_shapes.py` — Validate response JSON keys and types match snapshots (not values — those change per run)

---

## Execution Sequence

```
Phase 1: Instrument (prereq)
├── 2A: Add timing instrumentation
└── Analyze: Where are the milliseconds?

Phase 2: Push to Vespa
├── 2B: Vespa DocProc for document validation
└── 2C: Additional optimizations per instrumentation data

Phase 3: Multi-Tenant (can overlap with Phase 2)
├── 1A: Routing header model & parser
├── 1B: Multi-endpoint client pools
├── 1C: Thread routing through request path  ← biggest change
├── 1D: Per-request auth (defense-in-depth)
├── 1E: Rate limiting & circuit breakers
└── 1F: Index meta cache update

Phase 4: Go Prep (parallel with Phase 3)
├── 3A: OpenAPI spec
├── 3B: Contract test suite
└── 3C: Response snapshot tests
```

---

## Critical Files Summary

| File | Change | WS |
|------|--------|-----|
| `components/marqo/src/marqo/config.py` | Add client pools, routing-aware getters | 1 |
| `components/marqo/src/marqo/tensor_search/api.py` | Extract routing headers, auth, rate limit | 1 |
| `components/marqo/src/marqo/tensor_search/tensor_search.py` | Use routed clients, add timing | 1,2 |
| `components/marqo/src/marqo/core/vespa_index/add_documents_handler.py` | Feature-flag Python validation | 2 |
| `components/marqo/src/marqo/core/document/document.py` | Accept routing, use routed clients | 1 |
| `components/marqo/src/marqo/core/search/hybrid_search.py` | Accept routing, use routed clients | 1 |
| `components/marqo/src/marqo/core/search/recommender.py` | Accept routing | 1 |
| `components/marqo/src/marqo/core/embed/embed.py` | Accept routing | 1 |
| `components/marqo/src/marqo/core/typeahead/typeahead.py` | Accept routing | 1 |
| `components/marqo/src/marqo/core/index_management/index_management.py` | Accept routing for per-index Vespa | 1 |
| `components/marqo/src/marqo/tensor_search/index_meta_cache.py` | Route-aware cache lookup | 1 |
| `components/marqo/src/marqo/vespa/vespa_client.py` | No change (already parameterized) | — |
| `components/marqo/src/marqo/tensor_search/enums.py` | New env vars for feature flags | 1,2 |
| `components/marqo/src/marqo/api/configs.py` | Defaults for new env vars | 1,2 |
| `components/marqo/vespa/src/main/java/...HybridSearcher.java` | Already has RRF — no change | — |
| `components/marqo/vespa/src/main/java/.../MarqoDocumentValidator.java` | NEW: Vespa DocProc | 2 |
| `components/marqo/src/marqo/core/index_management/vespa_application_package.py` | Register DocProc chain | 2 |

## New Files Summary

| File | Purpose | WS |
|------|---------|-----|
| `src/marqo/core/routing/__init__.py` | Package | 1 |
| `src/marqo/core/routing/routing_headers.py` | Parse routing from request headers | 1 |
| `src/marqo/vespa/vespa_client_pool.py` | Cached VespaClient instances per endpoint | 1 |
| `src/marqo/core/inference/inference_client/inference_client_pool.py` | Cached InferenceClient per endpoint | 1 |
| `src/marqo/api/auth.py` | Per-request auth validation | 1 |
| `src/marqo/api/rate_limiter.py` | Per-index rate limiting | 1 |
| `src/marqo/api/circuit_breaker.py` | Per-endpoint circuit breaker | 1 |
| `vespa/.../docproc/MarqoDocumentValidator.java` | Vespa-side doc validation | 2 |
| `openapi/marqo-api.yaml` | Formalized API contract | 3 |
| `tests/contract_tests/` | Language-agnostic API tests | 3 |

(All paths relative to `components/marqo/`)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Threading routing through every call path is invasive | Use `Optional[RoutingHeaders] = None` default — existing callers unchanged |
| Client pool memory growth with many unique endpoints | LRU eviction with configurable max size; monitor with StatsD |
| Vespa DocProc error messages differ from Python errors | Map errors in `_handle_vespa_response()`, add contract tests |
| Index meta cache doesn't know about routed endpoints | Per-request cache miss fetches from routed endpoint; background refresh covers default |
| Rate limiter is per-pod, not global | Acceptable for initial rollout; global rate limiting can be added at dataplane |

## Verification Plan

**Workstream 2:**
- Existing unit tests pass: `PYTHONPATH=./src pytest tests/unit_tests/`
- Vespa Java tests: `cd vespa && mvn test`
- Integration: Add docs with flag on/off, verify identical responses
- Benchmark: Measure add_documents latency with/without DocProc

**Workstream 1:**
- Unit tests for routing headers, client pool, auth, rate limiter, circuit breaker
- Integration (single-tenant): No routing headers → identical behavior to today
- Integration (multi-tenant): Two indexes with different routing headers → correct endpoint routing
- Load test: Concurrent requests with different routing, verify isolation

**Workstream 3:**
- OpenAPI spec validates: `openapi-generator validate`
- Contract tests pass against running Python service
- Same tests become Go migration gate
