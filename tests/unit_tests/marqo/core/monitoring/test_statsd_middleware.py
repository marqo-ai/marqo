from http import HTTPStatus

import unittest
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from typing import Dict, List, Optional, Tuple

from marqo.core.monitoring import statsd_middleware as sm


class _StubStatsD:
    """A stub StatsD client that captures sent metrics for testing."""

    def __init__(self) -> None:
        self.sent: List[Tuple[str, str]] = []

    def _fmt(self, metric: str, value: int, suff: str, tags: Optional[Dict[str, str]]) -> str:
        tag_str = "" if not tags else "|#" + ",".join(f"{k}:{v}" for k, v in tags.items())
        return f"{metric}:{value}|{suff}{tag_str}"

    def increment(self, metric: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        self.sent.append(("increment", self._fmt(metric, value, "c", tags)))

    def timing(self, metric: str, value_ms: int, tags: Optional[Dict[str, str]] = None):
        self.sent.append(("timing", self._fmt(metric, value_ms, "ms", tags)))


def _extract(stub: _StubStatsD, kind: str, prefix: str = "") -> List[str]:
    """Return payloads captured by the stub that match kind & prefix."""
    return [m for k, m in stub.sent if k == kind and m.startswith(prefix)]


def _build_basic_app(statsd_stub: _StubStatsD) -> FastAPI:
    """Build a basic FastAPI app with StatsDMiddleware for testing."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware, statsd_client=statsd_stub)

    @app.get("/")
    def root():
        return JSONResponse({"ping": "pong"})

    @app.post("/indexes/{name}/search")
    def search(name: str):
        return JSONResponse({"ok": True})

    @app.post("/indexes/{name}/documents")
    def add_docs(name: str):
        return JSONResponse(
            {"ok": True},
            headers={"x-count-success": "5", "x-count-failure": "1", "x-count-error": "0"},
        )

    return app


def _app_with_docs_and_fail(statsd_stub: _StubStatsD) -> FastAPI:
    """App with a GET endpoint that sanitises document IDs and a failing endpoint."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware, statsd_client=statsd_stub)

    @app.get("/indexes/{name}/documents/{doc_id}")
    def get_doc(name: str, doc_id: str):
        return JSONResponse({"id": doc_id})

    @app.get("/fail")
    def always_fail():
        raise HTTPException(status_code=503, detail="boom")

    return app


def _app_with_patch_and_bad_headers(statsd_stub: _StubStatsD) -> FastAPI:
    """App with a PATCH endpoint that returns malformed headers."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware, statsd_client=statsd_stub)

    @app.patch("/indexes/{name}/documents")
    def patch_docs(name: str):
        return JSONResponse(
            {"ok": True},
            headers={"x-count-success": "NaN", "x-count-failure": "oops", "x-count-error": ""},
        )

    return app


class TestStatsDMiddleware(unittest.TestCase):
    """Unit tests for StatsDMiddleware to ensure it emits expected metrics."""
    def setUp(self):
        self.stub = _StubStatsD()
        self.client_ctx = TestClient(_build_basic_app(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

    def tearDown(self):
        """Clean up the client context and reset the stub."""
        self.client_ctx.__exit__(None, None, None)
        self.stub.sent.clear()

    def test_root_request_metrics(self):
        """Health-check path ‘/’ should NOT emit metrics."""
        self.client.get("/")
        self.assertEqual(self.stub.sent, [])

    def test_search_metrics(self):
        """Search emits request.duration_ms with proper tags."""
        resp = self.client.post("/indexes/foo/search")
        self.assertEqual(resp.status_code, 200)

        timings = _extract(self.stub, "timing", "request.duration_ms")
        self.assertTrue(any("path:/indexes/foo/search" in m and "method:POST" in m for m in timings))
        self.assertTrue(any("status_code:200" in m for m in timings))

    def test_index_docs_metrics_and_headers(self):
        """Indexing documents emits request.duration_ms and batch.* counters."""
        self.client.post("/indexes/foo/documents")

        self.assertTrue(any(m.startswith("request.duration_ms") for k, m in self.stub.sent if k == "timing"))
        incs = _extract(self.stub, "increment")
        self.assertTrue(any("batch.success:5" in m for m in incs))
        self.assertTrue(any("batch.failure:1" in m for m in incs))
        self.assertTrue(any("batch.error:0" in m for m in incs))

    def test_path_sanitisation(self):
        """Ensure document ID is redacted in request.duration_ms tags."""
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_docs_and_fail(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.get("/indexes/foo/documents/abc123")
        timings = _extract(self.stub, "timing", "request.duration_ms")
        self.assertTrue(any("path:/indexes/foo/documents/<document_id>" in m for m in timings))
        self.assertFalse(any("abc123" in m for m in timings))

    def test_duration_metrics_on_5xx(self):
        """Ensure 5XX responses emit duration_ms with correct tag."""
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_docs_and_fail(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.get("/fail")
        timings = _extract(self.stub, "timing", "request.duration_ms")
        self.assertTrue(any("status_code:503" in m for m in timings))

    def test_patch_docs_malformed_headers(self):
        """PATCH requests emit duration, malformed x-count headers are ignored."""
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_patch_and_bad_headers(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.patch("/indexes/foo/documents")

        self.assertTrue(any(m.startswith("request.duration_ms") for k, m in self.stub.sent if k == "timing"))
        self.assertFalse(any("batch.success" in m or "batch.failure" in m or "batch.error" in m
                             for k, m in self.stub.sent if k == "increment"))

    def test_empty_headers_dont_crash(self):
        """PATCH requests with empty headers should not raise exceptions or send bad metrics."""
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_patch_and_bad_headers(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.patch("/indexes/foo/documents")

        self.assertTrue(any(k == "timing" and m.startswith("request.duration_ms") for k, m in self.stub.sent))
        self.assertFalse(any("batch." in m for k, m in self.stub.sent if k == "increment"))

    def test_duration_metrics_on_4xx(self):
        """404s and other client errors still emit duration metric."""
        resp = self.client.get("/nonexistent/path")
        self.assertEqual(resp.status_code, 404)

        timings = _extract(self.stub, "timing", "request.duration_ms")
        self.assertTrue(any("status_code:404" in m for m in timings))

    def test_sanitize_path(self):
        """Test the _sanitize_path method for various scenarios."""
        sanitize = sm.StatsDMiddleware._sanitize_path

        self.assertEqual(
            sanitize("/indexes/foo/documents/abc123"),
            "/indexes/foo/documents/<document_id>",
        )
        self.assertEqual(
            sanitize("/indexes/foo/documents/delete-batch"),
            "/indexes/foo/documents/delete-batch",
        )

        self.assertEqual(
            sanitize("/indexes/foo/documents/get-batch"),
            "/indexes/foo/documents/get-batch",
        )

    def test_status_code_tag_is_string_integer(self):
        """Ensure status_code tag is always a plain integer string, not an Enum representation."""
        # Create a mock response with an IntEnum-like status_code
        # This simulates cases where status_code could be an IntEnum subclass
        # Create a new stub and middleware for this test
        test_cass = [
            (HTTPStatus.OK, "200"),
            (HTTPStatus.BAD_REQUEST, "400"),
            (HTTPStatus.INTERNAL_SERVER_ERROR, "500")
        ]
        for return_enum, expected_str in test_cass:
            with self.subTest(msg=f"status_code tag for {return_enum}"):
                stub = _StubStatsD()
                app = FastAPI()
                app.add_middleware(sm.StatsDMiddleware, statsd_client=stub)

                @app.get("/test")
                async def test_endpoint():
                    # Create a response with IntEnum status_code
                    response = JSONResponse({"message": "test"})
                    response.status_code = return_enum
                    return response

                with TestClient(app) as client:
                    client.get("/test")

                    timings = _extract(stub, "timing", "request.duration_ms")
                    self.assertTrue(any(f"status_code:{expected_str}" in m for m in timings),
                                   f"Expected 'status_code:{expected_str}' in metrics, got: {timings}")
                    # Ensure it doesn't contain enum representation
                    for timing in timings:
                        if "status_code:" in timing:
                            status_part = [part for part in timing.split("|#")[1].split(",") if "status_code:" in part][0]
                            status_value = status_part.split(":")[1]
                            self.assertEqual(status_value, expected_str,
                                           f"Status code should be '{expected_str}', got '{status_value}'")