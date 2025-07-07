from typing import Dict, List, Optional, Tuple
import unittest

from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

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
    def setUp(self):
        self.stub = _StubStatsD()
        self.client_ctx = TestClient(_build_basic_app(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

    def tearDown(self):
        self.client_ctx.__exit__(None, None, None)
        self.stub.sent.clear()

    def test_root_request_metrics(self):
        """Health-check path ‘/’ should NOT emit metrics."""
        self.client.get("/")
        self.assertEqual(_extract(self.stub, "increment", "requests.completed"), [])
        self.assertFalse(any(m.startswith("marqo_processing_time") for _, m in self.stub.sent))

    def test_search_metrics(self):
        resp = self.client.post("/indexes/foo/search")
        self.assertEqual(resp.status_code, 200)

        self.assertTrue(any(m.startswith("search_processing_time") for _, m in self.stub.sent))
        self.assertTrue(any(
            "path:/indexes/foo/search" in m and "method:POST" in m
            for m in _extract(self.stub, "increment", "requests.completed")
        ))
        self.assertTrue(any("status_code:2XX" in m for m in _extract(self.stub, "increment", "requests.completed")))

    def test_index_docs_metrics_and_headers(self):
        self.client.post("/indexes/foo/documents")

        self.assertTrue(any(m.startswith("index_processing_time") for _, m in self.stub.sent))
        incs = _extract(self.stub, "increment")
        self.assertTrue(any(m.startswith("x-count-success:5") for m in incs))
        self.assertTrue(any(m.startswith("x-count-failure:1") for m in incs))
        self.assertTrue(any(m.startswith("x-count-error:0") for m in incs))

    def test_requests_completed_path_sanitised(self):
        # Change app context to include document GET
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_docs_and_fail(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.get("/indexes/foo/documents/abc123")
        msgs = _extract(self.stub, "increment", "requests.completed")
        self.assertTrue(any("path:/indexes/foo/documents/<document_id>" in m for m in msgs))
        self.assertFalse(any("abc123" in m for m in msgs))

    def test_requests_completed_5xx(self):
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_docs_and_fail(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.get("/fail")
        msgs = _extract(self.stub, "increment", "requests.completed")
        self.assertTrue(any("status_code:5XX" in m for m in msgs))

    def test_patch_docs_metrics_and_malformed_headers(self):
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_patch_and_bad_headers(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.patch("/indexes/foo/documents")

        self.assertTrue(any(k == "timing" and m.startswith("index_processing_time") for k, m in self.stub.sent))
        self.assertFalse(any(m.startswith("x-count-success") for k, m in self.stub.sent))

    def test_headers_with_empty_strings_dont_crash(self):
        self.client_ctx.__exit__(None, None, None)
        self.client_ctx = TestClient(_app_with_patch_and_bad_headers(self.stub))
        self.client = self.client_ctx.__enter__()
        self.stub.sent.clear()

        self.client.patch("/indexes/foo/documents")

        self.assertTrue(any(k == "timing" and m.startswith("index_processing_time") for k, m in self.stub.sent))
        self.assertFalse(any(m.startswith("x-count-success") for k, m in self.stub.sent))

    def test_requests_completed_4xx(self):
        resp = self.client.get("/nonexistent/path")
        self.assertEqual(resp.status_code, 404)

        msgs = _extract(self.stub, "increment", "requests.completed")
        self.assertTrue(any("status_code:4XX" in m for m in msgs))