import types
from typing import Dict, List, Tuple, Optional

import pytest
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
import importlib

from marqo.core.monitoring import statsd_client as sc
from marqo.core.monitoring import statsd_middleware as sm


class _StubStatsD:
    """Captures increment/timing payloads without opening a socket."""

    def __init__(self) -> None:
        self.sent: List[Tuple[str, str]] = []

    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        tag_str = "" if not tags else "|#" + ",".join(f"{k}:{v}" for k, v in tags.items())
        self.sent.append(("increment", f"{metric}:{value}|c{tag_str}"))

    def timing(
        self,
        metric: str,
        value_ms: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        tag_str = "" if not tags else "|#" + ",".join(f"{k}:{v}" for k, v in tags.items())
        self.sent.append(("timing", f"{metric}:{value_ms}|ms{tag_str}"))


@pytest.fixture(autouse=True)
def stub_statsd(monkeypatch):
    """
    Replace both get_client (in statsd_client and statsd_middleware)
    with a stub so every call in the middleware is captured.
    """
    stub = _StubStatsD()
    monkeypatch.setattr(sc, "get_client", lambda: stub, raising=False)
    monkeypatch.setattr(sm, "get_client", lambda: stub, raising=False)
    yield stub


def test_encode_tags_and_formatting():
    """Test the _encode_tags helper and its formatting."""
    assert sc.StatsDClient._encode_tags(None) == ""
    encoded = sc.StatsDClient._encode_tags({"a": "1", "b": "2"})
    assert encoded in ("|#a:1,b:2", "|#b:2,a:1")  # dict order not guaranteed


def test_send_format_increment_and_timing(monkeypatch):
    # test the _send helpers directly
    captured: List[bytes] = []

    # patch the _send helper (easier than socket.sendto)
    monkeypatch.setattr(
        sc.StatsDClient,
        "_send",
        lambda self, msg: captured.append(msg.encode()),
        raising=True,
    )

    client = sc.StatsDClient(host="127.0.0.1", port=9999)
    client.increment("foo", 3, {"k": "v"})
    client.timing("bar", 123)

    assert b"foo:3|c|#k:v" in captured
    assert b"bar:123|ms" in captured[1]


def test_sanitize_path():
    assert sm._sanitize_path("/indexes/my/documents/abc123") == "/indexes/my/documents/<document_id>"
    assert sm._sanitize_path("/indexes/x/search") == "/indexes/x/search"


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with StatsD middleware."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware)

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
            headers={
                "x-count-success": "5",
                "x-count-failure": "1",
                "x-count-error": "0",
            },
        )

    return app


@pytest.fixture(scope="function")
def client():
    """Fixture to create a TestClient for the FastAPI app with StatsD middleware."""
    with TestClient(_build_app()) as c:
        yield c


def _extract(stub: _StubStatsD, kind: str, prefix: str) -> List[str]:
    """ Extract messages from the stub that match the given kind and prefix. """
    return [msg for k, msg in stub.sent if k == kind and msg.startswith(prefix)]


def test_root_request_metrics(client, stub_statsd):
    """Health-check path (‘/’) should not emit any StatsD packets."""
    client.get("/")

    # No increments or timings expected
    assert _extract(stub_statsd, "increment", "requests.completed") == []
    assert not any(m.startswith("marqo_processing_time") for _, m in stub_statsd.sent)


def test_search_metrics(client, stub_statsd):
    """Test that search requests emit the correct StatsD metrics."""
    stub_statsd.sent.clear()
    client.post("/indexes/foo/search")

    assert any(m.startswith("search_processing_time") for _, m in stub_statsd.sent)
    assert any(
        "path:/indexes/foo/search" in m and "method:POST" in m
        for m in _extract(stub_statsd, "increment", "requests.completed")
    )


def test_index_docs_metrics_and_headers(client, stub_statsd):
    """Test that adding documents emits the correct StatsD metrics and headers."""
    stub_statsd.sent.clear()
    client.post("/indexes/foo/documents")

    assert any(m.startswith("index_processing_time") for _, m in stub_statsd.sent)
    incs = _extract(stub_statsd, "increment", "")
    assert any(m.startswith("x-count-success:5") for m in incs)
    assert any(m.startswith("x-count-failure:1") for m in incs)
    assert any(m.startswith("x-count-error:0") for m in incs)


def _msgs(stub, kind: str, prefix: str = "") -> List[str]:
    """Return payloads captured by the _StubStatsD that match kind & prefix."""
    return [msg for k, msg in stub.sent if k == kind and msg.startswith(prefix)]


def _app_with_docs_and_fail() -> FastAPI:
    """Build a FastAPI app with document endpoints and a failure endpoint."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware)

    @app.get("/indexes/{name}/documents/{doc_id}")
    def get_doc(name: str, doc_id: str):
        return JSONResponse({"id": doc_id})

    @app.get("/fail")
    def always_fail():
        raise HTTPException(status_code=503, detail="boom")

    return app


@pytest.fixture(scope="function")
def client_extra():
    """Fixture to create a TestClient for the FastAPI app with document endpoints."""
    with TestClient(_app_with_docs_and_fail()) as c:
        yield c


def test_requests_completed_path_sanitised(client_extra, stub_statsd):
    """Test that requests to document endpoints have their path sanitized."""
    stub_statsd.sent.clear()
    client_extra.get("/indexes/foo/documents/abc123")

    msgs = _msgs(stub_statsd, "increment", "requests.completed")
    # Look for the variant that contains full tag set (path+method+status)
    assert any("path:/indexes/foo/documents/<document_id>" in m for m in msgs)
    # Ensure the raw ID isn't present
    assert not any("abc123" in m for m in msgs)


def test_requests_completed_5xx(client_extra, stub_statsd):
    """Test that requests to a failing endpoint emit 5XX status code."""
    stub_statsd.sent.clear()
    client_extra.get("/fail")

    msgs = _msgs(stub_statsd, "increment", "requests.completed")
    assert any("status_code:5XX" in m for m in msgs)


def _app_with_patch_and_bad_headers() -> FastAPI:
    """Build a FastAPI app with a PATCH endpoint that returns malformed headers."""
    app = FastAPI()
    app.add_middleware(sm.StatsDMiddleware)

    @app.patch("/indexes/{name}/documents")
    def patch_docs(name: str):
        # send malformed counts (non‑int strings)
        return JSONResponse(
            {"ok": True},
            headers={
                "x-count-success": "NaN",
                "x-count-failure": "oops",
                "x-count-error": "",
            },
        )

    return app


def test_patch_docs_metrics_and_malformed_headers(stub_statsd):
    """Test that PATCH requests emit metrics and handle malformed headers gracefully."""
    with TestClient(_app_with_patch_and_bad_headers()) as client:
        stub_statsd.sent.clear()
        client.patch("/indexes/foo/documents")

    # index_processing_time should be present
    assert any(k == "timing" and msg.startswith("index_processing_time") for k, msg in stub_statsd.sent)
    # malformed headers should NOT raise, and no increment for NaN values should be present
    assert not any(msg.startswith("x-count-success") for k, msg in stub_statsd.sent)


def test_statsd_prefix(monkeypatch):
    """Test that the StatsD client uses the configured prefix for all metrics."""
    # Ensure prefix is honoured for new instances
    client = sc.StatsDClient(prefix="marqo.")
    captured: List[bytes] = []
    monkeypatch.setattr(client, "_send", lambda msg: captured.append(msg.encode()), raising=True)
    client.increment("foo", 1)
    assert captured and captured[0].startswith(b"marqo.foo:1|c")


def test_common_tags_parsed_and_applied(monkeypatch):
    """Test that common tags from the environment variable are parsed and applied to all metrics."""
    # 1) Set env-var before module reload so singleton sees it
    monkeypatch.setenv("STATSD_COMMON_TAGS", "env:staging,team:search")
    importlib.reload(sc)

    client = sc.StatsDClient(prefix="")

    captured = []
    monkeypatch.setattr(client, "_send", lambda msg: captured.append(msg), raising=True)

    client.increment("foo", 1, tags={"team": "ops"})

    assert captured == ["foo:1|c|#env:staging,team:ops"]


def test_no_common_tags_when_env_missing(monkeypatch):
    """Test that if STATSD_COMMON_TAGS is not set, no common tags are applied."""
    monkeypatch.delenv("STATSD_COMMON_TAGS", raising=False)
    importlib.reload(sc)

    client = sc.StatsDClient(prefix="")
    captured = []
    monkeypatch.setattr(client, "_send", lambda msg: captured.append(msg), raising=True)

    client.timing("bar", 42)

    assert captured == ["bar:42|ms"]