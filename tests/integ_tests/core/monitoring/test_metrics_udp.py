import importlib
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import marqo.core.monitoring.statsd_client as sc


class _UDPSink:
    """A UDP sink that captures packets sent to it."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self.port = self._sock.getsockname()[1]
        self._running = True
        self.packets: List[bytes] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Continuously listen for UDP packets and store them."""
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                self.packets.append(data)
            except OSError:
                break

    def stop(self):
        """Stop the UDP sink and close the socket."""
        self._running = False
        self._sock.close()
        self._thread.join()

    def wait(self, n: int, timeout: float = 2.0):
        """Wait until at least `n` packets are captured or timeout occurs."""
        deadline = time.time() + timeout
        while len(self.packets) < n and time.time() < deadline:
            time.sleep(0.03)

    def decoded(self) -> List[str]:
        """Return the captured packets as a list of decoded strings."""
        return [p.decode() for p in self.packets]


@contextmanager
def udp_sink():
    """Context manager to create a UDP sink for capturing metrics."""
    sink = _UDPSink()
    try:
        yield sink
    finally:
        sink.stop()


@pytest.fixture(scope="module")
def client_and_sink():
    """Fixture that sets up a FastAPI app with StatsD middleware and a UDP sink."""
    with udp_sink() as sink:
        # Point StatsD at our sink before reloading modules
        os.environ["STATSD_HOST"] = "127.0.0.1"
        os.environ["STATSD_PORT"] = str(sink.port)

        # Re‑exec statsd_client so the singleton picks the new env‑vars
        importlib.reload(sc)

        # Also reload the middleware module so its imported get_client() refers
        # to the newly reloaded statsd_client module.
        import marqo.core.monitoring.statsd_middleware as sm
        importlib.reload(sm)

        # Build a minimal app _after_ the reloads so everything lines up.
        def _build_stub_app():
            app = FastAPI()
            app.add_middleware(sm.StatsDMiddleware)

            @app.get("/")
            def root():
                return {"ok": True}

            @app.get("/indexes/{name}/search")
            def search(name: str):
                return {"hits": []}

            @app.post("/indexes/{name}/documents")
            def add_docs(name: str):
                # Return explicit JSONResponse so we control headers precisely
                return JSONResponse(
                    content={"indexed": 1},
                    headers={
                        "x-count-success": "1",
                        "x-count-failure": "0",
                        "x-count-error": "0",
                    },
                    status_code=200,
                )

            @app.get("/indexes/{name}/documents/{doc_id}")
            def get_doc(name: str, doc_id: str):
                return {"id": doc_id}

            return app

        with TestClient(_build_stub_app()) as client:
            yield client, sink

        os.environ.pop("STATSD_HOST", None)
        os.environ.pop("STATSD_PORT", None)


def _has(pkt: List[str], pattern: str) -> bool:
    """ Check if any packet in the list matches the given regex pattern. """
    return any(re.search(pattern, p) for p in pkt)


def test_metrics_roundtrip(client_and_sink):
    """Test that the StatsD middleware emits the expected metrics."""
    client, sink = client_and_sink

    # exercise all code‑paths the middleware cares about
    client.get("/")
    client.get("/indexes/foo/search")  # search timing
    client.post("/indexes/foo/documents")  # index timing + headers
    client.get("/indexes/foo/documents/abc123")  # redaction

    sink.wait(n=8)

    pkt = sink.decoded()

    assert _has(pkt, r"marqo_processing_time:\d+\|ms")
    assert _has(pkt, r"requests\.completed:1\|c\|#status_code:\dXX")
    assert _has(pkt, r"search_processing_time:\d+\|ms")
    assert _has(pkt, r"index_processing_time:\d+\|ms")
    assert _has(pkt, r"x-count-success:\d+\|c")
    assert _has(
        pkt,
        r"requests\.completed:1\|c\|#path:/indexes/foo/documents/<document_id>,method:GET,status_code:\dXX",
    )
