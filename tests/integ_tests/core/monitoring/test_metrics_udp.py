import re
import socket
import threading
import time
import unittest
from contextlib import contextmanager
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import marqo.core.monitoring.statsd_client as sc
import marqo.core.monitoring.statsd_middleware as sm


class _UDPSink:
    """A UDP sink that captures packets sent to it, thread-safe via _lock."""
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # ensure recvfrom() wakes up regularly so stop()/join() can’t hang
        self._sock.settimeout(0.2)
        self._sock.bind((host, port))
        self.port = self._sock.getsockname()[1]

        self._lock = threading.Lock()
        self._running = True
        self.packets: List[bytes] = []

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Continuously listen for UDP packets and store them."""
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                with self._lock:
                    self.packets.append(data)
            except socket.timeout:
                continue  # polling tick
            except OSError:
                break

    def stop(self):
        """Stop the UDP sink and close the socket."""
        self._running = False
        # poke the socket so recvfrom() unblocks on stubborn kernels
        try:
            self._sock.sendto(b"", ("127.0.0.1", self.port))
        except OSError:
            pass
        self._sock.close()
        self._thread.join()

    def wait(self, n: int, timeout: float = 5.0):
        """Block until >= n packets captured or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self.packets) >= n:
                    break
            time.sleep(0.03)

    def decoded(self) -> List[str]:
        """Return the captured packets as a list of decoded strings."""
        with self._lock:
            return [p.decode() for p in list(self.packets)]


@contextmanager
def udp_sink():
    """Context manager to create a UDP sink for capturing metrics."""
    sink = _UDPSink()
    try:
        yield sink
    finally:
        sink.stop()


def _has(pkt: List[str], pattern: str) -> bool:
    """Check if any packet matches the given regex pattern."""
    return any(re.search(pattern, p) for p in pkt)


# --------------------------------------------------------------------------- #
#                              Test case class                                #
# --------------------------------------------------------------------------- #
class TestStatsDMiddlewareUDP(unittest.TestCase):
    """End-to-end: StatsDMiddleware emits expected packets over UDP."""

    @classmethod
    def setUpClass(cls):
        """Set up a UDP sink and a FastAPI client for testing."""
        cls._sink_cm = udp_sink()
        cls.sink = cls._sink_cm.__enter__()

        def _build_stub_app():
            """Build a FastAPI app with StatsDMiddleware for testing."""
            app = FastAPI()
            statsd = sc.StatsDClient(host="127.0.0.1", port=cls.sink.port)
            app.add_middleware(sm.StatsDMiddleware, statsd_client=statsd)

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

        cls.client_ctx = TestClient(_build_stub_app())
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        cls._sink_cm.__exit__(None, None, None)

    def test_metrics_roundtrip(self):
        """Test that the middleware emits expected metrics over UDP."""
        self.client.get("/")
        self.client.get("/indexes/foo/search")  # search timing
        self.client.post("/indexes/foo/documents")  # index timing + headers
        self.client.get("/indexes/foo/documents/abc123")  # redaction

        patterns = [
            r"marqo_processing_time:\d+\|ms",
            r"requests\.completed:1\|c\|#status_code:\dXX",
            r"search_processing_time:\d+\|ms",
            r"index_processing_time:\d+\|ms",
            r"x-count-success:\d+\|c",
            r"x-count-failure:\d+\|c",
            r"x-count-error:\d+\|c",
            r"requests\.completed:1\|c\|#path:/indexes/foo/documents(?:/<document_id>)?,method:(?:GET|POST),status_code:\dXX",
        ]

        # Wait until the six packets we assert on have arrived
        self.sink.wait(n=len(patterns))
        pkt = self.sink.decoded()

        for pat in patterns:
            self.assertTrue(
                _has(pkt, pat),
                msg=f"Missing packet matching /{pat}/ in {pkt}",
            )
