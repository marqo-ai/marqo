import re
import socket
import threading
import time
import unittest
from contextlib import contextmanager
from typing import List
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import marqo.core.monitoring.statsd_client as sc
import marqo.core.monitoring.statsd_middleware as sm


class _UDPSink:
    """A UDP sink that captures packets sent to it, thread-safe via _lock."""
    def __init__(self, host: str = "127.0.0.1", port: int = 0, family: int = socket.AF_INET):
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        # ensure recvfrom() wakes up regularly so stop()/join() can't hang
        self._sock.settimeout(0.2)
        self._sock.bind((host, port))
        self.host = self._sock.getsockname()[0]
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
                continue
            except OSError:
                break

    def stop(self):
        """Stop the UDP sink and close the socket."""
        self._running = False
        # poke the socket so recvfrom() unblocks on stubborn kernels
        try:
            self._sock.sendto(b"", (self.host, self.port))
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
def udp_sink(host: str = "127.0.0.1", family: int = socket.AF_INET):
    """Context manager to create a UDP sink for capturing metrics."""
    sink = _UDPSink(host=host, family=family)
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

            # Search & related
            @app.post("/indexes/{name}/search")
            def search(name: str):
                return {"hits": []}

            @app.post("/indexes/{name}/recommend")
            def recommend(name: str):
                return {"recommendations": []}

            @app.post("/indexes/{name}/embed")
            def embed(name: str):
                return {"vectors": []}

            @app.post("/indexes/bulk/search")
            def bulk_search():
                return {"results": []}

            # Documents collection endpoints
            @app.post("/indexes/{name}/documents")
            def add_docs(name: str):
                # Return explicit JSONResponse so we control headers precisely
                return JSONResponse(
                    content={"indexed": 1},
                    headers={"x-count-success": "1", "x-count-failure": "0", "x-count-error": "0"},
                    status_code=200,
                )

            @app.get("/indexes/{name}/documents")
            def get_docs(name: str):
                return {"docs": []}

            @app.patch("/indexes/{name}/documents")
            def patch_docs(name: str):
                return {"patched": True}

            @app.post("/indexes/{name}/documents/delete-batch")
            def delete_batch(name: str):
                return {"deleted": 2}

            @app.post("/indexes/{name}/documents/get-batch")
            def get_batch(name: str):
                return {"docs": [{"id": "a"}, {"id": "b"}]}

            # Single document (MUST be redacted)
            @app.get("/indexes/{name}/documents/{doc_id}")
            def get_doc(name: str, doc_id: str):
                return {"id": doc_id}

            # Stats/settings
            @app.get("/indexes/{name}/stats")
            def stats(name: str):
                return {"stats": {"docs": 10}}

            @app.get("/indexes/{name}/settings")
            def settings(name: str):
                return {"settings": {"x": 1}}

            # Models & devices
            @app.get("/models")
            def models_get():
                return {"models": []}

            @app.delete("/models")
            def models_delete():
                return {"deleted": True}

            @app.get("/device/cuda")
            def device_cuda():
                return {"cuda": True}

            @app.get("/device/cpu")
            def device_cpu():
                return {"cpu": True}

            return app

        cls.client_ctx = TestClient(_build_stub_app())
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        cls._sink_cm.__exit__(None, None, None)

    def _count(self, pkt: List[str], pattern: str) -> int:
        """Count how many packets match the given regex pattern."""
        return sum(1 for p in pkt if re.search(pattern, p))

    def _wait_for_all_patterns(self, sink, patterns: List[str], timeout: float = 8.0) -> List[str]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            pkt = sink.decoded()
            if all(any(re.search(p, x) for x in pkt) for p in patterns):
                return pkt
            time.sleep(0.05)
        return sink.decoded()

    def test_metrics_roundtrip_all_endpoints(self):
        self.client.post("/indexes/foo/search")
        self.client.post("/indexes/foo/recommend")
        self.client.post("/indexes/foo/embed")
        self.client.post("/indexes/bulk/search")

        self.client.post("/indexes/foo/documents")  # emits batch counters
        self.client.get("/indexes/foo/documents")  # collection GET
        self.client.patch("/indexes/foo/documents")  # partial update
        self.client.get("/indexes/foo/documents/abc123")  # MUST redact id
        self.client.post("/indexes/foo/documents/delete-batch")  # MUST NOT redact
        self.client.post("/indexes/foo/documents/delete-batch?telemetry=true")  # MUST NOT redact (even with query)
        self.client.post("/indexes/foo/documents/get-batch")  # MUST NOT redact
        self.client.post("/indexes/foo/documents/get-batch?telemetry=true")  # MUST NOT redact (even with query)

        self.client.get("/indexes/foo/stats")
        self.client.get("/indexes/foo/settings")

        self.client.get("/models")
        self.client.delete("/models")
        self.client.get("/device/cuda")
        self.client.get("/device/cpu")

        # Expected metric patterns
        patterns = [

            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/search,method:POST,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/recommend,method:POST,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/embed,method:POST,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/bulk/search,method:POST,status_code:200",

            # documents collection endpoints (no redaction)
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents,method:POST,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents,method:GET,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents,method:PATCH,status_code:200",

            # single doc (redacted)
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents/<document_id>,method:GET,status_code:200",

            # fixed subpaths (no redaction)
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents/delete-batch,method:POST,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents/get-batch,method:POST,status_code:200",

            # stats/settings
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/stats,method:GET,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/settings,method:GET,status_code:200",

            # models & devices
            r"request\.duration_ms:\d+\|ms\|#path:/models,method:GET,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/models,method:DELETE,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/device/cuda,method:GET,status_code:200",
            r"request\.duration_ms:\d+\|ms\|#path:/device/cpu,method:GET,status_code:200",

            # batch counters from POST /documents (headers-driven)
            r"batch\.success:1\|c\|#path:/indexes/foo/documents,method:POST,status_code:200",
            r"batch\.failure:0\|c\|#path:/indexes/foo/documents,method:POST,status_code:200",
            r"batch\.error:0\|c\|#path:/indexes/foo/documents,method:POST,status_code:200",
        ]

        # Wait for all the expected packets
        pkt = self._wait_for_all_patterns(self.sink, patterns, timeout=8.0)

        for pat in patterns:
            self.assertTrue(_has(pkt, pat), msg=f"Missing packet /{pat}/\nSeen:\n{pkt}")

        # 1) Query string must NEVER appear inside the #path tag.
        # Detect any '#path:...?...,'
        self.assertFalse(
            any(re.search(r"#path:[^,]*\?", p) for p in pkt),
            msg=f"Query string leaked into path tag\nSeen:\n{pkt}",
        )
        # 2) We sent delete-batch twice (with and without ?telemetry=true),
        # so the normalized metric line should appear exactly twice.
        delete_batch_pat = (
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents/delete-batch,method:POST,status_code:200"
        )
        self.assertEqual(
            self._count(pkt, delete_batch_pat), 2,
            msg=f"Expected exactly 2 delete-batch packets (query stripped)\nSeen:\n{pkt}",
        )

        # 3) Same for get-batch (also called twice).
        get_batch_pat = (
            r"request\.duration_ms:\d+\|ms\|#path:/indexes/foo/documents/get-batch,method:POST,status_code:200"
        )
        self.assertEqual(
            self._count(pkt, get_batch_pat), 2,
            msg=f"Expected exactly 2 get-batch packets (query stripped)\nSeen:\n{pkt}",
        )

    def test_idempotent_redaction(self):
        # sanity: redaction is stable on reprocessing
        self.client.get("/indexes/foo/documents/abc123")
        # at least one packet with the redacted path must exist; second pass shouldn't change it
        self.sink.wait(n=1)
        pkt = self.sink.decoded()
        self.assertTrue(
            _has(pkt, r"#path:/indexes/foo/documents/<document_id>,method:GET,status_code:200"),
            msg=f"Missing redacted packet\nSeen:\n{pkt}",
        )


class TestStatsDClientDualStack(unittest.TestCase):
    """Integration tests for StatsDClient dual-stack (IPv4/IPv6) socket support."""

    def test_statsd_client_ipv4_sends_metrics(self):
        """Test that StatsDClient can send metrics over IPv4."""
        with udp_sink(host="127.0.0.1", family=socket.AF_INET) as sink:
            client = sc.StatsDClient(host="127.0.0.1", port=sink.port)

            # Verify socket is IPv4
            self.assertEqual(client._sock.family, socket.AF_INET)

            # Send a metric and verify it arrives
            client.increment("test.ipv4.counter", 1)
            sink.wait(n=1)

            pkt = sink.decoded()
            self.assertTrue(
                _has(pkt, r"test\.ipv4\.counter:1\|c"),
                msg=f"IPv4 metric not received\nSeen:\n{pkt}",
            )

    def test_statsd_client_ipv6_sends_metrics(self):
        """Test that StatsDClient can send metrics over IPv6."""
        with udp_sink(host="::1", family=socket.AF_INET6) as sink:
            client = sc.StatsDClient(host="::1", port=sink.port)

            # Verify socket is IPv6
            self.assertEqual(client._sock.family, socket.AF_INET6)

            # Send a metric and verify it arrives
            client.increment("test.ipv6.counter", 1)
            sink.wait(n=1)

            pkt = sink.decoded()
            self.assertTrue(
                _has(pkt, r"test\.ipv6\.counter:1\|c"),
                msg=f"IPv6 metric not received\nSeen:\n{pkt}",
            )

    def test_statsd_client_ipv4_timing_metric(self):
        """Test that timing metrics work over IPv4."""
        with udp_sink(host="127.0.0.1", family=socket.AF_INET) as sink:
            client = sc.StatsDClient(host="127.0.0.1", port=sink.port)
            client.timing("test.ipv4.latency", 150)
            sink.wait(n=1)

            pkt = sink.decoded()
            self.assertTrue(
                _has(pkt, r"test\.ipv4\.latency:150\|ms"),
                msg=f"IPv4 timing metric not received\nSeen:\n{pkt}",
            )

    def test_statsd_client_ipv6_timing_metric(self):
        """Test that timing metrics work over IPv6."""
        with udp_sink(host="::1", family=socket.AF_INET6) as sink:
            client = sc.StatsDClient(host="::1", port=sink.port)
            client.timing("test.ipv6.latency", 250)
            sink.wait(n=1)

            pkt = sink.decoded()
            self.assertTrue(
                _has(pkt, r"test\.ipv6\.latency:250\|ms"),
                msg=f"IPv6 timing metric not received\nSeen:\n{pkt}",
            )

    def test_statsd_client_fallback_to_ipv4_when_resolution_fails(self):
        """If getaddrinfo raises, we fall back to IPv4 and do not raise on send."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("mock resolution failure")):
            client = sc.StatsDClient(host="does-not-resolve.invalid", port=8125)
            # Fallback path should create an IPv4 socket
            self.assertEqual(client._sock.family, socket.AF_INET)
            # Sending should not raise (errors are swallowed and logged at debug)
            client.increment("test.fallback.counter", 1)
