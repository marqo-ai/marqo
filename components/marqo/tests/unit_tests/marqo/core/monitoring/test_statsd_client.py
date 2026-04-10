import socket
from typing import List
from unittest.mock import patch

import pytest

from marqo.core.monitoring import statsd_client as sc


class CaptureStatsDClient(sc.StatsDClient):
    """A StatsDClient that captures sent messages instead of sending over UDP."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: List[bytes] = []

    def _send(self, msg: str) -> None:
        self.sent.append(msg.encode())


class TestStatsDClient:
    """Tests for the StatsDClient class."""
    def test_encode_tags_none(self):
        assert sc.StatsDClient._encode_tags(None) == ""

    def test_encode_tags_order_insensitive(self):
        tagged = sc.StatsDClient._encode_tags({"a": "1", "b": "2"})
        assert tagged in {"|#a:1,b:2", "|#b:2,a:1"}

    @pytest.mark.parametrize(
        "tags,expected",
        [
            (None, b"counter:1|c"),
            ({"k": "v"}, b"counter:1|c|#k:v"),
        ],
    )
    def test_increment_serialisation(self, tags, expected):
        """Test that increment serialisation works correctly with and without tags."""
        client = CaptureStatsDClient(host="127.0.0.1", port=9999)
        client.increment("counter", 1, tags=tags)
        assert client.sent == [expected]

    def test_timing_serialisation_with_prefix(self):
        """Test that timing serialisation works correctly with a prefix."""
        client = CaptureStatsDClient(host="127.0.0.1", port=9999, prefix="marqo.")
        client.timing("latency", 321)
        assert client.sent == [b"marqo.latency:321|ms"]

    def test_common_tags_merge(self, monkeypatch):
        """Test that common tags from ENV are merged with per-call tags."""
        monkeypatch.setenv("STATSD_COMMON_TAGS", "env:dev,team:search")

        client = CaptureStatsDClient()
        client.increment("foo", 2, tags={"team": "ops"})

        # order of tags not guaranteed
        assert client.sent in [
            [b"foo:2|c|#env:dev,team:ops"],
            [b"foo:2|c|#team:ops,env:dev"],
        ]

    def test_parse_common_tags_empty_and_malformed(self):
        """Test that _parse_common_tags handles empty and malformed strings."""
        assert sc.StatsDClient._parse_common_tags("") == {}
        assert sc.StatsDClient._parse_common_tags("foo:bar,,baz:") == {"foo": "bar"}


class TestCreateSocket:
    """Tests for the _create_socket method and dual-stack support."""

    def test_create_socket_ipv4_address(self):
        """Test that an IPv4 address creates an IPv4 socket."""
        sock, addr = sc.StatsDClient._create_socket("127.0.0.1", 8125)
        try:
            assert sock.family == socket.AF_INET
            assert addr == ("127.0.0.1", 8125)
        finally:
            sock.close()

    def test_create_socket_ipv6_address(self):
        """Test that an IPv6 address creates an IPv6 socket."""
        sock, addr = sc.StatsDClient._create_socket("::1", 8125)
        try:
            assert sock.family == socket.AF_INET6
            # IPv6 sockaddr is (host, port, flowinfo, scope_id)
            assert addr[0] == "::1"
            assert addr[1] == 8125
        finally:
            sock.close()

    def test_create_socket_fallback_to_ipv4_on_resolution_failure(self):
        """Test that socket falls back to IPv4 when address resolution fails."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("mock failure")):
            sock, addr = sc.StatsDClient._create_socket("unresolvable.invalid", 8125)
            try:
                assert sock.family == socket.AF_INET
                assert addr == ("unresolvable.invalid", 8125)
            finally:
                sock.close()

    def test_create_socket_is_non_blocking(self):
        """Test that created sockets are non-blocking."""
        sock, _ = sc.StatsDClient._create_socket("127.0.0.1", 8125)
        try:
            assert sock.getblocking() is False
        finally:
            sock.close()

    def test_create_socket_is_udp(self):
        """Test that created sockets are UDP (SOCK_DGRAM)."""
        sock, _ = sc.StatsDClient._create_socket("127.0.0.1", 8125)
        try:
            assert sock.type == socket.SOCK_DGRAM
        finally:
            sock.close()
