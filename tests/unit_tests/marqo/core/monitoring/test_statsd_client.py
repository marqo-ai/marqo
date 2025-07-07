from typing import List

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
        assert sc.StatsDClient._parse_common_tags("") == {}
        assert sc.StatsDClient._parse_common_tags("foo:bar,,baz:") == {"foo": "bar"}
