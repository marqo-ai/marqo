import os
import socket
from typing import Dict, Optional


def _parse_common_tags(raw: Optional[str]) -> Dict[str, str]:
    """
    Convert 'k1:v1,k2:v2' into {'k1': 'v1', 'k2': 'v2'}.
    Ignores empty or malformed pairs so mis-configuration can’t break metrics.
    """
    if not raw:
        return {}

    tags: Dict[str, str] = {}
    for pair in raw.split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                tags[key] = value
    return tags


class StatsDClient:
    """
    Minimal DogStatsD-compatible UDP client.
    CloudWatch Agent listens on 127.0.0.1:8125 by default.

    Every metric automatically includes the tag set defined in the
    STATSD_COMMON_TAGS environment variable (e.g. "env:prod,team:search").
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        self.addr = (
            host or os.getenv("STATSD_HOST", "127.0.0.1"),
            int(port or os.getenv("STATSD_PORT", 8125)),
        )
        self.prefix = prefix
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

        # Parse once; reused for every metric
        self._common_tags = _parse_common_tags(os.getenv("STATSD_COMMON_TAGS"))

    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.
        Args:
            metric: the name of the metric.
            value: the value to increment by (default is 1).
            tags: Optional tags to include with the metric.
        """
        merged = {**self._common_tags, **(tags or {})}
        msg = f"{self.prefix}{metric}:{value}|c{self._encode_tags(merged)}"
        self._send(msg)

    def timing(
        self,
        metric: str,
        value_ms: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a timing metric in milliseconds.
        Args:
            metric: the name of the metric.
            value_ms: the value in milliseconds.
            tags: Optional tags to include with the metric.
        """
        merged = {**self._common_tags, **(tags or {})}
        msg = f"{self.prefix}{metric}:{value_ms}|ms{self._encode_tags(merged)}"
        self._send(msg)

    @staticmethod
    def _encode_tags(tags: Optional[Dict[str, str]]) -> str:
        """
        Encode tags into a string suitable for DogStatsD.
        Args:
            tags: A dictionary of tags to encode.

        Returns:
            str: Encoded tags in the format "|#k1:v1,k2:v2".
        """
        if not tags:
            return ""
        return "|#" + ",".join(f"{k}:{v}" for k, v in tags.items())

    def _send(self, msg: str) -> None:
        """
        Send a message to the StatsD server.
        Args:
            msg: The message to send, formatted as per DogStatsD protocol.
        """
        try:
            # UDP – fire and forget
            self._sock.sendto(msg.encode("utf-8"), self.addr)
        except Exception:
            # Metrics must never break request handling
            pass


# Module-level singleton – keeps one UDP socket open per process         #
_default_client = StatsDClient(prefix=os.getenv("STATSD_PREFIX", ""))


def get_client() -> StatsDClient:
    """Return the process-wide StatsD client instance."""
    return _default_client
