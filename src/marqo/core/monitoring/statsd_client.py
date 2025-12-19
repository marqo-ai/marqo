import logging
import socket
from typing import Dict, Optional, Tuple

import marqo.logging
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints, read_env_vars_and_defaults

logger = marqo.logging.get_logger(__name__)


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
        host = host or read_env_vars_and_defaults(EnvVars.STATSD_HOST)
        port = int(port or read_env_vars_and_defaults_ints(EnvVars.STATSD_PORT))

        self.prefix = prefix
        self._sock, self.addr = self._create_socket(host, port)

        # Parse once; reused for every metric
        self._common_tags = self._parse_common_tags(read_env_vars_and_defaults(EnvVars.STATSD_COMMON_TAGS))

    @staticmethod
    def _create_socket(host: str, port: int) -> Tuple[socket.socket, Tuple]:
        """
        Create a UDP socket that supports both IPv4 and IPv6 (dual-stack).

        Args:
            host: The hostname or IP address to connect to.
            port: The port number.

        Returns:
            A tuple of (socket, address) where address is suitable for sendto().
        """
        # getaddrinfo returns a list of 5-tuples: (family, type, proto, canonname, sockaddr)
        logger.debug(f"Resolving address for StatsD: host={host}, port={port}")
        try:
            addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        except socket.gaierror as e:
            logger.debug(f"Address resolution failed for {host}:{port}: {e}")
            addr_info = None

        if addr_info:
            # Use the first resolved address - socket family matches the host address type
            # (IPv4 addresses get AF_INET, IPv6 addresses get AF_INET6)
            family, socktype, proto, canonname, addr = addr_info[0]
            family_name = "IPv6" if family == socket.AF_INET6 else "IPv4"
            logger.debug(f"Address resolved: family={family_name}, addr={addr}")
        else:
            # Fall back to IPv4 if address resolution fails
            family, socktype, proto = socket.AF_INET, socket.SOCK_DGRAM, 0
            addr = (host, port)
            logger.debug(f"Falling back to IPv4 socket: addr={addr}")

        sock = socket.socket(family, socktype, proto)
        sock.setblocking(False)
        logger.debug(f"StatsD socket created: family={sock.family}, blocking=False")

        return sock, addr

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
        except Exception as exc:
            # Metrics must never break request handling
            logging.debug(f"Failed to send StatsD message: {exc}")
