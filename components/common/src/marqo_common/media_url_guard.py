"""Destination checks for outbound fetches of user supplied media URLs.

Marqo downloads media from URLs that callers place in documents and in search queries.
A hostname says nothing about where the connection ends up, so a URL is only safe to
fetch once the addresses behind it have been classified. Without that check the
indexing and search paths can be aimed at addresses that are reachable only from inside
the deployment - sibling containers, private subnets, or the cloud instance metadata
endpoint - which turns them into a server side request forgery primitive.

The checks live here rather than in either component because both the API and the
inference orchestrator fetch user supplied URLs and must agree on what is reachable.
Only the standard library is used so that `marqo-common` keeps no dependencies.
"""

import ipaddress
import socket
from typing import Iterable, Optional, Union
from urllib.parse import urlparse

IpAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IpNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

# Name of the environment variable both components read. Kept here so the two settings
# layers and every error message refer to the same variable.
ALLOWED_NETWORKS_ENV_VAR = "MARQO_MEDIA_DOWNLOAD_ALLOWED_NETWORKS"

# Redirects are followed one hop at a time so every hop can be checked. The cap bounds
# the work done for a single fetch and stops a redirect loop.
MAX_MEDIA_REDIRECTS = 10


class UnsafeMediaUrlError(ValueError):
    """Raised when a media URL points at a destination Marqo must not connect to."""


def parse_allowed_networks(value: Optional[str]) -> tuple[IpNetwork, ...]:
    """Parse the comma separated CIDR blocks that are exempt from the destination check.

    Deployments that legitimately serve media from a private network - an object store
    running beside Marqo, for example - name those networks here rather than turning the
    check off.

    Args:
        value: Comma separated CIDR blocks, for example `10.0.0.0/8,192.168.0.0/16`.
            An empty or missing value means no network is exempt.

    Returns:
        The parsed networks.

    Raises:
        ValueError: If an entry is not a valid CIDR block.
    """
    if not value:
        return ()

    networks: list[IpNetwork] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as e:
            raise ValueError(
                f"`{entry}` in {ALLOWED_NETWORKS_ENV_VAR} is not a valid IP network. "
                f"Expected a comma separated list of CIDR blocks, for example "
                f"`10.0.0.0/8,192.168.0.0/16`. Original error: {e}"
            ) from e
    return tuple(networks)


def _unmap(address: IpAddress) -> IpAddress:
    """Reduce an IPv4 mapped IPv6 address to the IPv4 address it carries.

    `::ffff:127.0.0.1` reaches the same host as `127.0.0.1`, so both have to be
    classified the same way and match the same allow list entry.
    """
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def is_ip_allowed(ip: str, allowed_networks: Iterable[IpNetwork] = ()) -> bool:
    """Return whether Marqo may open a connection to `ip`.

    An address is allowed when it is globally routable and belongs to none of the
    special purpose ranges. Both halves of that test are needed: `is_global` on its own
    admits multicast and NAT64 addresses, while `is_private` on its own admits shared
    address space such as `100.64.0.0/10`.

    An unparseable address is treated as not allowed so that a caller can use this as
    the only gate on a connection.
    """
    try:
        address = _unmap(ipaddress.ip_address(ip))
    except ValueError:
        return False

    for network in allowed_networks:
        if network.version == address.version and address in network:
            return True

    return (
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_reserved
        and not address.is_multicast
        and not address.is_unspecified
    )


def refusal_message(url: str, host: Optional[str] = None) -> str:
    """Build the message used when a media fetch is refused.

    The address the URL resolved to is deliberately left out. Echoing it back would let
    a caller read internal DNS records through the error, which is the same class of
    disclosure the destination check exists to prevent.
    """
    subject = f"the host `{host}` is" if host else "it resolved to an address that is"
    return (
        f"Marqo will not download media from `{url}` because {subject} not publicly "
        f"routable. If this destination is intended, add its network to the "
        f"{ALLOWED_NETWORKS_ENV_VAR} environment variable."
    )


def assert_scheme_allowed(url: str) -> None:
    """Check that `url` uses a scheme Marqo is willing to fetch.

    Raises:
        UnsafeMediaUrlError: If the scheme is anything other than http or https.
    """
    scheme = urlparse(url).scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeMediaUrlError(
            f"Marqo only downloads media over http and https, but the url `{url}` uses "
            f"the scheme `{scheme or '<none>'}`."
        )


def assert_destination_allowed(
    url: str, allowed_networks: Iterable[IpNetwork] = ()
) -> tuple[str, ...]:
    """Check that `url` may be fetched, resolving its host to decide.

    Every address the host resolves to has to be allowed. A host that answers with a
    routable address alongside an internal one is refused, because which of the two the
    client ends up using is not under Marqo's control.

    Args:
        url: The URL about to be fetched.
        allowed_networks: Networks exempt from the check, from `parse_allowed_networks`.

    Returns:
        The addresses the host resolved to.

    Raises:
        UnsafeMediaUrlError: If the scheme, the host, or any resolved address is not
            allowed, or if the host cannot be resolved.
    """
    assert_scheme_allowed(url)

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise UnsafeMediaUrlError(f"The media url `{url}` does not contain a host.")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as e:
        raise UnsafeMediaUrlError(
            f"The media url `{url}` does not have a valid port. Original error: {e}"
        ) from e

    try:
        address_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeMediaUrlError(
            f"Marqo could not resolve the host `{host}` of the media url `{url}`. "
            f"Original error: {e}"
        ) from e

    addresses = tuple(dict.fromkeys(info[4][0] for info in address_info))
    if not addresses:
        raise UnsafeMediaUrlError(
            f"The host `{host}` of the media url `{url}` did not resolve to an address."
        )

    if not all(is_ip_allowed(address, allowed_networks) for address in addresses):
        raise UnsafeMediaUrlError(refusal_message(url, host))

    return addresses
