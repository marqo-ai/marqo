import ipaddress
import socket
import unittest
from unittest.mock import patch

from marqo_common.media_url_guard import (
    UnsafeMediaUrlError,
    assert_destination_allowed,
    assert_scheme_allowed,
    is_ip_allowed,
    parse_allowed_networks,
)


class TestIsIpAllowed(unittest.TestCase):
    """Classification of the addresses Marqo may open a connection to."""

    def test_refuses_addresses_that_are_not_publicly_routable(self):
        """Every address that is reachable only from inside a network must be refused."""
        test_cases = [
            ("IPv4 loopback", "127.0.0.1"),
            ("IPv4 loopback, other host in 127/8", "127.1.2.3"),
            ("unspecified", "0.0.0.0"),
            ("RFC1918 10/8", "10.0.0.5"),
            ("RFC1918 172.16/12", "172.16.0.1"),
            ("RFC1918 192.168/16", "192.168.1.1"),
            ("cloud instance metadata", "169.254.169.254"),
            ("shared address space, missed by is_private", "100.64.0.1"),
            ("benchmarking range", "198.18.0.1"),
            ("multicast, missed by is_global", "224.0.0.1"),
            ("broadcast", "255.255.255.255"),
            ("IPv6 loopback", "::1"),
            ("IPv6 unspecified", "::"),
            ("IPv4 mapped loopback", "::ffff:127.0.0.1"),
            ("IPv4 mapped metadata endpoint", "::ffff:169.254.169.254"),
            ("IPv6 link local", "fe80::1"),
            ("IPv6 unique local", "fd00::1"),
            ("NAT64 of loopback, missed by is_global", "64:ff9b::7f00:1"),
            ("6to4 of loopback", "2002:7f00:1::"),
            ("not an address", "not-an-ip"),
            ("empty", ""),
        ]

        for message, address in test_cases:
            with self.subTest(msg=message, address=address):
                self.assertFalse(is_ip_allowed(address))

    def test_allows_publicly_routable_addresses(self):
        """Ordinary public addresses must keep working."""
        test_cases = [
            ("public IPv4", "8.8.8.8"),
            ("public IPv4, other", "1.1.1.1"),
            ("public IPv4 in a CDN range", "104.20.23.154"),
            ("public IPv6", "2001:4860:4860::8888"),
            ("public IPv6, other", "2606:4700:10::6814:179a"),
        ]

        for message, address in test_cases:
            with self.subTest(msg=message, address=address):
                self.assertTrue(is_ip_allowed(address))

    def test_allowed_networks_exempt_an_address(self):
        """A named network makes an otherwise refused address reachable."""
        networks = parse_allowed_networks("10.0.0.0/8")

        self.assertTrue(is_ip_allowed("10.1.2.3", networks))
        self.assertFalse(is_ip_allowed("192.168.1.1", networks))

    def test_allowed_networks_match_the_ipv4_mapped_form(self):
        """An allow list entry covers the IPv4 mapped spelling of the same address."""
        networks = parse_allowed_networks("10.0.0.0/8")

        self.assertTrue(is_ip_allowed("::ffff:10.1.2.3", networks))

    def test_allowed_networks_of_a_different_family_do_not_raise(self):
        """Comparing an address against a network of the other family must not error."""
        networks = parse_allowed_networks("fd00::/8")

        self.assertFalse(is_ip_allowed("10.1.2.3", networks))


class TestParseAllowedNetworks(unittest.TestCase):
    """Parsing of the comma separated CIDR setting."""

    def test_parses_a_list_of_networks(self):
        """Whitespace and empty entries are tolerated so the setting is easy to write."""
        self.assertEqual(
            (ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("fd00::/8")),
            parse_allowed_networks(" 10.0.0.0/8 , fd00::/8 ,"),
        )

    def test_empty_value_allows_nothing_extra(self):
        """An unset value must not widen what Marqo may reach."""
        test_cases = [("none", None), ("empty string", ""), ("only separators", " , ")]

        for message, value in test_cases:
            with self.subTest(msg=message, value=value):
                self.assertEqual((), parse_allowed_networks(value))

    def test_invalid_entry_is_rejected(self):
        """A misspelled network must fail loudly rather than be dropped."""
        with self.assertRaises(ValueError) as context:
            parse_allowed_networks("10.0.0.0/8,not-a-network")

        self.assertIn("not-a-network", str(context.exception))
        self.assertIn("MARQO_MEDIA_DOWNLOAD_ALLOWED_NETWORKS", str(context.exception))


class TestAssertSchemeAllowed(unittest.TestCase):
    """Only http and https are fetched."""

    def test_refuses_schemes_other_than_http_and_https(self):
        """Schemes that libcurl also speaks must be refused before it sees them."""
        test_cases = [
            ("local file", "file:///etc/passwd"),
            ("gopher", "gopher://127.0.0.1:8882/_x"),
            ("dict", "dict://127.0.0.1:11211/"),
            ("ftp", "ftp://example.com/x.png"),
            ("no scheme", "example.com/x.png"),
        ]

        for message, url in test_cases:
            with self.subTest(msg=message, url=url):
                with self.assertRaises(UnsafeMediaUrlError):
                    assert_scheme_allowed(url)

    def test_accepts_http_and_https(self):
        """The schemes media is actually served over must pass."""
        for url in ("http://example.com/x.png", "HTTPS://example.com/x.png"):
            with self.subTest(url=url):
                self.assertIsNone(assert_scheme_allowed(url))


class TestAssertDestinationAllowed(unittest.TestCase):
    """Resolution of a URL's host and classification of every address behind it."""

    @staticmethod
    def _resolves_to(*addresses):
        def _getaddrinfo(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in addresses
            ]

        return _getaddrinfo

    def test_hostname_resolving_to_an_internal_address_is_refused(self):
        """A public looking hostname that resolves inward must be refused."""
        with patch("socket.getaddrinfo", side_effect=self._resolves_to("127.0.0.1")):
            with self.assertRaises(UnsafeMediaUrlError) as context:
                assert_destination_allowed("http://images.example.com/x")

        self.assertIn("not publicly routable", str(context.exception))

    def test_refusal_does_not_disclose_the_resolved_address(self):
        """The message must not turn the refusal into a way of reading internal DNS."""
        with patch("socket.getaddrinfo", side_effect=self._resolves_to("10.1.2.3")):
            with self.assertRaises(UnsafeMediaUrlError) as context:
                assert_destination_allowed("http://db.internal.example.com/x")

        self.assertNotIn("10.1.2.3", str(context.exception))

    def test_any_internal_address_in_the_answer_refuses_the_host(self):
        """A host answering with a public and an internal address must be refused."""
        with patch(
            "socket.getaddrinfo", side_effect=self._resolves_to("93.184.216.34", "10.1.2.3")
        ):
            with self.assertRaises(UnsafeMediaUrlError):
                assert_destination_allowed("http://images.example.com/x")

    def test_publicly_routable_host_is_allowed(self):
        """The resolved addresses are returned so a caller can reuse them."""
        with patch("socket.getaddrinfo", side_effect=self._resolves_to("93.184.216.34")):
            self.assertEqual(
                ("93.184.216.34",),
                assert_destination_allowed("http://images.example.com/x"),
            )

    def test_userinfo_does_not_disguise_the_host(self):
        """`http://example.com@127.0.0.1/` connects to 127.0.0.1 and must be treated as such."""
        with self.assertRaises(UnsafeMediaUrlError) as context:
            assert_destination_allowed("http://example.com@127.0.0.1/x")

        self.assertIn("not publicly routable", str(context.exception))

    def test_unresolvable_host_is_refused(self):
        """A host that cannot be resolved must not fall through to the fetch."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with self.assertRaises(UnsafeMediaUrlError) as context:
                assert_destination_allowed("http://nonexistent.example.com/x")

        self.assertIn("could not resolve", str(context.exception))

    def test_url_without_a_host_is_refused(self):
        """A URL with no host has nothing to classify."""
        with self.assertRaises(UnsafeMediaUrlError) as context:
            assert_destination_allowed("http:///x.png")

        self.assertIn("does not contain a host", str(context.exception))


if __name__ == "__main__":
    unittest.main()
