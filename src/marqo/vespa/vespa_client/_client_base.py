"""
This file defines the `VespaClientBase` class, which serves as the foundation for `VespaClient`.

The base class contains **shared attributes** and an HTTP client (`httpx.Client`) that is used across all mixins.
Any mixin that requires HTTP requests or Vespa configuration settings will inherit from this base class.

Attributes:
-----------
- config_url (str): Base URL for Vespa's configuration API.
- document_url (str): Base URL for Vespa's document API.
- query_url (str): Base URL for Vespa's search API.
- http_client (httpx.Client): Persistent HTTP client used for requests.

By centralizing these attributes in `VespaClientBase`, all mixins can access them without redundancy.

Usage:
-------
Mixins should **inherit** from `VespaClientBase` to gain access to shared attributes.
"""

import httpx

class VespaClientBase:
    """Base class containing common attributes shared across mixins."""
    def __init__(self, config_url: str, document_url: str, query_url: str, pool_size: int = 10):
        self.config_url = config_url.strip('/')
        self.document_url = document_url.strip('/')
        self.query_url = query_url.strip('/')
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        )