"""
This file contains the `VespaClient` class, which integrates multiple mixins to provide a complete interface
for interacting with a Vespa search engine.

The `VespaClient` class is composed of multiple mixins, each handling a distinct category of operations, such as:
- Document operations (indexing, retrieval, deletion)
- Application deployment and management
- Querying (search) operations
- Error handling
- Utility functions

By using mixins, the `VespaClient` remains modular, readable, and easier to maintain.

Usage Example:
--------------
from vespa_client import VespaClient

vespa = VespaClient(config_url="http://localhost:8080", document_url="http://localhost:8080", query_url="http://localhost:8080")

# Feed a document
vespa.feed_document(document=my_doc, schema="products")

# Query Vespa
response = vespa.query()

# Close the client
vespa.close()
"""

from ._search import VespaSearchMixin
from ._document import VespaDocumentMixin
from ._deploy import VespaDeployMixin
from ._errors import VespaErrorHandlingMixin
from ._client_base import VespaClientBase

class VespaClient(
    VespaClientBase,
    VespaSearchMixin,
    VespaDocumentMixin,
    VespaDeployMixin,
    VespaErrorHandlingMixin,
):
    def __init__(
            self, config_url: str, document_url: str, query_url: str, content_cluster_name: str,
            default_search_timeout_ms: int = 1000, pool_size: int = 10, feed_pool_size: int = 10,
            get_pool_size: int = 10, delete_pool_size: int = 10, partial_update_pool_size: int = 10):
        """
        Create a VespaClient object.
        Args:
            config_url: Vespa Deploy API base URL
            document_url: Vespa Document API base URL
            query_url: Vespa Query API base URL
            pool_size: Number of connections to keep in the connection pool
            feed_pool_size: Number of connections to keep in batch feed requests connection pool to Vespa
            get_pool_size: Number of connections to keep in batch get requests connection pool to Vespa
            delete_pool_size: Number of connections to keep batch delete requests connection pool to Vespa
            partial_update_pool_size: Number of connections to keep batch partial update requests connection pool to Vespa
        """
        super().__init__(config_url, document_url, query_url, pool_size)
        self.default_search_timeout_ms = default_search_timeout_ms
        self.content_cluster_name = content_cluster_name
        self.feed_pool_size = feed_pool_size
        self.get_pool_size = get_pool_size
        self.delete_pool_size = delete_pool_size
        self.partial_pool_size = partial_update_pool_size