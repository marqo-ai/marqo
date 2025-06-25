import unittest
from unittest.mock import Mock, patch, AsyncMock
from marqo.vespa.vespa_client import VespaClient


class TestVespaClient(unittest.TestCase):
    """Test VespaClient functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.vespa_client = VespaClient(
            config_url="http://localhost:19071",
            document_url="http://localhost:8080",
            query_url="http://localhost:8080",
            content_cluster_name="test_cluster"
        )

    def test_vespa_client_close_calls_async_transport_aclose(self):
        """Test that VespaClient.close calls async_transport.aclose()"""
        
        # Mock the async_transport with aclose method
        mock_async_transport = Mock()
        mock_async_transport.aclose = Mock()
        
        # Set the mock as the async_transport attribute
        self.vespa_client.async_transport = mock_async_transport
        
        # Call the close method
        self.vespa_client.close()
        
        # Verify that aclose was called
        mock_async_transport.aclose.assert_called_once()



    def test_get_content_url_single_path(self):
        """Test get_content_url with single path component"""
        
        base_url = "http://example.com/base"
        result = self.vespa_client.get_content_url(base_url, "path1")
        self.assertEqual(result, "http://example.com/basepath1")

    def test_get_content_url_multiple_paths(self):
        """Test get_content_url with multiple path components"""
        
        base_url = "http://example.com/base"
        result = self.vespa_client.get_content_url(base_url, "path1", "path2", "path3")
        # The actual implementation adds separators between path components
        self.assertEqual(result, "http://example.com/basepath1/path2/path3")

    def test_get_content_url_no_paths(self):
        """Test get_content_url with no path components"""
        
        base_url = "http://example.com/base"
        result = self.vespa_client.get_content_url(base_url)
        self.assertEqual(result, "http://example.com/base")

    def test_get_content_url_empty_path(self):
        """Test get_content_url with empty path component"""
        
        base_url = "http://example.com/base"
        result = self.vespa_client.get_content_url(base_url, "")
        self.assertEqual(result, "http://example.com/base")

    def test_vespa_client_initialization(self):
        """Test VespaClient initialization with required parameters"""
        
        client = VespaClient(
            config_url="http://localhost:19071",
            document_url="http://localhost:8080",
            query_url="http://localhost:8080",
            content_cluster_name="test_cluster"
        )
        
        self.assertEqual(client.config_url, "http://localhost:19071")
        self.assertEqual(client.document_url, "http://localhost:8080")
        self.assertEqual(client.query_url, "http://localhost:8080")
        self.assertEqual(client.content_cluster_name, "test_cluster")

    def test_vespa_client_initialization_with_optional_parameters(self):
        """Test VespaClient initialization with optional parameters"""
        
        client = VespaClient(
            config_url="http://localhost:19071",
            document_url="http://localhost:8080",
            query_url="http://localhost:8080",
            content_cluster_name="test_cluster",
            default_search_timeout_ms=5000
        )
        
        self.assertEqual(client.default_search_timeout_ms, 5000)
        # Note: pool_size and async_pool_size are not stored as instance attributes

    def test_vespa_client_initialization_missing_required_params_fails(self):
        """Test VespaClient initialization fails with missing required parameters"""
        
        with self.assertRaises(TypeError):
            VespaClient()

    def test_vespa_client_initialization_with_none_config_url_fails(self):
        """Test VespaClient initialization with None config_url"""
        
        # This should fail since the implementation calls strip() on config_url
        with self.assertRaises(AttributeError):
            VespaClient(
                config_url=None,
                document_url="http://localhost:8080",
                query_url="http://localhost:8080",
                content_cluster_name="test_cluster"
            )

    def test_get_batch_with_empty_ids_returns_empty_response(self):
        """Test get_batch with empty ids list returns empty response"""
        result = self.vespa_client.get_batch(
            ids=[],  # Empty list
            schema="test_schema"
        )
        
        # Should return empty response without making any requests
        self.assertEqual(len(result.responses), 0)
        self.assertFalse(result.errors)

    def test_delete_batch_with_empty_ids_returns_empty_response(self):
        """Test delete_batch with empty ids list returns empty response"""
        result = self.vespa_client.delete_batch(
            ids=[],  # Empty list
            schema="test_schema"
        )
        
        # Should return empty response without making any requests
        self.assertEqual(len(result.responses), 0)
        self.assertFalse(result.errors)


if __name__ == '__main__':
    unittest.main() 