import unittest
from unittest.mock import Mock, patch
import httpx
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

    def test_vespa_client_close_calls_http_client_close(self):
        """Test that VespaClient.close calls http_client.close()"""
        
        # Mock the http_client.close method
        with patch.object(self.vespa_client.http_client, 'close') as mock_close:
            # Call the close method
            self.vespa_client.close()
            
            # Verify that close was called
            mock_close.assert_called_once()

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
        # Note: pool_size and get_pool_size are stored as instance attributes

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

    def test_query_httpx_timeout_configuration_small_vespa_timeout(self):
        """Test that httpx read timeout is set to max(5.0, (vespa_timeout + 1000) / 1000) for Vespa timeouts"""
        def mock_post(*args, **kwargs):
            # Return a mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"root": {"id": "test", "relevance": 1.0, "children": []}}'
            return mock_response

        test_cases = [
            (1000, 5.0, "1000ms", "Vespa timeout 1000ms -> httpx timeout 5.0s"),
            (1, 5.0, "1ms", "Vespa timeout 1ms -> httpx timeout 5.0s"),
            (6000, 7.0, "6000ms", "Vespa timeout 6000ms -> httpx timeout 7.0s"),
            (None, 5.0, "1000ms", "Vespa timeout None -> Default to 1000 -> httpx timeout 5.0s"),
            (0, 5.0, "1000ms", "Vespa timeout 0ms -> Default to 1000 -> httpx timeout 5.0s"),
        ]

        for provided_vespa_timeout_ms, httpx_read_timeout_second, expected_vespa_timeout_ms, msg in test_cases:
            with self.subTest(msg=msg):
                with patch.object(httpx.Client, 'post', side_effect=mock_post) as mock_query:
                    self.vespa_client.query(
                        yql="select * from sources * where test;",
                        timeout=provided_vespa_timeout_ms
                    )

                    timeout_obj = mock_query.call_args.kwargs["timeout"]
                    vespa_time_out = mock_query.call_args.kwargs["json"]["timeout"]
                    self.assertEqual(expected_vespa_timeout_ms, vespa_time_out)
                    self.assertEqual(httpx_read_timeout_second, timeout_obj.read)
                    self.assertEqual(5.0, timeout_obj.connect)
                    self.assertEqual(5.0, timeout_obj.write)
                    self.assertEqual(5.0, timeout_obj.pool)



if __name__ == '__main__':
    unittest.main()