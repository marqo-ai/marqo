import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
import httpx
import orjson
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


    def test_get_document_deserializes_response(self):
        """Test that get_document correctly deserializes the Vespa response"""
        response_data = {
            'pathId': '/document/v1/test_schema/test_schema/docid/doc1',
            'id': 'id:test_schema:test_schema::doc1',
            'fields': {'title': 'Test Title', 'body': 'Test Body'}
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(response_data)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            result = self.vespa_client.get_document(id='doc1', schema='test_schema')

        self.assertEqual(result.path_id, response_data['pathId'])
        self.assertEqual(result.document.id, response_data['id'])
        self.assertEqual(result.document.fields, response_data['fields'])

    def test_get_all_documents_deserializes_response(self):
        """Test that get_all_documents correctly deserializes the Vespa response"""
        response_data = {
            'pathId': '/document/v1/test_schema/test_schema/docid',
            'documents': [
                {'id': 'id:test_schema:test_schema::doc1', 'fields': {'title': 'Title 1'}},
                {'id': 'id:test_schema:test_schema::doc2', 'fields': {'title': 'Title 2'}},
            ],
            'documentCount': 2
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(response_data)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            result = self.vespa_client.get_all_documents(schema='test_schema')

        self.assertEqual(result.path_id, response_data['pathId'])
        self.assertEqual(result.document_count, 2)
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.documents[0].id, 'id:test_schema:test_schema::doc1')

    @patch('marqo.vespa.vespa_client.httpx.AsyncClient')
    def test_get_batch_deserializes_response(self, mock_async_client_class):
        """Test that get_batch correctly deserializes the Vespa response"""
        response_data = {
            'pathId': '/document/v1/test_schema/test_schema/docid/doc1',
            'id': 'id:test_schema:test_schema::doc1',
            'fields': {'title': 'Test Title'}
        }
        mock_async_client = mock_async_client_class.return_value.__aenter__.return_value
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(response_data)
        mock_async_client.get.return_value = mock_response

        result = self.vespa_client.get_batch(ids=['doc1'], schema='test_schema')

        self.assertEqual(len(result.responses), 1)
        self.assertFalse(result.errors)
        self.assertEqual(result.responses[0].status, 200)
        self.assertEqual(result.responses[0].document.fields, {'title': 'Test Title'})

    def test_get_document_async_with_specific_fields_deserializes_response(self):
        """Test that _get_document_async_with_specific_fields correctly deserializes the response"""
        response_data = {
            'pathId': '/document/v1/test_schema/test_schema/docid/doc1',
            'id': 'id:test_schema:test_schema::doc1',
            'fields': {'title': 'Test Title'}
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(response_data)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        async def _run():
            semaphore = asyncio.Semaphore(1)
            return await self.vespa_client._get_document_async_with_specific_fields(
                semaphore, mock_client, 'doc1', ['title'], 'test_schema', 60
            )

        result = asyncio.run(_run())

        self.assertEqual(result.status, 200)
        self.assertEqual(result.document.fields, {'title': 'Test Title'})
        # Verify the fieldSet parameter was included in the URL
        call_url = mock_client.get.call_args[0][0]
        self.assertIn('fieldSet=test_schema:title', call_url)


    def test_get_convergence_status_all_converged(self):
        """Test _get_convergence_status when all services are converged."""
        response_data = {
            'currentGeneration': 9,
            'wantedGeneration': 9,
            'converged': True,
            'services': [
                {'host': 'node1', 'port': 8080, 'type': 'container', 'currentGeneration': 9},
                {'host': 'node1', 'port': 19108, 'type': 'searchnode', 'currentGeneration': 9},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            status = self.vespa_client._get_convergence_status()

        self.assertTrue(status.converged)
        self.assertEqual(status.current_generation, 9)
        self.assertEqual(status.wanted_generation, 9)
        self.assertEqual(status.non_converged_services, [])

    def test_get_convergence_status_some_not_converged(self):
        """Test _get_convergence_status lists services not at wantedGeneration."""
        response_data = {
            'currentGeneration': 8,
            'wantedGeneration': 9,
            'converged': False,
            'services': [
                {'host': 'node1', 'port': 8080, 'type': 'container', 'currentGeneration': 9},
                {'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8},
                {'host': 'node2', 'port': 19108, 'type': 'searchnode', 'currentGeneration': 7},
                {'host': 'node1', 'port': 19108, 'type': 'searchnode', 'currentGeneration': 9},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            status = self.vespa_client._get_convergence_status()

        self.assertFalse(status.converged)
        self.assertEqual(len(status.non_converged_services), 2)
        self.assertEqual(status.non_converged_services[0], {
            'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8
        })
        self.assertEqual(status.non_converged_services[1], {
            'host': 'node2', 'port': 19108, 'type': 'searchnode', 'currentGeneration': 7
        })

    def test_convergence_status_to_dict_includes_non_converged_services(self):
        """Test that to_dict includes nonConvergedServices only when non-empty."""
        status_converged = VespaClient._ConvergenceStatus(
            current_generation=9, wanted_generation=9, converged=True,
            non_converged_services=[]
        )
        d = status_converged.to_dict()
        self.assertNotIn('nonConvergedServices', d)

        non_converged = [{'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8}]
        status_not_converged = VespaClient._ConvergenceStatus(
            current_generation=8, wanted_generation=9, converged=False,
            non_converged_services=non_converged
        )
        d = status_not_converged.to_dict()
        self.assertIn('nonConvergedServices', d)
        self.assertEqual(d['nonConvergedServices'], non_converged)

    def test_wait_for_convergence_error_message_contains_non_converged_services(self):
        """Test that the timeout error message includes non-converged service details."""
        from marqo.vespa.exceptions import VespaNotConvergedError

        convergence_response = {
            'currentGeneration': 8,
            'wantedGeneration': 9,
            'converged': False,
            'services': [
                {'host': 'node1', 'port': 8080, 'type': 'container', 'currentGeneration': 9},
                {'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = convergence_response

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError) as ctx:
                self.vespa_client.wait_for_application_convergence(timeout=1)

        error_msg = str(ctx.exception)
        self.assertIn('node2', error_msg)
        self.assertIn('wantedGeneration', error_msg)
        self.assertIn("'converged': False", error_msg)


if __name__ == '__main__':
    unittest.main()
