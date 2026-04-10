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
        mock_response.content = orjson.dumps(response_data)

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
        mock_response.content = orjson.dumps(response_data)

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
        mock_response.content = orjson.dumps(convergence_response)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError) as ctx:
                self.vespa_client.wait_for_application_convergence(timeout=1)

        expected_status = {
            'current_generation': 8,
            'wanted_generation': 9,
            'converged': False,
            'non_converged_services': [
                {'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8}
            ],
        }
        self.assertEqual(
            str(ctx.exception),
            f"Vespa application did not converge within 1 seconds. "
            f"The convergence status is {expected_status}"
        )

    def test_check_for_application_convergence_error_message_contains_status(self):
        """Test that check_for_application_convergence error includes convergence status details."""
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
        mock_response.content = orjson.dumps(convergence_response)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError) as ctx:
                self.vespa_client.check_for_application_convergence()

        expected_status = {
            'current_generation': 8,
            'wanted_generation': 9,
            'converged': False,
            'non_converged_services': [
                {'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8}
            ],
        }
        self.assertEqual(
            str(ctx.exception),
            f"Vespa application has not converged. "
            f"The convergence status is {expected_status}"
        )

    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_uses_exponential_backoff(self, mock_time):
        """Test that wait_for_application_convergence sleeps 1s for the first 8 attempts,
        then uses exponential backoff (2s, 4s, 8s, 16s cap)."""
        from marqo.vespa.exceptions import VespaNotConvergedError

        # Simulate time progressing by the amount slept
        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        convergence_response = {
            'currentGeneration': 8,
            'wantedGeneration': 9,
            'converged': False,
            'services': []
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(convergence_response)

        # Use a large timeout so the last few sleeps aren't clamped by remaining time
        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError):
                self.vespa_client.wait_for_application_convergence(timeout=200)

        sleep_calls = [call.args[0] for call in mock_time.sleep.call_args_list]

        # First 8 attempts: 1s each
        self.assertEqual(sleep_calls[:8], [1, 1, 1, 1, 1, 1, 1, 1])
        # Attempt 8: 2s, attempt 9: 4s, attempt 10: 8s, attempt 11: 16s (cap)
        self.assertEqual(sleep_calls[8], 2)
        self.assertEqual(sleep_calls[9], 4)
        self.assertEqual(sleep_calls[10], 8)
        self.assertEqual(sleep_calls[11], 16)
        # After cap, remaining full sleeps should be 16s
        for s in sleep_calls[12:-1]:
            self.assertEqual(s, 16)

    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_does_not_sleep_past_timeout(self, mock_time):
        """Test that sleep duration is capped to not exceed the remaining timeout."""
        from marqo.vespa.exceptions import VespaNotConvergedError

        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        convergence_response = {
            'currentGeneration': 8,
            'wantedGeneration': 9,
            'converged': False,
            'services': []
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(convergence_response)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError):
                self.vespa_client.wait_for_application_convergence(timeout=10)

        sleep_calls = [call.args[0] for call in mock_time.sleep.call_args_list]
        # Total sleep should not exceed the timeout
        self.assertLessEqual(sum(sleep_calls), 10)
        # No individual sleep should exceed the remaining time
        elapsed = 0.0
        for s in sleep_calls:
            self.assertLessEqual(s, 10 - elapsed + 0.001)  # small float tolerance
            elapsed += s

    def test_wait_for_convergence_returns_immediately_when_converged(self):
        """Test that wait returns immediately when already converged, with no sleep."""
        converged_response = {
            'currentGeneration': 9,
            'wantedGeneration': 9,
            'converged': True,
            'services': []
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(converged_response)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            # Should not raise
            self.vespa_client.wait_for_application_convergence(timeout=10)

    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_returns_after_retries(self, mock_time):
        """Test that wait returns successfully when convergence happens after a few retries."""
        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        not_converged = {
            'currentGeneration': 8, 'wantedGeneration': 9, 'converged': False, 'services': []
        }
        converged = {
            'currentGeneration': 9, 'wantedGeneration': 9, 'converged': True, 'services': []
        }

        def make_response(data):
            r = Mock()
            r.status_code = 200
            r.content = orjson.dumps(data)
            return r

        responses = [make_response(not_converged)] * 3 + [make_response(converged)]
        with patch.object(self.vespa_client.http_client, 'get', side_effect=responses):
            self.vespa_client.wait_for_application_convergence(timeout=120)

        # Should have slept 3 times (1s each, all within first 8 attempts)
        self.assertEqual(mock_time.sleep.call_count, 3)

    @patch('marqo.vespa.vespa_client.logger')
    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_logs_warning_on_slow_convergence(self, mock_time, mock_logger):
        """Test that a warning is logged when convergence takes longer than 10 seconds."""
        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        not_converged = {
            'currentGeneration': 8, 'wantedGeneration': 9, 'converged': False, 'services': []
        }
        converged = {
            'currentGeneration': 9, 'wantedGeneration': 9, 'converged': True, 'services': []
        }

        def make_response(data):
            r = Mock()
            r.status_code = 200
            r.content = orjson.dumps(data)
            return r

        responses = [make_response(not_converged)] * 11 + [make_response(converged)]
        with patch.object(self.vespa_client.http_client, 'get', side_effect=responses):
            self.vespa_client.wait_for_application_convergence(timeout=120)

        # Should have logged a warning about slow convergence on success
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        converged_warnings = [c for c in warning_calls if 'converged after' in c]
        self.assertTrue(len(converged_warnings) > 0,
                        f"Expected a 'converged after' warning, got: {warning_calls}")

    @patch('marqo.vespa.vespa_client.logger')
    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_logs_warning_with_status_when_slow(self, mock_time, mock_logger):
        """Test that warnings include convergence status when not converged after 10s."""
        from marqo.vespa.exceptions import VespaNotConvergedError

        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        not_converged_response = {
            'currentGeneration': 8,
            'wantedGeneration': 9,
            'converged': False,
            'services': [
                {'host': 'node2', 'port': 8080, 'type': 'container', 'currentGeneration': 8},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(not_converged_response)

        with patch.object(self.vespa_client.http_client, 'get', return_value=mock_response):
            with self.assertRaises(VespaNotConvergedError):
                self.vespa_client.wait_for_application_convergence(timeout=20)

        # Should have logged warnings with convergence status after 10s
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        status_warnings = [c for c in warning_calls if 'has not converged after' in c]
        self.assertTrue(len(status_warnings) > 0,
                        f"Expected 'has not converged after' warnings, got: {warning_calls}")
        # Verify the status dict is included
        self.assertTrue(any('node2' in c for c in status_warnings),
                        f"Expected non-converged service details in warnings, got: {status_warnings}")

    @patch('marqo.vespa.vespa_client.logger')
    @patch('marqo.vespa.vespa_client.time')
    def test_wait_for_convergence_no_warning_when_fast(self, mock_time, mock_logger):
        """Test that no warning is logged when convergence happens within 10 seconds."""
        current_time = [0.0]

        def fake_time():
            return current_time[0]

        def fake_sleep(duration):
            current_time[0] += duration

        mock_time.time.side_effect = fake_time
        mock_time.sleep.side_effect = fake_sleep

        not_converged = {
            'currentGeneration': 8, 'wantedGeneration': 9, 'converged': False, 'services': []
        }
        converged = {
            'currentGeneration': 9, 'wantedGeneration': 9, 'converged': True, 'services': []
        }

        def make_response(data):
            r = Mock()
            r.status_code = 200
            r.content = orjson.dumps(data)
            return r

        responses = [make_response(not_converged)] * 3 + [make_response(converged)]
        with patch.object(self.vespa_client.http_client, 'get', side_effect=responses):
            self.vespa_client.wait_for_application_convergence(timeout=120)

        mock_logger.warning.assert_not_called()

    def test_query_sends_connection_close_header_when_should_drop(self):
        """Test that query sends 'Connection: close' header when _should_drop_connection returns True."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"root": {"id": "test", "relevance": 1.0, "children": []}}'

        with patch.object(self.vespa_client, '_should_drop_connection', return_value=True), \
             patch.object(httpx.Client, 'post', return_value=mock_response) as mock_post:
            self.vespa_client.query(yql="select * from sources * where test;")
            headers = mock_post.call_args.kwargs["headers"]
            self.assertEqual(headers, {"Connection": "close"})

    def test_query_no_connection_close_header_when_should_not_drop(self):
        """Test that query does not send 'Connection: close' header when _should_drop_connection returns False."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"root": {"id": "test", "relevance": 1.0, "children": []}}'

        with patch.object(self.vespa_client, '_should_drop_connection', return_value=False), \
             patch.object(httpx.Client, 'post', return_value=mock_response) as mock_post:
            self.vespa_client.query(yql="select * from sources * where test;")
            headers = mock_post.call_args.kwargs["headers"]
            if headers:
                self.assertNotIn("Connection", headers)

    @patch('marqo.vespa.vespa_client.settings')
    def test_query_no_connection_close_header_when_rate_is_zero(self, mock_settings):
        """Test that query does not send 'Connection: close' header when rate is 0 (default)."""
        mock_settings.marqo_search_random_connection_close_rate = 0

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"root": {"id": "test", "relevance": 1.0, "children": []}}'

        with patch.object(httpx.Client, 'post', return_value=mock_response) as mock_post:
            self.vespa_client.query(yql="select * from sources * where test;")
            headers = mock_post.call_args.kwargs["headers"]
            if headers:
                self.assertNotIn("Connection", headers)

    @patch('marqo.vespa.vespa_client.settings')
    def test_query_connection_close_header_with_rate_1(self, mock_settings):
        """Test that query always sends 'Connection: close' header when rate is 1.0."""
        mock_settings.marqo_search_random_connection_close_rate = 1.0

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"root": {"id": "test", "relevance": 1.0, "children": []}}'

        with patch.object(httpx.Client, 'post', return_value=mock_response) as mock_post:
            self.vespa_client.query(yql="select * from sources * where test;", drop_connection_random_seed=12)
            headers = mock_post.call_args.kwargs["headers"]
            self.assertEqual(headers, {"Connection": "close"})

    @patch('marqo.vespa.vespa_client.settings')
    def test_should_drop_connection_is_reproducible_with_same_seed(self, mock_settings):
        """Test that _should_drop_connection returns the same result for the same seed."""
        mock_settings.marqo_search_random_connection_close_rate = 0.5
        result1 = self.vespa_client._should_drop_connection(seed=42)
        result2 = self.vespa_client._should_drop_connection(seed=42)
        self.assertEqual(result1, result2)

    @patch('marqo.vespa.vespa_client.logger')
    def test_wait_for_convergence_retries_on_httpx_timeout(self, mock_logger):
        """Test that httpx timeout exceptions are caught and retried."""
        not_converged_response = {
            'currentGeneration': 8, 'wantedGeneration': 9, 'converged': False, 'services': []
        }
        converged_response = {
            'currentGeneration': 9, 'wantedGeneration': 9, 'converged': True, 'services': []
        }

        mock_response_not_converged = Mock()
        mock_response_not_converged.status_code = 200
        mock_response_not_converged.content = orjson.dumps(not_converged_response)

        mock_response_converged = Mock()
        mock_response_converged.status_code = 200
        mock_response_converged.content = orjson.dumps(converged_response)

        # First call raises timeout, second returns not converged, third returns converged
        with patch.object(self.vespa_client.http_client, 'get',
                          side_effect=[httpx.TimeoutException("timeout"),
                                       mock_response_not_converged,
                                       mock_response_converged]):
            self.vespa_client.wait_for_application_convergence(timeout=10)

        # Should have logged the timeout error
        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any('timed out' in c.lower() for c in error_calls))


    @patch('marqo.vespa.vespa_client.httpx.AsyncClient')
    def test_get_batch_preserves_input_order_with_out_of_order_completion(self, mock_async_client_class):
        """Test that get_batch returns responses in the same order as the input IDs,
        even when the underlying async requests complete out of order.

        This is a critical assumption used by partial_update_documents, which maps
        get_batch response positions back to original document indices."""
        from marqo.vespa.models.get_document_response import GetBatchDocumentResponse

        schema = 'test_schema'
        ids = ['doc_a', 'doc_b', 'doc_c', 'doc_d']

        for not_found_ids, subtest_name in [
            (set(), 'all documents found'),
            ({'doc_a', 'doc_c'}, 'some documents not found'),
        ]:
            with self.subTest(msg=subtest_name):
                completion_order = []

                async def mock_get_document(semaphore, async_client, doc_id, fields, schema_name, timeout,
                                            _not_found=not_found_ids, _order=completion_order):
                    # Stagger so tasks complete in reverse: doc_d, doc_c, doc_b, doc_a
                    delays = {'doc_a': 0.04, 'doc_b': 0.03, 'doc_c': 0.02, 'doc_d': 0.01}
                    await asyncio.sleep(delays[doc_id])
                    _order.append(doc_id)

                    if doc_id in _not_found:
                        return GetBatchDocumentResponse(
                            status=404,
                            pathId=f'/document/v1/{schema_name}/{schema_name}/docid/{doc_id}',
                            id=f'id:{schema_name}:{schema_name}::{doc_id}',
                            message='Document not found',
                        )
                    return GetBatchDocumentResponse(
                        status=200,
                        pathId=f'/document/v1/{schema_name}/{schema_name}/docid/{doc_id}',
                        id=f'id:{schema_name}:{schema_name}::{doc_id}',
                        fields={'doc_id_field': doc_id}
                    )

                with patch.object(self.vespa_client, '_get_document_async', side_effect=mock_get_document):
                    result = self.vespa_client.get_batch(ids=ids, schema=schema)

                # Verify requests actually completed out of order
                self.assertNotEqual(completion_order, ids,
                                    "Test setup error: requests should complete out of order due to staggered delays")

                # Verify responses are returned in the same order as input IDs
                self.assertEqual(len(result.responses), len(ids))
                for i, expected_id in enumerate(ids):
                    resp = result.responses[i]
                    actual_id = resp.id.split('::')[-1] if resp.id else None
                    self.assertEqual(actual_id, expected_id,
                                     f"Response at position {i} should be for '{expected_id}', got '{actual_id}'")

    def test_get_document_async_catches_invalid_url_error(self):
        """Test that _get_document_async catches httpx.InvalidURL and returns a 400 response
        instead of crashing. This handles IDs with non-printable ASCII characters."""
        from marqo.vespa.models.get_document_response import GetBatchDocumentResponse

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.InvalidURL(
            "Invalid non-printable ASCII character in URL"
        )

        async def _run():
            semaphore = asyncio.Semaphore(1)
            return await self.vespa_client._get_document_async(
                semaphore, mock_client, 'doc\x00with\x01control\x7fchars', None, 'test_schema', 60
            )

        result = asyncio.run(_run())

        self.assertIsInstance(result, GetBatchDocumentResponse)
        self.assertEqual(result.status, 400)
        self.assertIn("Invalid document ID", result.message)
        self.assertIn("doc\x00with\x01control\x7fchars", result.message)
        self.assertEqual(result.path_id, "")
        self.assertIsNone(result.document)

    @patch('marqo.vespa.vespa_client.httpx.AsyncClient')
    def test_get_batch_handles_invalid_url_ids_with_valid_ids(self, mock_async_client_class):
        """Test that get_batch returns proper 400 responses for IDs with non-printable ASCII
        characters while still returning correct results for valid IDs, preserving order."""
        from marqo.vespa.models.get_document_response import GetBatchDocumentResponse

        schema = 'test_schema'
        ids = ['valid_doc', 'doc\x00invalid', 'another_valid']

        async def mock_get_document(semaphore, async_client, doc_id, fields, schema_name, timeout):
            if '\x00' in doc_id:
                return GetBatchDocumentResponse(
                    status=400,
                    pathId="",
                    message=f"Invalid document ID: {doc_id}. Original error: Invalid URL"
                )
            return GetBatchDocumentResponse(
                status=200,
                pathId=f'/document/v1/{schema_name}/{schema_name}/docid/{doc_id}',
                id=f'id:{schema_name}:{schema_name}::{doc_id}',
                fields={'title': f'Title for {doc_id}'}
            )

        with patch.object(self.vespa_client, '_get_document_async', side_effect=mock_get_document):
            result = self.vespa_client.get_batch(ids=ids, schema=schema)

        self.assertEqual(len(result.responses), 3)
        self.assertTrue(result.errors)

        # First: valid doc
        self.assertEqual(result.responses[0].status, 200)
        self.assertEqual(result.responses[0].document.fields['title'], 'Title for valid_doc')

        # Second: invalid URL doc
        self.assertEqual(result.responses[1].status, 400)
        self.assertIn("Invalid document ID", result.responses[1].message)
        self.assertIsNone(result.responses[1].document)

        # Third: valid doc
        self.assertEqual(result.responses[2].status, 200)
        self.assertEqual(result.responses[2].document.fields['title'], 'Title for another_valid')


if __name__ == '__main__':
    unittest.main()
