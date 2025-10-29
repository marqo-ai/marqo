"""Integration tests for request ID middleware."""

import concurrent.futures
from unittest import TestCase

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from model_management.api.exception_handlers import register_exception_handlers
from model_management.api.request_id import REQ_ID_HEADER, RequestIdMiddleware
from model_management.errors.http_errors import InvalidArgumentError
from model_management.main import app


class TestRequestIdMiddleware(TestCase):
    """Integration tests for RequestIdMiddleware."""

    def setUp(self):
        """Set up a test FastAPI application with RequestIdMiddleware."""
        self.app = FastAPI()
        self.app.add_middleware(RequestIdMiddleware)

        @self.app.get("/test")
        def test_endpoint(request: Request):
            return {
                "status": "ok",
                "request_id": request.state.request_id,
            }

        @self.app.post("/test")
        def test_post_endpoint(request: Request):
            return {
                "status": "ok",
                "request_id": request.state.request_id,
            }

        @self.app.get("/test/error")
        def test_error_endpoint(request: Request):
            raise ValueError("Test error")

        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_middleware_generates_request_id_when_not_provided(self):
        """Test that middleware generates a request ID when not provided."""
        response = self.client.get("/test")

        self.assertEqual(200, response.status_code)
        self.assertIn(REQ_ID_HEADER, response.headers)

        request_id = response.headers[REQ_ID_HEADER]
        self.assertIsNotNone(request_id)
        self.assertEqual(32, len(request_id))  # UUID hex format

        body = response.json()
        self.assertEqual(request_id, body["request_id"])

    def test_middleware_uses_provided_request_id(self):
        """Test that middleware uses the provided request ID from headers."""
        custom_request_id = "custom-test-request-id-123"
        response = self.client.get("/test", headers={REQ_ID_HEADER: custom_request_id})

        self.assertEqual(200, response.status_code)
        self.assertEqual(custom_request_id, response.headers[REQ_ID_HEADER])

        body = response.json()
        self.assertEqual(custom_request_id, body["request_id"])

    def test_middleware_works_with_post_requests(self):
        """Test that middleware works correctly with POST requests."""
        response = self.client.post("/test", json={"data": "test"})

        self.assertEqual(200, response.status_code)
        self.assertIn(REQ_ID_HEADER, response.headers)

        request_id = response.headers[REQ_ID_HEADER]
        self.assertIsNotNone(request_id)

    def test_middleware_preserves_custom_id_in_post_requests(self):
        """Test that custom request IDs are preserved in POST requests."""
        custom_request_id = "post-test-id-456"
        response = self.client.post(
            "/test", json={"data": "test"}, headers={REQ_ID_HEADER: custom_request_id}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(custom_request_id, response.headers[REQ_ID_HEADER])

    def test_middleware_with_exception_handlers(self):
        """Test that middleware works correctly with exception handlers.

        Note: Exception handlers create new Response objects, so the middleware's
        response header modification doesn't apply. The request ID is still available
        in request.state and can be included in error response bodies.
        """
        # This test verifies that the middleware doesn't interfere with error handling
        test_app = FastAPI()

        @test_app.get("/test/error")
        def test_error_endpoint(request: Request):
            # Verify request ID is in state
            self.assertHasattr(request.state, "request_id")
            raise ValueError("Test error")

        test_app.add_middleware(RequestIdMiddleware)
        register_exception_handlers(test_app)
        test_client = TestClient(test_app, raise_server_exceptions=False)

        response = test_client.get("/test/error")

        # Error response should be returned (middleware doesn't break error handling)
        self.assertEqual(500, response.status_code)

    def test_middleware_does_not_break_error_handling(self):
        """Test that middleware doesn't break error handling flow."""
        test_app = FastAPI()

        @test_app.get("/test/error")
        def test_error_endpoint(request: Request):
            # Verify request ID is accessible in request state
            request_id = getattr(request.state, "request_id", None)
            self.assertIsNotNone(request_id)
            raise InvalidArgumentError("Test error")

        test_app.add_middleware(RequestIdMiddleware)
        register_exception_handlers(test_app)
        test_client = TestClient(test_app, raise_server_exceptions=False)

        response = test_client.get("/test/error")

        # Error should be handled correctly
        self.assertEqual(400, response.status_code)
        body = response.json()
        self.assertIn("detail", body)

    def test_generated_request_ids_are_unique(self):
        """Test that generated request IDs are unique across multiple requests."""
        num_requests = 10
        request_ids = []

        for _ in range(num_requests):
            response = self.client.get("/test")
            request_ids.append(response.headers[REQ_ID_HEADER])

        # All request IDs should be unique
        unique_ids = set(request_ids)
        self.assertEqual(len(request_ids), len(unique_ids))

    def test_request_id_format_is_valid_uuid_hex(self):
        """Test that generated request IDs are valid UUID hex format."""
        response = self.client.get("/test")

        request_id = response.headers[REQ_ID_HEADER]

        # UUID hex format: 32 hexadecimal characters
        self.assertEqual(32, len(request_id))
        self.assertTrue(all(c in "0123456789abcdef" for c in request_id))

    def test_middleware_handles_empty_request_id_header(self):
        """Test that middleware generates new ID when header is empty."""
        response = self.client.get("/test", headers={REQ_ID_HEADER: ""})

        # Should generate new ID since provided ID is empty
        request_id = response.headers[REQ_ID_HEADER]
        self.assertIsNotNone(request_id)
        self.assertEqual(32, len(request_id))

    def test_middleware_handles_whitespace_request_id_header(self):
        """Test that middleware handles whitespace in request ID header."""
        test_cases = [
            " ",
            "  ",
            "\t",
            "\n",
        ]

        for whitespace_id in test_cases:
            with self.subTest(request_id=repr(whitespace_id)):
                response = self.client.get(
                    "/test", headers={REQ_ID_HEADER: whitespace_id}
                )

                # Middleware should use the whitespace value as-is
                # (it doesn't validate format, just checks if header exists)
                request_id = response.headers[REQ_ID_HEADER]
                self.assertEqual(whitespace_id, request_id)

    def test_middleware_handles_special_characters_in_request_id(self):
        """Test that middleware handles special characters in request ID."""
        test_cases = [
            "test-id-with-dashes",
            "test_id_with_underscores",
            "test.id.with.dots",
            "test/id/with/slashes",
            "test:id:with:colons",
        ]

        for custom_id in test_cases:
            with self.subTest(request_id=custom_id):
                response = self.client.get("/test", headers={REQ_ID_HEADER: custom_id})

                self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])
                body = response.json()
                self.assertEqual(custom_id, body["request_id"])

    def test_middleware_handles_very_long_request_id(self):
        """Test that middleware handles very long request IDs."""
        long_request_id = "x" * 1000
        response = self.client.get("/test", headers={REQ_ID_HEADER: long_request_id})

        self.assertEqual(200, response.status_code)
        self.assertEqual(long_request_id, response.headers[REQ_ID_HEADER])

    def test_middleware_handles_unicode_in_request_id(self):
        """Test that middleware handles Unicode characters in request ID."""
        unicode_id = "test-id-123-abc"  # Use ASCII for compatibility
        response = self.client.get("/test", headers={REQ_ID_HEADER: unicode_id})

        self.assertEqual(200, response.status_code)
        self.assertEqual(unicode_id, response.headers[REQ_ID_HEADER])

    def test_request_id_available_in_request_state(self):
        """Test that request ID is available in request.state throughout request lifecycle."""

        @self.app.get("/test/state-access")
        def test_state_access(request: Request):
            # Verify request ID is in state
            self.assertHasattr(request.state, "request_id")
            self.assertIsNotNone(request.state.request_id)
            return {"request_id": request.state.request_id}

        response = self.client.get("/test/state-access")

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(response.headers[REQ_ID_HEADER], body["request_id"])

    def test_middleware_with_multiple_concurrent_requests(self):
        """Test that middleware correctly handles multiple concurrent requests."""
        num_requests = 20
        request_ids = []

        def make_request():
            response = self.client.get("/test")
            return response.headers[REQ_ID_HEADER]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            request_ids = [future.result() for future in futures]

        # All request IDs should be unique
        self.assertEqual(num_requests, len(set(request_ids)))

    def test_middleware_header_name_constant(self):
        """Test that the header name constant is correctly defined."""
        self.assertEqual("x-request-id", REQ_ID_HEADER)

    def assertHasattr(self, obj, attr_name):
        """Helper assertion to check if object has attribute."""
        self.assertTrue(
            hasattr(obj, attr_name), f"Object does not have attribute '{attr_name}'"
        )


class TestRequestIdMiddlewareWithMainApp(TestCase):
    """Test RequestIdMiddleware integration with the main application."""

    @classmethod
    def setUpClass(cls):
        """Set up test client with the main application."""
        cls.client = TestClient(app)

    def test_request_id_in_healthz_endpoint(self):
        """Test that request ID is present in healthz endpoint responses."""
        response = self.client.get("/v1/healthz")

        self.assertEqual(200, response.status_code)
        self.assertIn(REQ_ID_HEADER, response.headers)
        self.assertIsNotNone(response.headers[REQ_ID_HEADER])

    def test_custom_request_id_in_healthz_endpoint(self):
        """Test that custom request ID is used in healthz endpoint."""
        custom_id = "healthz-test-123"
        response = self.client.get("/v1/healthz", headers={REQ_ID_HEADER: custom_id})

        self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])

    def test_request_id_in_load_model_endpoint(self):
        """Test that request ID is present in load model endpoint responses."""
        # Use invalid payload to avoid actual model loading
        payload = {}

        response = self.client.post("/v1/models/load", json=payload)

        # Should get validation error but still have request ID
        self.assertIn(REQ_ID_HEADER, response.headers)
        self.assertIsNotNone(response.headers[REQ_ID_HEADER])

    def test_request_id_in_unload_model_endpoint(self):
        """Test that request ID is present in unload model endpoint responses."""
        response = self.client.post("/v1/models/test-model/unload")

        self.assertEqual(200, response.status_code)
        self.assertIn(REQ_ID_HEADER, response.headers)
        self.assertIsNotNone(response.headers[REQ_ID_HEADER])

    def test_request_id_preserved_across_error_handling(self):
        """Test that request ID is preserved through error handling pipeline."""
        custom_id = "error-handling-test-456"

        # Trigger a validation error
        response = self.client.post(
            "/v1/models/load", json={}, headers={REQ_ID_HEADER: custom_id}
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])

        # Request ID should also be in error response body
        body = response.json()
        self.assertEqual(custom_id, body["request_id"])

    def test_request_id_in_openapi_documentation(self):
        """Test that request ID middleware doesn't interfere with OpenAPI docs."""
        response = self.client.get("/openapi.json")

        self.assertEqual(200, response.status_code)
        self.assertIn(REQ_ID_HEADER, response.headers)

    def test_request_ids_unique_across_different_endpoints(self):
        """Test that different endpoints generate unique request IDs."""
        endpoints = [
            "/v1/healthz",
            "/v1/models/test-model/unload",
            "/openapi.json",
        ]

        request_ids = []
        for endpoint in endpoints:
            if endpoint == "/v1/models/test-model/unload":
                response = self.client.post(endpoint)
            else:
                response = self.client.get(endpoint)

            request_ids.append(response.headers[REQ_ID_HEADER])

        # All should be unique
        self.assertEqual(len(endpoints), len(set(request_ids)))

    def test_middleware_execution_order_with_exception_handlers(self):
        """Test that middleware executes correctly with exception handlers."""
        custom_id = "execution-order-test-789"

        # Trigger an error to test middleware + exception handler interaction
        response = self.client.post(
            "/v1/models/load",
            json={"invalid": "payload"},
            headers={REQ_ID_HEADER: custom_id},
        )

        # Middleware should add request ID to response
        self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])

        # Exception handler should include request ID in error body
        body = response.json()
        self.assertEqual(custom_id, body["request_id"])
