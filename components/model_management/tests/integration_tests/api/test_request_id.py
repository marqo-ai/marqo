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
            return {"status": "ok", "request_id": request.state.request_id}

        @self.app.post("/test")
        def test_post_endpoint(request: Request):
            return {"status": "ok", "request_id": request.state.request_id}

        self.client = TestClient(self.app, raise_server_exceptions=False)

    def _assert_valid_uuid_hex(self, request_id):
        """Helper to assert request ID is valid UUID hex format."""
        self.assertEqual(32, len(request_id))
        self.assertTrue(all(c in "0123456789abcdef" for c in request_id))

    def test_middleware_request_id_handling(self):
        """Test middleware generates and preserves request IDs for GET and POST requests."""
        test_cases = [
            (
                "GET",
                "/test",
                None,
                None,
                True,
            ),  # method, endpoint, custom_id, json_data, should_generate
            ("GET", "/test", "custom-get-id", None, False),
            ("POST", "/test", None, {"data": "test"}, True),
            ("POST", "/test", "custom-post-id", {"data": "test"}, False),
        ]

        for method, endpoint, custom_id, json_data, should_generate in test_cases:
            with self.subTest(
                method=method, custom_id=custom_id, should_generate=should_generate
            ):
                headers = {REQ_ID_HEADER: custom_id} if custom_id else {}

                if method == "GET":
                    response = self.client.get(endpoint, headers=headers)
                else:
                    response = self.client.post(
                        endpoint, json=json_data, headers=headers
                    )

                self.assertEqual(200, response.status_code)
                self.assertIn(REQ_ID_HEADER, response.headers)

                request_id = response.headers[REQ_ID_HEADER]
                self.assertIsNotNone(request_id)

                if should_generate:
                    self._assert_valid_uuid_hex(request_id)
                else:
                    self.assertEqual(custom_id, request_id)

                body = response.json()
                self.assertEqual(request_id, body["request_id"])

    def test_middleware_with_exception_handlers(self):
        """Test that middleware works correctly with exception handlers."""
        test_app = FastAPI()

        @test_app.get("/test/value-error")
        def test_value_error_endpoint(request: Request):
            self.assertTrue(hasattr(request.state, "request_id"))
            raise ValueError("Test error")

        @test_app.get("/test/invalid-arg-error")
        def test_invalid_arg_error_endpoint(request: Request):
            request_id = getattr(request.state, "request_id", None)
            self.assertIsNotNone(request_id)
            raise InvalidArgumentError("Test error")

        test_app.add_middleware(RequestIdMiddleware)
        register_exception_handlers(test_app)
        test_client = TestClient(test_app, raise_server_exceptions=False)

        test_cases = [
            ("/test/value-error", 500, None),  # endpoint, expected_status, check_detail
            ("/test/invalid-arg-error", 400, True),
        ]

        for endpoint, expected_status, check_detail in test_cases:
            with self.subTest(endpoint=endpoint):
                response = test_client.get(endpoint)
                self.assertEqual(expected_status, response.status_code)
                if check_detail:
                    body = response.json()
                    self.assertIn("detail", body)

    def test_generated_request_ids_are_unique(self):
        """Test that generated request IDs are unique and valid UUID hex format."""
        num_requests = 10
        request_ids = []

        for _ in range(num_requests):
            response = self.client.get("/test")
            request_id = response.headers[REQ_ID_HEADER]
            request_ids.append(request_id)
            # Verify each ID is valid UUID hex format
            self._assert_valid_uuid_hex(request_id)

        # All request IDs should be unique
        self.assertEqual(num_requests, len(set(request_ids)))

    def test_middleware_handles_empty_request_id_header(self):
        """Test that middleware generates new ID when header is empty."""
        response = self.client.get("/test", headers={REQ_ID_HEADER: ""})

        request_id = response.headers[REQ_ID_HEADER]
        self.assertIsNotNone(request_id)
        self._assert_valid_uuid_hex(request_id)

    def test_middleware_handles_custom_request_ids(self):
        """Test that middleware handles various custom request ID formats."""
        test_cases = [
            (" ", "whitespace"),
            ("  ", "multiple spaces"),
            ("\t", "tab"),
            ("\n", "newline"),
            ("test-id-with-dashes", "dashes"),
            ("test_id_with_underscores", "underscores"),
            ("test.id.with.dots", "dots"),
            ("test/id/with/slashes", "slashes"),
            ("test:id:with:colons", "colons"),
            ("x" * 1000, "very long"),
            ("test-id-123-abc", "alphanumeric"),
        ]

        for custom_id, description in test_cases:
            with self.subTest(description=description):
                response = self.client.get("/test", headers={REQ_ID_HEADER: custom_id})
                self.assertEqual(200, response.status_code)
                self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])
                body = response.json()
                self.assertEqual(custom_id, body["request_id"])

    def test_middleware_with_multiple_concurrent_requests(self):
        """Test that middleware correctly handles multiple concurrent requests."""
        num_requests = 20

        def make_request():
            response = self.client.get("/test")
            return response.headers[REQ_ID_HEADER]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            request_ids = [future.result() for future in futures]

        # All request IDs should be unique
        self.assertEqual(num_requests, len(set(request_ids)))


class TestRequestIdMiddlewareWithMainApp(TestCase):
    """Test RequestIdMiddleware integration with the main application."""

    @classmethod
    def setUpClass(cls):
        """Set up test client with the main application."""
        cls.client = TestClient(app)

    def test_request_id_present_in_endpoints(self):
        """Test that request ID is present in all endpoint responses."""
        test_cases = [
            ("GET", "/v1/healthz", None, 200),
            ("GET", "/openapi.json", None, 200),
            (
                "POST",
                "/v1/models/load",
                {},
                400,
            ),  # Invalid payload triggers validation error
        ]

        for method, endpoint, json_data, expected_status in test_cases:
            with self.subTest(method=method, endpoint=endpoint):
                if method == "GET":
                    response = self.client.get(endpoint)
                else:
                    response = self.client.post(endpoint, json=json_data)

                self.assertEqual(expected_status, response.status_code)
                self.assertIn(REQ_ID_HEADER, response.headers)
                self.assertIsNotNone(response.headers[REQ_ID_HEADER])

    def test_custom_request_id_preserved_in_endpoints(self):
        """Test that custom request IDs are preserved across endpoints and error handling."""
        test_cases = [
            ("GET", "/v1/healthz", None, 200, "healthz-test-123", False),
            ("POST", "/v1/models/load", {}, 400, "error-test-456", True),
            (
                "POST",
                "/v1/models/load",
                {"invalid": "payload"},
                400,
                "execution-order-test",
                True,
            ),
        ]

        for (
            method,
            endpoint,
            json_data,
            expected_status,
            custom_id,
            check_body,
        ) in test_cases:
            with self.subTest(endpoint=endpoint, custom_id=custom_id):
                headers = {REQ_ID_HEADER: custom_id}

                if method == "GET":
                    response = self.client.get(endpoint, headers=headers)
                else:
                    response = self.client.post(
                        endpoint, json=json_data, headers=headers
                    )

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual(custom_id, response.headers[REQ_ID_HEADER])

                if check_body:
                    body = response.json()
                    self.assertEqual(custom_id, body["request_id"])

    def test_request_ids_unique_across_different_endpoints(self):
        """Test that different endpoints generate unique request IDs."""
        endpoints = [
            ("GET", "/v1/healthz"),
            ("GET", "/openapi.json"),
        ]

        request_ids = []
        for method, endpoint in endpoints:
            response = self.client.get(endpoint)
            request_ids.append(response.headers[REQ_ID_HEADER])

        # All should be unique
        self.assertEqual(len(endpoints), len(set(request_ids)))
