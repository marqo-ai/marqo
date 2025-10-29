"""Integration tests for exception handlers."""

from unittest import TestCase

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from model_management.api.exception_handlers import register_exception_handlers
from model_management.api.request_id import RequestIdMiddleware
from model_management.errors.http_errors import (
    DependencyBadGatewayError,
    InternalServerError,
    InvalidArgumentError,
    NotFoundError,
    OperationConflictError,
)
from model_management.main import app
from model_management.services.errors import InternalServerError as ServiceInternalError
from model_management.services.errors import (
    ModelDownloadFailedError,
    ModelOperationInProgressError,
    ServiceError,
    TritonCommunicationError,
)


class TestExceptionHandlers(TestCase):
    """Integration tests for exception handler registration and behavior."""

    def setUp(self):
        """Set up a test FastAPI application with exception handlers."""
        self.app = FastAPI()
        self.app.add_middleware(RequestIdMiddleware)

        # Add test routes that raise various exceptions
        @self.app.get("/test/validation-error")
        def raise_validation_error():
            raise HTTPException(status_code=400, detail="Validation failed")

        @self.app.get("/test/invalid-argument")
        def raise_invalid_argument():
            raise InvalidArgumentError("Invalid argument provided")

        @self.app.get("/test/not-found")
        def raise_not_found():
            raise NotFoundError("Resource not found")

        @self.app.get("/test/operation-conflict")
        def raise_operation_conflict():
            raise OperationConflictError("Operation in progress")

        @self.app.get("/test/internal-error")
        def raise_internal_error():
            raise InternalServerError("Internal server error")

        @self.app.get("/test/dependency-error")
        def raise_dependency_error():
            raise DependencyBadGatewayError("Dependency error")

        @self.app.get("/test/service-download-error")
        def raise_service_download_error():
            raise ModelDownloadFailedError("Failed to download model")

        @self.app.get("/test/service-operation-in-progress")
        def raise_service_operation_in_progress():
            raise ModelOperationInProgressError("Model operation in progress")

        @self.app.get("/test/service-triton-error")
        def raise_service_triton_error():
            raise TritonCommunicationError("Triton communication failed")

        @self.app.get("/test/service-internal-error")
        def raise_service_internal_error():
            raise ServiceInternalError("Service internal error")

        @self.app.get("/test/unexpected-error")
        def raise_unexpected_error():
            raise ValueError("Unexpected error")

        @self.app.get("/test/zero-division")
        def raise_zero_division():
            _ = 1 / 0
            return {"result": "ok"}

        register_exception_handlers(self.app)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_invalid_argument_error_returns_400(self):
        """Test that InvalidArgumentError returns 400 with problem+json format."""
        response = self.client.get("/test/invalid-argument")

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("InvalidArgumentError", body["title"])
        self.assertEqual(400, body["status"])
        self.assertEqual("INVALID_ARGUMENT", body["code"])
        self.assertIn("Invalid argument provided", body["detail"])
        self.assertIn("instance", body)
        self.assertIn("request_id", body)

    def test_not_found_error_returns_404(self):
        """Test that NotFoundError returns 404 with problem+json format."""
        response = self.client.get("/test/not-found")

        self.assertEqual(404, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("NotFoundError", body["title"])
        self.assertEqual(404, body["status"])
        self.assertEqual("NOT_FOUND", body["code"])
        self.assertIn("Resource not found", body["detail"])

    def test_operation_conflict_error_returns_409(self):
        """Test that OperationConflictError returns 409 with problem+json format."""
        response = self.client.get("/test/operation-conflict")

        self.assertEqual(409, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("OperationConflictError", body["title"])
        self.assertEqual(409, body["status"])
        self.assertEqual("OPERATION_CONFLICT", body["code"])

    def test_internal_server_error_returns_500(self):
        """Test that InternalServerError returns 500 with problem+json format."""
        response = self.client.get("/test/internal-error")

        self.assertEqual(500, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("InternalServerError", body["title"])
        self.assertEqual(500, body["status"])
        self.assertEqual("INTERNAL_ERROR", body["code"])

    def test_dependency_bad_gateway_error_returns_502(self):
        """Test that DependencyBadGatewayError returns 502 with problem+json format."""
        response = self.client.get("/test/dependency-error")

        self.assertEqual(502, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("DependencyBadGatewayError", body["title"])
        self.assertEqual(502, body["status"])
        self.assertEqual("DEPENDENCY_BAD_GATEWAY", body["code"])

    def test_service_error_mapping_to_http_errors(self):
        """Test that service errors are correctly mapped to HTTP errors."""
        test_cases = [
            (
                "/test/service-download-error",
                400,
                "InvalidArgumentError",
                "INVALID_ARGUMENT",
                "Failed to download model",
            ),
            (
                "/test/service-operation-in-progress",
                409,
                "OperationConflictError",
                "OPERATION_CONFLICT",
                "Model operation in progress",
            ),
            (
                "/test/service-triton-error",
                502,
                "DependencyBadGatewayError",
                "DEPENDENCY_BAD_GATEWAY",
                "Triton communication failed",
            ),
            (
                "/test/service-internal-error",
                500,
                "InternalServerError",
                "INTERNAL_ERROR",
                "Service internal error",
            ),
        ]

        for (
            endpoint,
            expected_status,
            expected_title,
            expected_code,
            expected_detail,
        ) in test_cases:
            with self.subTest(endpoint=endpoint, expected_status=expected_status):
                response = self.client.get(endpoint)

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual(
                    "application/problem+json", response.headers["content-type"]
                )

                body = response.json()
                self.assertEqual(expected_title, body["title"])
                self.assertEqual(expected_status, body["status"])
                self.assertEqual(expected_code, body["code"])
                self.assertIn(expected_detail, body["detail"])

    def test_unexpected_error_returns_500(self):
        """Test that unexpected exceptions return 500 with generic error message."""
        response = self.client.get("/test/unexpected-error")

        self.assertEqual(500, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual("InternalServerError", body["title"])
        self.assertEqual(500, body["status"])
        # Generic error message should be present
        self.assertIsNotNone(body["detail"])

    def test_zero_division_error_returns_500(self):
        """Test that ZeroDivisionError is caught and returns 500."""
        response = self.client.get("/test/zero-division")

        self.assertEqual(500, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual(500, body["status"])

    def test_error_response_includes_instance_url(self):
        """Test that error responses include the instance URL from the request."""
        response = self.client.get("/test/invalid-argument")

        body = response.json()
        self.assertIn("instance", body)
        self.assertIn("/test/invalid-argument", body["instance"])

    def test_error_response_includes_request_id(self):
        """Test that error responses include the request ID for tracing."""
        custom_request_id = "test-error-trace-123"
        response = self.client.get(
            "/test/invalid-argument", headers={"x-request-id": custom_request_id}
        )

        self.assertEqual(400, response.status_code)

        body = response.json()
        self.assertEqual(custom_request_id, body["request_id"])
        self.assertEqual(custom_request_id, response.headers["x-request-id"])

    def test_problem_response_structure_follows_rfc7807(self):
        """Test that problem responses follow RFC 7807 structure."""
        response = self.client.get("/test/not-found")

        body = response.json()

        # Required RFC 7807 fields
        self.assertIn("type", body)
        self.assertIn("title", body)
        self.assertIn("status", body)

        # Our custom fields
        self.assertIn("code", body)
        self.assertIn("detail", body)
        self.assertIn("instance", body)
        self.assertIn("request_id", body)

        # Verify types
        self.assertIsInstance(body["type"], str)
        self.assertIsInstance(body["title"], str)
        self.assertIsInstance(body["status"], int)
        self.assertIsInstance(body["code"], str)

    def test_error_responses_are_json_serializable(self):
        """Test that all error responses are properly JSON serializable."""
        endpoints = [
            "/test/invalid-argument",
            "/test/not-found",
            "/test/operation-conflict",
            "/test/internal-error",
            "/test/dependency-error",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint)

                # If this doesn't raise an exception, the response is JSON serializable
                body = response.json()
                self.assertIsInstance(body, dict)

    def test_multiple_error_types_maintain_consistent_structure(self):
        """Test that different error types maintain consistent response structure."""
        endpoints = [
            "/test/invalid-argument",
            "/test/not-found",
            "/test/operation-conflict",
            "/test/internal-error",
        ]

        required_fields = [
            "type",
            "title",
            "status",
            "code",
            "detail",
            "instance",
            "request_id",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint)
                body = response.json()

                # Verify all required fields are present
                for field in required_fields:
                    self.assertIn(field, body, f"Field '{field}' missing in {endpoint}")

    def test_error_handler_with_custom_request_id_preserves_id(self):
        """Test that custom request IDs are preserved through error handling."""
        test_cases = [
            ("custom-id-1", "/test/invalid-argument"),
            ("custom-id-2", "/test/not-found"),
            ("custom-id-3", "/test/internal-error"),
        ]

        for request_id, endpoint in test_cases:
            with self.subTest(request_id=request_id, endpoint=endpoint):
                response = self.client.get(
                    endpoint, headers={"x-request-id": request_id}
                )

                body = response.json()
                self.assertEqual(request_id, body["request_id"])
                self.assertEqual(request_id, response.headers["x-request-id"])

    def test_service_error_without_mapping_defaults_to_internal_server_error(self):
        """Test that unmapped service errors default to InternalServerError."""
        # Create a custom service error that isn't explicitly mapped

        @self.app.get("/test/custom-service-error")
        def raise_custom_service_error():
            raise ServiceError("Custom unmapped error")

        response = self.client.get("/test/custom-service-error")

        self.assertEqual(500, response.status_code)
        body = response.json()
        self.assertEqual("InternalServerError", body["title"])
        self.assertEqual("INTERNAL_ERROR", body["code"])


class TestExceptionHandlerIntegrationWithMainApp(TestCase):
    """Test exception handlers with the actual main application."""

    @classmethod
    def setUpClass(cls):
        """Set up test client with the main application."""
        cls.client = TestClient(app)

    def test_invalid_json_payload_returns_400(self):
        """Test that invalid JSON payload returns 400 with validation error."""
        response = self.client.post(
            "/v1/models/load",
            content="invalid json{",
            headers={"content-type": "application/json"},
        )

        # FastAPI returns 422 for malformed JSON
        self.assertIn(response.status_code, [400, 422])

    def test_missing_content_type_with_body_returns_validation_error(self):
        """Test that missing content-type header with body returns validation error."""
        response = self.client.post("/v1/models/load", content='{"test": "data"}')

        # Should handle gracefully
        self.assertIn(response.status_code, [400, 415, 422])

    def test_empty_request_body_returns_validation_error(self):
        """Test that empty request body returns validation error."""
        response = self.client.post("/v1/models/load", json={})

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertIn("InvalidArgumentError", body["title"])

    def test_load_model_with_missing_required_fields_returns_validation_error(self):
        """Test that loading a model with missing required fields returns validation error."""
        # Use completely empty payload to trigger validation error before processing
        payload = {"tritonModelProperties": {}}

        response = self.client.post("/v1/models/load", json=payload)

        # Should return validation error
        self.assertEqual(400, response.status_code)
        body = response.json()
        self.assertIn("status", body)
        self.assertIn("InvalidArgumentError", body["title"])

    def test_load_model_with_invalid_data_types_returns_validation_error(self):
        """Test that loading a model with invalid data types returns validation error."""
        test_cases = [
            (
                {
                    "tritonModelProperties": {
                        "name": 12345,  # Should be string
                        "sources": ["s3://bucket/model.onnx"],
                        "input": [
                            {"name": "input", "dims": [1], "dataType": "TYPE_FP32"}
                        ],
                        "output": [
                            {"name": "output", "dims": [1], "dataType": "TYPE_FP32"}
                        ],
                    }
                },
                "name as integer",
            ),
            (
                {
                    "tritonModelProperties": {
                        "name": "test",
                        "sources": "not-a-list",  # Should be list
                        "input": [
                            {"name": "input", "dims": [1], "dataType": "TYPE_FP32"}
                        ],
                        "output": [
                            {"name": "output", "dims": [1], "dataType": "TYPE_FP32"}
                        ],
                    }
                },
                "sources as string",
            ),
            (
                {
                    "tritonModelProperties": {
                        "name": "test",
                        "sources": ["s3://bucket/model.onnx"],
                        "input": "not-a-list",  # Should be list
                        "output": [
                            {"name": "output", "dims": [1], "dataType": "TYPE_FP32"}
                        ],
                    }
                },
                "input as string",
            ),
        ]

        for payload, description in test_cases:
            with self.subTest(description=description):
                response = self.client.post("/v1/models/load", json=payload)

                self.assertEqual(400, response.status_code)
                body = response.json()
                self.assertIn("InvalidArgumentError", body["title"])

    def test_unload_model_with_invalid_query_param_type(self):
        """Test that unload with invalid query parameter type is handled."""
        # Boolean parameter with non-boolean value should be coerced or rejected
        response = self.client.post("/v1/models/test-model/unload?remove-files=invalid")

        # FastAPI may coerce "invalid" to False or return validation error
        self.assertIn(response.status_code, [200, 400, 422])

    def test_non_existent_endpoint_returns_404(self):
        """Test that accessing non-existent endpoint returns 404."""
        response = self.client.get("/v1/nonexistent/endpoint")

        self.assertEqual(404, response.status_code)

    def test_wrong_http_method_returns_405(self):
        """Test that using wrong HTTP method returns 405."""
        # GET on POST-only endpoint
        response = self.client.get("/v1/models/load")

        self.assertEqual(405, response.status_code)

    def test_error_responses_include_all_required_fields(self):
        """Test that error responses consistently include all required fields."""
        # Trigger a validation error
        response = self.client.post("/v1/models/load", json={})

        body = response.json()

        required_fields = ["type", "title", "status", "code", "instance", "request_id"]
        for field in required_fields:
            self.assertIn(
                field, body, f"Required field '{field}' missing in error response"
            )
            self.assertIsNotNone(body[field], f"Required field '{field}' is None")
