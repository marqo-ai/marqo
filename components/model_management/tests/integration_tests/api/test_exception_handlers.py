"""Integration tests for exception handlers."""

from unittest import TestCase

from fastapi import FastAPI
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

        @self.app.get("/test/custom-service-error")
        def raise_custom_service_error():
            raise ServiceError("Custom unmapped error")

        register_exception_handlers(self.app)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def _assert_problem_json_response(
        self,
        response,
        expected_status,
        expected_title,
        expected_code,
        expected_detail=None,
    ):
        """Helper method to assert common problem+json response structure."""
        self.assertEqual(expected_status, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        self.assertEqual(expected_title, body["title"])
        self.assertEqual(expected_status, body["status"])
        self.assertEqual(expected_code, body["code"])

        if expected_detail:
            self.assertIn(expected_detail, body["detail"])

        # Verify required fields exist
        self.assertIn("type", body)
        self.assertIn("detail", body)
        self.assertIn("instance", body)
        self.assertIn("request_id", body)

        return body

    def test_http_errors_return_correct_status_and_format(self):
        """Test that HTTP errors return correct status codes with problem+json format."""
        test_cases = [
            (
                "/test/invalid-argument",
                400,
                "InvalidArgumentError",
                "INVALID_ARGUMENT",
                "Invalid argument provided",
            ),
            (
                "/test/not-found",
                404,
                "NotFoundError",
                "NOT_FOUND",
                "Resource not found",
            ),
            (
                "/test/operation-conflict",
                409,
                "OperationConflictError",
                "OPERATION_CONFLICT",
                "Operation in progress",
            ),
            (
                "/test/internal-error",
                500,
                "InternalServerError",
                "INTERNAL_ERROR",
                "Internal server error",
            ),
            (
                "/test/dependency-error",
                502,
                "DependencyBadGatewayError",
                "DEPENDENCY_BAD_GATEWAY",
                "Dependency error",
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
                self._assert_problem_json_response(
                    response,
                    expected_status,
                    expected_title,
                    expected_code,
                    expected_detail,
                )

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
                self._assert_problem_json_response(
                    response,
                    expected_status,
                    expected_title,
                    expected_code,
                    expected_detail,
                )

    def test_unexpected_errors_return_500(self):
        """Test that unexpected exceptions return 500 with InternalServerError format."""
        response = self.client.get("/test/unexpected-error")
        body = self._assert_problem_json_response(
            response, 500, "InternalServerError", "INTERNAL_ERROR"
        )
        self.assertIsNotNone(body["detail"])

    def test_custom_request_id_preservation(self):
        """Test that custom request IDs are preserved in error responses."""
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

    def test_unmapped_service_error_defaults_to_internal_server_error(self):
        """Test that unmapped service errors default to InternalServerError."""
        response = self.client.get("/test/custom-service-error")
        self._assert_problem_json_response(
            response, 500, "InternalServerError", "INTERNAL_ERROR"
        )


class TestExceptionHandlerIntegrationWithMainApp(TestCase):
    """Test exception handlers with the actual main application."""

    @classmethod
    def setUpClass(cls):
        """Set up test client with the main application."""
        cls.client = TestClient(app)

    def test_validation_errors_return_400(self):
        """Test that various validation errors return 400 with InvalidArgumentError."""
        test_cases = [
            ({}, "empty request body"),
            ({"tritonModelProperties": {}}, "missing required fields"),
        ]

        for payload, description in test_cases:
            with self.subTest(description=description):
                response = self.client.post("/v1/models/load", json=payload)
                self.assertEqual(400, response.status_code)
                self.assertEqual(
                    "application/problem+json", response.headers["content-type"]
                )
                body = response.json()
                self.assertIn("InvalidArgumentError", body["title"])
                # Verify required fields exist
                for field in [
                    "type",
                    "title",
                    "status",
                    "code",
                    "instance",
                    "request_id",
                ]:
                    self.assertIn(field, body)
                    self.assertIsNotNone(body[field])

    def test_invalid_data_types_return_400(self):
        """Test that invalid data types in request return 400 validation error."""
        test_cases = [
            (
                {
                    "tritonModelProperties": {
                        "name": 12345,
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
                        "sources": "not-a-list",
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
                        "input": "not-a-list",
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

    def test_http_method_errors(self):
        """Test that HTTP method and endpoint errors are handled correctly."""
        test_cases = [
            ("GET", "/v1/models/load", 405, "wrong HTTP method"),
            ("GET", "/v1/nonexistent/endpoint", 404, "non-existent endpoint"),
        ]

        for method, endpoint, expected_status, description in test_cases:
            with self.subTest(description=description):
                if method == "GET":
                    response = self.client.get(endpoint)
                self.assertEqual(expected_status, response.status_code)
