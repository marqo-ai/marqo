import asyncio
import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import model_management.errors.http_errors as http_errors
import model_management.services.errors as service_errors
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from model_management.api.exception_handlers import (
    _app_error_handler,
    _catch_all_handler,
    _map_service_errors_to_http_errors,
    _normalize_validation_errors,
    _problem_response,
    _service_error_handler,
    _validation_error_handler,
    register_exception_handlers,
)
from model_management.errors.base import AppError
from model_management.errors.http_errors import (
    DependencyTimeoutError,
    InternalServerError,
    InvalidArgumentError,
    NotFoundError,
    OperationConflictError,
)
from starlette.requests import Request


class TestExceptionHandlers(TestCase):
    """Test class for exception handlers in model_management.api.exception_handlers."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.mock_request = MagicMock(spec=Request)
        self.mock_request.url = "http://testserver/api/v1/models/load"
        self.mock_request.state.request_id = "test-request-id-123"

    def test_register_exception_handlers(self):
        """Test that all exception handlers are registered correctly."""
        with patch.object(self.app, "add_exception_handler") as mock_add_handler:
            register_exception_handlers(self.app)

            self.assertEqual(4, mock_add_handler.call_count)

            # Verify each handler was registered
            call_args_list = [call[0] for call in mock_add_handler.call_args_list]
            exception_types = [args[0] for args in call_args_list]
            handlers = [args[1] for args in call_args_list]

            self.assertIn(RequestValidationError, exception_types)
            self.assertIn(service_errors.ServiceError, exception_types)
            self.assertIn(AppError, exception_types)
            self.assertIn(Exception, exception_types)

            self.assertIn(_validation_error_handler, handlers)
            self.assertIn(_service_error_handler, handlers)
            self.assertIn(_app_error_handler, handlers)
            self.assertIn(_catch_all_handler, handlers)

    def test_problem_response(self):
        """Test _problem_response converts AppError to Problem JSON response with various scenarios."""
        test_cases = [
            # (error, expected_status, expected_code, expected_title, description, extras_check, request_id_check, url_check)
            (
                InvalidArgumentError("Bad input"),
                400,
                "INVALID_ARGUMENT",
                "InvalidArgumentError",
                "basic_error",
                None,
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
            (
                NotFoundError("Not found"),
                404,
                "NOT_FOUND",
                "NotFoundError",
                "not_found_error",
                None,
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
            (
                OperationConflictError("Conflict"),
                409,
                "OPERATION_CONFLICT",
                "OperationConflictError",
                "conflict_error",
                None,
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
            (
                InternalServerError("Server error"),
                500,
                "INTERNAL_ERROR",
                "InternalServerError",
                "internal_error",
                None,
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
            (
                DependencyTimeoutError("Timeout"),
                504,
                "DEPENDENCY_TIMEOUT",
                "DependencyTimeoutError",
                "timeout_error",
                None,
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
            (
                NotFoundError(
                    "Model not found",
                    extras={"model_name": "test-model", "available_models": []},
                ),
                404,
                "NOT_FOUND",
                "NotFoundError",
                "error_with_extras",
                {"model_name": "test-model", "available_models": []},
                "test-request-id-123",
                "http://testserver/api/v1/models/load",
            ),
        ]

        for (
            error,
            expected_status,
            expected_code,
            expected_title,
            description,
            extras_check,
            expected_request_id,
            expected_url,
        ) in test_cases:
            with self.subTest(scenario=description):
                response = _problem_response(self.mock_request, error)
                body = json.loads(response.body)

                # Check response structure
                self.assertEqual(expected_status, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)

                # Check response body
                self.assertEqual(expected_title, body["title"])
                self.assertEqual(expected_status, body["status"])
                self.assertEqual(expected_code, body["code"])
                self.assertEqual(str(error), body["detail"])
                self.assertEqual(expected_url, body["instance"])
                self.assertEqual(expected_request_id, body["request_id"])

                # Check extras if present
                if extras_check is not None:
                    self.assertIsInstance(body["extras"], dict)
                    self.assertEqual(extras_check, body["extras"])

        # Test missing request_id scenario separately
        with self.subTest(scenario="missing_request_id"):
            mock_request = MagicMock(spec=Request)
            mock_request.url = "http://testserver/test"
            del mock_request.state.request_id

            error = InternalServerError("Something went wrong")
            response = _problem_response(mock_request, error)
            body = json.loads(response.body)

            self.assertIsNone(body["request_id"])

    def test_validation_error_handler(self):
        """Test validation_error_handler converts RequestValidationError to Problem JSON."""
        test_cases = [
            # (errors, description, expected_error_count)
            (
                [
                    {
                        "loc": ("body", "name"),
                        "msg": "field required",
                        "type": "value_error.missing",
                    },
                    {
                        "loc": ("body", "maxBatchSize"),
                        "msg": "value is not a valid integer",
                        "type": "type_error.integer",
                    },
                ],
                "multiple_validation_errors",
                2,
            ),
            (
                [
                    {
                        "loc": ("body", "name"),
                        "msg": "field required",
                        "type": "value_error.missing",
                    },
                ],
                "single_validation_error",
                1,
            ),
        ]

        for errors, description, expected_count in test_cases:
            with self.subTest(scenario=description):
                mock_validation_error = MagicMock(spec=RequestValidationError)
                mock_validation_error.errors.return_value = errors

                response = asyncio.run(
                    _validation_error_handler(self.mock_request, mock_validation_error)
                )

                # Check response structure
                self.assertEqual(400, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)

                body = json.loads(response.body)
                self.assertEqual(400, body["status"])
                self.assertEqual("INVALID_ARGUMENT", body["code"])
                self.assertIn("loc", body["detail"])
                self.assertIn("msg", body["detail"])

                # Detail should be a JSON string of error messages
                error_messages = json.loads(body["detail"])
                self.assertIsInstance(error_messages, list)
                self.assertEqual(expected_count, len(error_messages))

                # Verify first error structure
                self.assertEqual(list(errors[0]["loc"]), error_messages[0]["loc"])
                self.assertEqual(errors[0]["msg"], error_messages[0]["msg"])
                self.assertEqual(errors[0]["type"], error_messages[0]["type"])

    def test_map_service_errors_to_http_errors(self):
        """Test mapping of service errors to HTTP errors."""
        test_cases = [
            # (service_error, expected_http_error_class, expected_status, description)
            (
                service_errors.ModelDownloadFailedError("Download failed"),
                InvalidArgumentError,
                400,
                "model_download_failed",
            ),
            (
                service_errors.ModelOperationInProgressError("Operation in progress"),
                OperationConflictError,
                409,
                "model_operation_in_progress",
            ),
            (
                service_errors.TritonCommunicationError("Triton error"),
                http_errors.DependencyBadGatewayError,
                502,
                "triton_communication_error",
            ),
            (
                service_errors.InternalServerError("Internal error"),
                InternalServerError,
                500,
                "internal_server_error",
            ),
        ]

        for (
            service_error,
            expected_http_error_class,
            expected_status,
            description,
        ) in test_cases:
            with self.subTest(scenario=description):
                http_error = _map_service_errors_to_http_errors(service_error)
                self.assertIsInstance(http_error, expected_http_error_class)
                self.assertEqual(expected_status, http_error.http_status)
                self.assertEqual(service_error.message, str(http_error))

        # Test unknown service error
        with self.subTest(scenario="unknown_service_error"):
            # Create a custom service error that's not in the mapping
            class UnknownServiceError(service_errors.ServiceError):
                pass

            unknown_error = UnknownServiceError("Unknown error")
            http_error = _map_service_errors_to_http_errors(unknown_error)

            self.assertIsInstance(http_error, InternalServerError)
            self.assertEqual(500, http_error.http_status)
            self.assertEqual("Unknown error", str(http_error))

    def test_service_error_handler(self):
        """Test service_error_handler maps service errors to HTTP errors."""
        test_cases = [
            # (service_error, expected_status, expected_code, description)
            (
                service_errors.ModelDownloadFailedError("Download failed"),
                400,
                "INVALID_ARGUMENT",
                "model_download_failed",
            ),
            (
                service_errors.ModelOperationInProgressError("Operation in progress"),
                409,
                "OPERATION_CONFLICT",
                "model_operation_in_progress",
            ),
            (
                service_errors.TritonCommunicationError("Triton error"),
                502,
                "DEPENDENCY_BAD_GATEWAY",
                "triton_communication_error",
            ),
            (
                service_errors.InternalServerError("Database connection failed"),
                500,
                "INTERNAL_ERROR",
                "internal_server_error",
            ),
        ]

        for service_error, expected_status, expected_code, description in test_cases:
            with self.subTest(scenario=description):
                response = asyncio.run(
                    _service_error_handler(self.mock_request, service_error)
                )

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)

                body = json.loads(response.body)
                self.assertEqual(expected_status, body["status"])
                self.assertEqual(expected_code, body["code"])
                self.assertEqual(service_error.message, body["detail"])

    def test_app_error_handler(self):
        """Test app_error_handler handles AppError and its subclasses."""
        test_cases = [
            (InvalidArgumentError("Invalid input"), 400),
            (NotFoundError("Resource not found"), 404),
            (InternalServerError("Internal error"), 500),
        ]

        for error, expected_status in test_cases:
            with self.subTest(error_type=type(error).__name__):
                response = asyncio.run(_app_error_handler(self.mock_request, error))

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)
                body = json.loads(response.body)
                self.assertEqual(expected_status, body["status"])

    def test_catch_all_handler(self):
        """Test catch_all_handler converts generic exceptions to InternalServerError."""
        test_cases = [
            # (exception, description)
            (ValueError("Value error"), "value_error"),
            (KeyError("Key error"), "key_error"),
            (RuntimeError("Runtime error"), "runtime_error"),
            (AttributeError("Attribute error"), "attribute_error"),
        ]

        for exc, description in test_cases:
            with self.subTest(scenario=description):
                response = asyncio.run(_catch_all_handler(self.mock_request, exc))

                self.assertEqual(500, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)

                body = json.loads(response.body)
                self.assertEqual(500, body["status"])
                self.assertEqual("INTERNAL_ERROR", body["code"])
                self.assertEqual("An unexpected error occurred.", body["detail"])

    def test_normalize_validation_errors_single_error(self):
        """Test _normalize_validation_errors with single validation error."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {
                "loc": ("body", "name"),
                "msg": "field required",
                "type": "value_error.missing",
                "input": None,
            },
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertEqual(1, result["count"])
        self.assertEqual(1, len(result["errors"]))
        self.assertEqual(("body", "name"), result["errors"][0]["location"])
        self.assertEqual("name", result["errors"][0]["field"])
        self.assertEqual("field required", result["errors"][0]["message"])
        self.assertEqual("value_error.missing", result["errors"][0]["type"])

    def test_normalize_validation_errors_multiple_errors(self):
        """Test _normalize_validation_errors with multiple validation errors."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {
                "loc": ("body", "name"),
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ("body", "maxBatchSize"),
                "msg": "value is not a valid integer",
                "type": "type_error.integer",
            },
            {
                "loc": ("body", "sources"),
                "msg": "ensure this value has at least 1 items",
                "type": "value_error.list.min_items",
            },
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertEqual(3, result["count"])
        self.assertEqual(3, len(result["errors"]))
        self.assertIn("name", result["field_errors"])
        self.assertIn("maxBatchSize", result["field_errors"])
        self.assertIn("sources", result["field_errors"])

    def test_normalize_validation_errors_strips_location_prefixes(self):
        """Test _normalize_validation_errors strips body/query/path prefixes from field paths."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {
                "loc": ("body", "model", "name"),
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ("query", "limit"),
                "msg": "value is not a valid integer",
                "type": "type_error.integer",
            },
            {
                "loc": ("path", "model_id"),
                "msg": "value is not a valid uuid",
                "type": "type_error.uuid",
            },
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertEqual("model.name", result["errors"][0]["field"])
        self.assertEqual("limit", result["errors"][1]["field"])
        self.assertEqual("model_id", result["errors"][2]["field"])

    def test_normalize_validation_errors_field_errors_mapping(self):
        """Test _normalize_validation_errors creates field_errors mapping correctly."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {
                "loc": ("body", "name"),
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ("body", "name"),
                "msg": "ensure this value has at most 100 characters",
                "type": "value_error.any_str.max_length",
            },
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertIn("name", result["field_errors"])
        self.assertEqual(2, len(result["field_errors"]["name"]))
        self.assertIn("field required", result["field_errors"]["name"])
        self.assertIn(
            "ensure this value has at most 100 characters",
            result["field_errors"]["name"],
        )

    def test_normalize_validation_errors_excludes_none_values(self):
        """Test _normalize_validation_errors excludes None values from error entries."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {
                "loc": ("body", "name"),
                "msg": "field required",
                "type": "value_error.missing",
            },
        ]

        result = _normalize_validation_errors(mock_validation_error)

        error_entry = result["errors"][0]
        # Should not contain 'input' or 'ctx' keys if they were None
        self.assertIn("location", error_entry)
        self.assertIn("field", error_entry)
        self.assertIn("message", error_entry)
        self.assertIn("type", error_entry)
