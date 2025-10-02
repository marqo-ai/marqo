import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.requests import Request

from marqo_model_management_container.api.exception_handlers import (
    register_exception_handlers,
    validation_error_handler,
    service_error_handler,
    app_error_handler,
    catch_all_handler,
    map_service_errors_to_http_errors,
    _problem_response,
    _normalize_validation_errors,
)
from marqo_model_management_container.errors.base import AppError
from marqo_model_management_container.errors.http_errors import (
    InternalServerError,
    InvalidArgumentError,
    NotFoundError,
    OperationConflictError,
    DependencyTimeoutError,
)
import marqo_model_management_container.errors.http_errors as http_errors
import marqo_model_management_container.services.errors as service_errors


class TestExceptionHandlers(TestCase):
    """Test class for exception handlers in marqo_model_management_container.api.exception_handlers."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.mock_request = MagicMock(spec=Request)
        self.mock_request.url = "http://testserver/api/v1/models/load"
        self.mock_request.state.request_id = "test-request-id-123"

    def test_register_exception_handlers(self):
        """Test that all exception handlers are registered correctly."""
        with patch.object(self.app, 'add_exception_handler') as mock_add_handler:
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

            self.assertIn(validation_error_handler, handlers)
            self.assertIn(service_error_handler, handlers)
            self.assertIn(app_error_handler, handlers)
            self.assertIn(catch_all_handler, handlers)

    def test_problem_response_basic_app_error(self):
        """Test _problem_response converts basic AppError to Problem JSON response."""
        error = InvalidArgumentError("Invalid model name")

        response = _problem_response(self.mock_request, error)

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.media_type)

        body = json.loads(response.body)
        self.assertEqual("InvalidArgumentError", body["title"])
        self.assertEqual(400, body["status"])
        self.assertEqual("INVALID_ARGUMENT", body["code"])
        self.assertEqual("Invalid model name", body["detail"])
        self.assertEqual("http://testserver/api/v1/models/load", body["instance"])
        self.assertEqual("test-request-id-123", body["request_id"])

    def test_problem_response_with_extras(self):
        """Test _problem_response includes extras field when present."""
        error = NotFoundError("Model not found", extras={"model_name": "test-model", "available_models": []})

        response = _problem_response(self.mock_request, error)

        body = json.loads(response.body)
        self.assertEqual(404, body["status"])
        self.assertEqual("NOT_FOUND", body["code"])
        self.assertIsInstance(body["extras"], dict)
        self.assertEqual("test-model", body["extras"]["model_name"])
        self.assertEqual([], body["extras"]["available_models"])

    def test_problem_response_without_request_id(self):
        """Test _problem_response handles missing request_id gracefully."""
        mock_request = MagicMock(spec=Request)
        mock_request.url = "http://testserver/test"
        # No request_id in state
        del mock_request.state.request_id

        error = InternalServerError("Something went wrong")
        response = _problem_response(mock_request, error)

        body = json.loads(response.body)
        self.assertIsNone(body["request_id"])

    def test_problem_response_different_error_types(self):
        """Test _problem_response handles different AppError subclasses correctly."""
        test_cases = [
            (InvalidArgumentError("Bad input"), 400, "INVALID_ARGUMENT", "InvalidArgumentError"),
            (NotFoundError("Not found"), 404, "NOT_FOUND", "NotFoundError"),
            (OperationConflictError("Conflict"), 409, "OPERATION_CONFLICT", "OperationConflictError"),
            (InternalServerError("Server error"), 500, "INTERNAL_ERROR", "InternalServerError"),
            (DependencyTimeoutError("Timeout"), 504, "DEPENDENCY_TIMEOUT", "DependencyTimeoutError"),
        ]

        for error, expected_status, expected_code, expected_title in test_cases:
            with self.subTest(error_type=type(error).__name__):
                response = _problem_response(self.mock_request, error)
                body = json.loads(response.body)

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual(expected_status, body["status"])
                self.assertEqual(expected_code, body["code"])
                self.assertEqual(expected_title, body["title"])

    @pytest.mark.anyio
    async def test_validation_error_handler(self):
        """Test validation_error_handler converts RequestValidationError to Problem JSON."""
        # Create a mock validation error
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing"},
            {"loc": ("body", "maxBatchSize"), "msg": "value is not a valid integer", "type": "type_error.integer"},
        ]

        response = await validation_error_handler(self.mock_request, mock_validation_error)

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.media_type)

        body = json.loads(response.body)
        self.assertEqual(400, body["status"])
        self.assertEqual("INVALID_ARGUMENT", body["code"])
        self.assertIn("loc", body["detail"])
        self.assertIn("msg", body["detail"])

    @pytest.mark.anyio
    async def test_validation_error_handler_formats_errors_as_json(self):
        """Test validation_error_handler formats validation errors as JSON string."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing"},
        ]

        response = await validation_error_handler(self.mock_request, mock_validation_error)
        body = json.loads(response.body)

        # Detail should be a JSON string of error messages
        error_messages = json.loads(body["detail"])
        self.assertIsInstance(error_messages, list)
        self.assertEqual(1, len(error_messages))
        self.assertEqual(("body", "name"), error_messages[0]["loc"])
        self.assertEqual("field required", error_messages[0]["msg"])
        self.assertEqual("value_error.missing", error_messages[0]["type"])

    def test_map_service_errors_to_http_errors(self):
        """Test mapping of service errors to HTTP errors."""
        test_cases = [
            (service_errors.ModelDownloadFailedError("Download failed"), InvalidArgumentError, 400),
            (service_errors.ModelOperationInProgressError("Operation in progress"), OperationConflictError, 409),
            (service_errors.TritonCommunicationError("Triton error"), http_errors.DependencyBadGatewayError, 502),
            (service_errors.InternalServerError("Internal error"), InternalServerError, 500),
        ]

        for service_error, expected_http_error_class, expected_status in test_cases:
            with self.subTest(service_error_type=type(service_error).__name__):
                http_error = map_service_errors_to_http_errors(service_error)
                self.assertIsInstance(http_error, expected_http_error_class)
                self.assertEqual(expected_status, http_error.http_status)
                self.assertEqual(service_error.message, str(http_error))

    def test_map_service_errors_to_http_errors_unknown_error(self):
        """Test that unknown service errors map to InternalServerError."""
        # Create a custom service error that's not in the mapping
        class UnknownServiceError(service_errors.ServiceError):
            pass

        unknown_error = UnknownServiceError("Unknown error")
        http_error = map_service_errors_to_http_errors(unknown_error)

        self.assertIsInstance(http_error, InternalServerError)
        self.assertEqual(500, http_error.http_status)
        self.assertEqual("Unknown error", str(http_error))

    @pytest.mark.anyio
    async def test_service_error_handler(self):
        """Test service_error_handler maps service errors to HTTP errors."""
        error = service_errors.InternalServerError("Database connection failed")

        response = await service_error_handler(self.mock_request, error)

        self.assertEqual(500, response.status_code)
        body = json.loads(response.body)
        self.assertEqual(500, body["status"])
        self.assertEqual("INTERNAL_ERROR", body["code"])
        self.assertEqual("Database connection failed", body["detail"])

    @pytest.mark.anyio
    async def test_service_error_handler_different_service_errors(self):
        """Test service_error_handler with different service error types."""
        test_cases = [
            (service_errors.ModelDownloadFailedError("Download failed"), 400, "INVALID_ARGUMENT"),
            (service_errors.ModelOperationInProgressError("Operation in progress"), 409, "OPERATION_CONFLICT"),
            (service_errors.TritonCommunicationError("Triton error"), 502, "DEPENDENCY_BAD_GATEWAY"),
            (service_errors.InternalServerError("Internal error"), 500, "INTERNAL_ERROR"),
        ]

        for service_error, expected_status, expected_code in test_cases:
            with self.subTest(service_error_type=type(service_error).__name__):
                response = await service_error_handler(self.mock_request, service_error)

                self.assertEqual(expected_status, response.status_code)
                body = json.loads(response.body)
                self.assertEqual(expected_status, body["status"])
                self.assertEqual(expected_code, body["code"])

    @pytest.mark.anyio
    async def test_app_error_handler(self):
        """Test app_error_handler handles AppError and its subclasses."""
        test_cases = [
            (InvalidArgumentError("Invalid input"), 400),
            (NotFoundError("Resource not found"), 404),
            (InternalServerError("Internal error"), 500),
        ]

        for error, expected_status in test_cases:
            with self.subTest(error_type=type(error).__name__):
                response = await app_error_handler(self.mock_request, error)

                self.assertEqual(expected_status, response.status_code)
                self.assertEqual("application/problem+json", response.media_type)
                body = json.loads(response.body)
                self.assertEqual(expected_status, body["status"])

    @pytest.mark.anyio
    async def test_catch_all_handler(self):
        """Test catch_all_handler converts generic exceptions to InternalServerError."""
        generic_error = ValueError("Unexpected error")

        response = await catch_all_handler(self.mock_request, generic_error)

        self.assertEqual(500, response.status_code)
        body = json.loads(response.body)
        self.assertEqual(500, body["status"])
        self.assertEqual("INTERNAL_ERROR", body["code"])
        self.assertEqual("An unexpected error occurred.", body["detail"])

    @pytest.mark.anyio
    async def test_catch_all_handler_with_different_exceptions(self):
        """Test catch_all_handler handles various exception types."""
        test_cases = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
            AttributeError("Attribute error"),
        ]

        for exc in test_cases:
            with self.subTest(exception_type=type(exc).__name__):
                response = await catch_all_handler(self.mock_request, exc)

                self.assertEqual(500, response.status_code)
                body = json.loads(response.body)
                self.assertEqual("INTERNAL_ERROR", body["code"])
                self.assertEqual("An unexpected error occurred.", body["detail"])

    def test_normalize_validation_errors_single_error(self):
        """Test _normalize_validation_errors with single validation error."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing", "input": None},
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
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing"},
            {"loc": ("body", "maxBatchSize"), "msg": "value is not a valid integer", "type": "type_error.integer"},
            {"loc": ("body", "sources"), "msg": "ensure this value has at least 1 items", "type": "value_error.list.min_items"},
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
            {"loc": ("body", "model", "name"), "msg": "field required", "type": "value_error.missing"},
            {"loc": ("query", "limit"), "msg": "value is not a valid integer", "type": "type_error.integer"},
            {"loc": ("path", "model_id"), "msg": "value is not a valid uuid", "type": "type_error.uuid"},
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertEqual("model.name", result["errors"][0]["field"])
        self.assertEqual("limit", result["errors"][1]["field"])
        self.assertEqual("model_id", result["errors"][2]["field"])

    def test_normalize_validation_errors_field_errors_mapping(self):
        """Test _normalize_validation_errors creates field_errors mapping correctly."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing"},
            {"loc": ("body", "name"), "msg": "ensure this value has at most 100 characters", "type": "value_error.any_str.max_length"},
        ]

        result = _normalize_validation_errors(mock_validation_error)

        self.assertIn("name", result["field_errors"])
        self.assertEqual(2, len(result["field_errors"]["name"]))
        self.assertIn("field required", result["field_errors"]["name"])
        self.assertIn("ensure this value has at most 100 characters", result["field_errors"]["name"])

    def test_normalize_validation_errors_excludes_none_values(self):
        """Test _normalize_validation_errors excludes None values from error entries."""
        mock_validation_error = MagicMock(spec=RequestValidationError)
        mock_validation_error.errors.return_value = [
            {"loc": ("body", "name"), "msg": "field required", "type": "value_error.missing"},
        ]

        result = _normalize_validation_errors(mock_validation_error)

        error_entry = result["errors"][0]
        # Should not contain 'input' or 'ctx' keys if they were None
        self.assertIn("location", error_entry)
        self.assertIn("field", error_entry)
        self.assertIn("message", error_entry)
        self.assertIn("type", error_entry)

    def test_error_response_includes_request_url(self):
        """Test that error responses include the request URL in the instance field."""
        test_urls = [
            "http://testserver/api/v1/models/load",
            "http://testserver/api/v1/models/test-model/unload",
            "http://localhost:8883/health",
        ]

        for url in test_urls:
            with self.subTest(url=url):
                mock_request = MagicMock(spec=Request)
                mock_request.url = url
                mock_request.state.request_id = "test-id"

                error = InvalidArgumentError("Test error")
                response = _problem_response(mock_request, error)
                body = json.loads(response.body)

                self.assertEqual(url, body["instance"])

    def test_error_response_content_type_is_problem_json(self):
        """Test that error responses use application/problem+json content type."""
        error = NotFoundError("Not found")
        response = _problem_response(self.mock_request, error)

        self.assertEqual("application/problem+json", response.media_type)
