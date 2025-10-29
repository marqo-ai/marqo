from unittest import TestCase

from fastapi.testclient import TestClient

from model_management.main import app


class TestV1RoutesIntegration(TestCase):
    """Integration tests for v1 API routes against the full application."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        cls.client = TestClient(app)

    def test_healthz_endpoint_returns_ok(self):
        """Test that the healthz endpoint returns 200 OK."""
        response = self.client.get("/v1/healthz")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.json())

    def test_healthz_endpoint_structure(self):
        """Test that the healthz endpoint returns correct response structure."""
        response = self.client.get("/v1/healthz")

        body = response.json()
        self.assertIsInstance(body, dict)
        self.assertIn("status", body)
        self.assertEqual("ok", body["status"])

    def test_request_id_middleware_generates_id(self):
        """Test that the request ID middleware generates and returns a request ID."""
        response = self.client.get("/v1/healthz")

        self.assertEqual(200, response.status_code)
        self.assertIn("x-request-id", response.headers)
        self.assertIsNotNone(response.headers["x-request-id"])
        self.assertEqual(32, len(response.headers["x-request-id"]))  # UUID hex format

    def test_request_id_middleware_uses_provided_id(self):
        """Test that the request ID middleware uses the provided request ID."""
        custom_request_id = "test-request-id-12345"
        response = self.client.get(
            "/v1/healthz", headers={"x-request-id": custom_request_id}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(custom_request_id, response.headers["x-request-id"])

    def test_load_model_with_invalid_payload_returns_400(self):
        """Test that loading a model with invalid payload returns 400 Bad Request."""
        invalid_payload = {
            "tritonModelProperties": {
                "name": "test-model",
                # Missing required fields
            }
        }

        response = self.client.post("/v1/models/load", json=invalid_payload)

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])
        body = response.json()
        self.assertIn("title", body)
        self.assertIn("status", body)
        self.assertIn("detail", body)
        self.assertEqual(400, body["status"])

    def test_load_model_with_missing_fields_returns_validation_error(self):
        """Test that loading a model with missing required fields returns validation error."""
        test_cases = [
            ({}, "empty payload"),
            ({"tritonModelProperties": {}}, "empty model properties"),
            (
                {"tritonModelProperties": {"name": "test"}},
                "missing sources and input/output",
            ),
        ]

        for payload, description in test_cases:
            with self.subTest(payload=payload, description=description):
                response = self.client.post("/v1/models/load", json=payload)

                self.assertEqual(400, response.status_code)
                body = response.json()
                self.assertIn("title", body)
                self.assertIn("InvalidArgumentError", body["title"])

    def test_load_model_with_invalid_sources_returns_validation_error(self):
        """Test that loading a model with invalid sources returns validation error."""
        invalid_payload = {
            "tritonModelProperties": {
                "name": "test-model",
                "sources": ["invalid-source.txt"],  # Must be model.onnx
                "input": [
                    {"name": "input", "dims": [1, 224, 224, 3], "dataType": "TYPE_FP32"}
                ],
                "output": [
                    {"name": "output", "dims": [1, 1000], "dataType": "TYPE_FP32"}
                ],
            }
        }

        response = self.client.post("/v1/models/load", json=invalid_payload)

        self.assertEqual(400, response.status_code)
        body = response.json()
        self.assertIn("detail", body)

    def test_unload_model_returns_success(self):
        """Test that unloading a model returns success (idempotent)."""
        model_name = "test-unload-model"
        response = self.client.post(f"/v1/models/{model_name}/unload")

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertIn("message", body)
        self.assertIn(model_name, body["message"])
        self.assertIn("unloaded successfully", body["message"])

    def test_unload_model_with_remove_files_parameter(self):
        """Test that unload endpoint accepts remove-files query parameter."""
        model_name = "test-unload-with-remove"
        response = self.client.post(
            f"/v1/models/{model_name}/unload", params={"remove-files": "true"}
        )

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertIn("message", body)

    def test_unload_model_with_special_characters_in_name(self):
        """Test that unload endpoint handles special characters in model name."""
        test_cases = [
            "model-with-dashes",
            "model_with_underscores",
            "model123",
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                response = self.client.post(f"/v1/models/{model_name}/unload")

                self.assertEqual(200, response.status_code)
                body = response.json()
                self.assertIn(model_name, body["message"])

    def test_api_returns_problem_json_for_errors(self):
        """Test that API returns RFC 7807 Problem+JSON format for errors."""
        response = self.client.post("/v1/models/load", json={})

        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        # RFC 7807 required fields
        self.assertIn("title", body)
        self.assertIn("status", body)
        self.assertIn("detail", body)
        self.assertIn("instance", body)

        # Additional fields
        self.assertIn("code", body)
        self.assertIn("request_id", body)

    def test_api_version_in_openapi_schema(self):
        """Test that the API version is included in the OpenAPI schema."""
        response = self.client.get("/openapi.json")

        self.assertEqual(200, response.status_code)
        schema = response.json()
        self.assertIn("info", schema)
        self.assertIn("version", schema["info"])
        self.assertIn("title", schema["info"])
        self.assertEqual("Marqo Model Management Container", schema["info"]["title"])

    def test_openapi_docs_accessible(self):
        """Test that OpenAPI documentation is accessible."""
        response = self.client.get("/docs")

        self.assertEqual(200, response.status_code)

    def test_multiple_requests_have_different_request_ids(self):
        """Test that multiple requests generate different request IDs."""
        response1 = self.client.get("/v1/healthz")
        response2 = self.client.get("/v1/healthz")

        request_id_1 = response1.headers["x-request-id"]
        request_id_2 = response2.headers["x-request-id"]

        self.assertNotEqual(request_id_1, request_id_2)

    def test_unload_model_endpoint_url_structure(self):
        """Test that the unload model endpoint URL structure is correct."""
        model_name = "test-model"

        response = self.client.post(f"/v1/models/{model_name}/unload")
        self.assertEqual(200, response.status_code)

    def test_error_response_includes_request_id(self):
        """Test that error responses include the request ID for tracing."""
        custom_request_id = "trace-test-12345"
        response = self.client.post(
            "/v1/models/load", json={}, headers={"x-request-id": custom_request_id}
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual(custom_request_id, response.headers["x-request-id"])

        body = response.json()
        self.assertEqual(custom_request_id, body["request_id"])

    def test_max_batch_size_validation(self):
        """Test that max_batch_size validation works correctly."""
        test_cases = [
            (0, 400, "zero batch size should fail validation"),
            (-1, 400, "negative batch size should fail validation"),
            (129, 400, "batch size above limit should fail validation"),
        ]

        for batch_size, expected_status, description in test_cases:
            with self.subTest(
                batch_size=batch_size,
                expected_status=expected_status,
                description=description,
            ):
                payload = {
                    "tritonModelProperties": {
                        "name": "test-model",
                        "maxBatchSize": batch_size,
                        "sources": ["s3://bucket/model.onnx"],
                        "input": [
                            {
                                "name": "input",
                                "dims": [1, 224, 224, 3],
                                "dataType": "TYPE_FP32",
                            }
                        ],
                        "output": [
                            {
                                "name": "output",
                                "dims": [1, 1000],
                                "dataType": "TYPE_FP32",
                            }
                        ],
                    }
                }

                response = self.client.post("/v1/models/load", json=payload)
                self.assertEqual(expected_status, response.status_code)

                # Verify it's a validation error
                body = response.json()
                self.assertIn("InvalidArgumentError", body["title"])
                self.assertIn("detail", body)
