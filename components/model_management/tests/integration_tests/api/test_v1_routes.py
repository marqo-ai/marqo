from unittest import TestCase

from fastapi.testclient import TestClient

from model_management.main import app


class TestV1RoutesIntegration(TestCase):
    """Integration tests for v1 API routes against the full application."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        cls.client = TestClient(app)

    def test_healthz_endpoint(self):
        """Test healthz endpoint returns correct response and handles request IDs."""
        # Test with auto-generated request ID
        response = self.client.get("/v1/healthz")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.json())
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(32, len(response.headers["x-request-id"]))

        # Test with custom request ID
        custom_request_id = "test-request-id-12345"
        response = self.client.get(
            "/v1/healthz", headers={"x-request-id": custom_request_id}
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(custom_request_id, response.headers["x-request-id"])

    def test_load_model_validation_errors(self):
        """Test that load model endpoint validates payloads and returns appropriate errors."""
        test_cases = [
            ({}, "empty payload"),
            ({"tritonModelProperties": {}}, "empty model properties"),
            (
                {"tritonModelProperties": {"name": "test"}},
                "missing sources and input/output",
            ),
            (
                {"tritonModelProperties": {"name": "test-model"}},
                "missing required fields",
            ),
            (
                {
                    "tritonModelProperties": {
                        "name": "test-model",
                        "sources": ["invalid-source.txt"],
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
                },
                "invalid source file extension",
            ),
        ]

        for payload, description in test_cases:
            with self.subTest(description=description):
                response = self.client.post("/v1/models/load", json=payload)
                self.assertEqual(400, response.status_code)
                self.assertEqual(
                    "application/problem+json", response.headers["content-type"]
                )

                body = response.json()
                self.assertIn("title", body)
                self.assertIn("status", body)
                self.assertIn("detail", body)
                self.assertIn("InvalidArgumentError", body["title"])
                self.assertEqual(400, body["status"])

    def test_error_responses_follow_rfc7807_with_request_ids(self):
        """Test that error responses follow RFC 7807 Problem+JSON format and include request IDs."""
        # Test with auto-generated request ID
        response = self.client.post("/v1/models/load", json={})
        self.assertEqual(400, response.status_code)
        self.assertEqual("application/problem+json", response.headers["content-type"])

        body = response.json()
        # RFC 7807 required fields
        for field in ["title", "status", "detail", "instance"]:
            self.assertIn(field, body)
        # Additional fields
        for field in ["code", "request_id"]:
            self.assertIn(field, body)

        # Test with custom request ID
        custom_request_id = "trace-test-12345"
        response = self.client.post(
            "/v1/models/load", json={}, headers={"x-request-id": custom_request_id}
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual(custom_request_id, response.headers["x-request-id"])
        body = response.json()
        self.assertEqual(custom_request_id, body["request_id"])

    def test_openapi_documentation(self):
        """Test that OpenAPI documentation is accessible and contains correct metadata."""
        # Test OpenAPI schema
        response = self.client.get("/openapi.json")
        self.assertEqual(200, response.status_code)

        schema = response.json()
        self.assertIn("info", schema)
        self.assertIn("version", schema["info"])
        self.assertIn("title", schema["info"])
        self.assertEqual("Marqo Model Management Container", schema["info"]["title"])

        # Test docs UI is accessible
        response = self.client.get("/docs")
        self.assertEqual(200, response.status_code)

    def test_max_batch_size_validation(self):
        """Test that max_batch_size validation rejects invalid values."""
        test_cases = [
            (0, "zero batch size"),
            (-1, "negative batch size"),
            (129, "batch size above limit"),
        ]

        for batch_size, description in test_cases:
            with self.subTest(description=description):
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
                self.assertEqual(400, response.status_code)

                body = response.json()
                self.assertIn("InvalidArgumentError", body["title"])
                self.assertIn("detail", body)
