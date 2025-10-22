"""Integration tests for the main FastAPI application.

These tests verify that the API endpoints in main.py work correctly with real requests.
They test the full request/response cycle including serialization, error handling, and model management.
"""

import msgpack
import msgpack_numpy
import numpy as np
from fastapi.testclient import TestClient

from inference_orchestrator.main import app
from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from tests.integration_tests.test_case import InferenceTestCase

msgpack_numpy.patch()


class TestMainAPIRootEndpoints(InferenceTestCase):
    """Test basic root and health check endpoints."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = TestClient(app)

    def test_root_endpoint_returns_welcome_message(self):
        """Test that the root endpoint returns welcome message and version."""
        response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("Welcome to Marqo Inference", data["message"])
        self.assertIn("version", data)

    def test_healthz_endpoint_returns_ok(self):
        """Test that the liveness check endpoint returns ok status."""
        response = self.client.get("/healthz")

        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual({"status": "ok"}, data)


class TestMainAPIVectoriseEndpoint(InferenceTestCase):
    """Test the /vectorise endpoint with various scenarios."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = TestClient(app)
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    def test_vectorise_with_valid_text_request(self):
        """Test vectorise endpoint with a valid text inference request."""
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        # Create inference request
        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["hello world", "test sentence"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Serialize request to msgpack
        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        # Send request
        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        # Verify response
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/msgpack", response.headers["content-type"])

        # Deserialize response
        result = msgpack.unpackb(response.content, raw=False)
        self.assertIn("result", result)
        self.assertEqual(2, len(result["result"]))

        # Verify first embedding
        first_result = result["result"][0]
        self.assertEqual(1, len(first_result))
        self.assertEqual("hello world", first_result[0][0])
        embedding = np.array(first_result[0][1])
        self.assertEqual(384, len(embedding))
        # Verify normalization
        self.validate_norm(embedding, normalize=True)

    def test_vectorise_with_single_content(self):
        """Test vectorise endpoint with a single content item."""
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["single text"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(200, response.status_code)
        result = msgpack.unpackb(response.content, raw=False)
        self.assertEqual(1, len(result["result"]))

    def test_vectorise_without_normalization(self):
        """Test vectorise endpoint with normalization disabled."""
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=False,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(200, response.status_code)
        result = msgpack.unpackb(response.content, raw=False)
        embedding = np.array(result["result"][0][0][1])
        # Verify it's NOT normalized
        self.validate_norm(embedding, normalize=False)

    def test_vectorise_with_invalid_content_type_returns_422(self):
        """Test that using wrong content type returns 422 error.

        FastAPI processes the request body before checking content-type,
        so sending JSON with application/json content type returns 422
        validation error rather than 415 unsupported media type.
        """
        response = self.client.post(
            "/vectorise",
            content=b'{"test": "data"}',
            headers={"Content-Type": "application/json"},
        )

        # FastAPI returns 422 (validation error) for wrong content type
        self.assertEqual(422, response.status_code)

    def test_vectorise_with_invalid_msgpack_returns_400(self):
        """Test that invalid msgpack data returns 400 error."""
        response = self.client.post(
            "/vectorise",
            content=b"invalid msgpack data",
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(400, response.status_code)
        error_data = response.json()
        self.assertIn("Invalid MessagePack format", error_data["detail"])

    def test_vectorise_with_invalid_request_schema_returns_422(self):
        """Test that invalid request schema returns 422 error."""
        # Create invalid request (missing required fields)
        invalid_request = {"modality": "TEXT"}  # Missing contents and other fields

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)

    def test_vectorise_with_empty_contents_returns_422(self):
        """Test that empty contents list returns 422 error."""
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request_dict = {
            "modality": "language",
            "contents": [],  # Empty list - should fail validation
            "embeddingModelConfig": {
                "modelName": model_name,
                "modelProperties": model_properties,
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(inference_request_dict, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)

    def test_vectorise_with_different_model_loads_new_model(self):
        """Test that using a different model loads it correctly."""
        # First request with one model
        model_name_1 = "hf/all-MiniLM-L6-v2"
        model_properties_1 = self.get_model_properties_from_registry(model_name_1)

        inference_request_1 = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test 1"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name_1,
                modelProperties=model_properties_1,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data_1 = msgpack.packb(
            inference_request_1.model_dump(by_alias=True), use_bin_type=True
        )

        response_1 = self.client.post(
            "/vectorise",
            content=request_data_1,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(200, response_1.status_code)

        # Second request with a different model
        model_name_2 = "hf/e5-base-v2"
        model_properties_2 = self.get_model_properties_from_registry(model_name_2)

        inference_request_2 = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test 2"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name_2,
                modelProperties=model_properties_2,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data_2 = msgpack.packb(
            inference_request_2.model_dump(by_alias=True), use_bin_type=True
        )

        response_2 = self.client.post(
            "/vectorise",
            content=request_data_2,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(200, response_2.status_code)

        # Verify different embedding dimensions
        result_1 = msgpack.unpackb(response_1.content, raw=False)
        result_2 = msgpack.unpackb(response_2.content, raw=False)

        embedding_1 = np.array(result_1["result"][0][0][1])
        embedding_2 = np.array(result_2["result"][0][0][1])

        self.assertEqual(384, len(embedding_1))  # all-MiniLM-L6-v2
        self.assertEqual(768, len(embedding_2))  # e5-base-v2


class TestMainAPIModelsEndpoints(InferenceTestCase):
    """Test the /models endpoints for model management."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = TestClient(app)
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    def test_get_loaded_models_when_empty(self):
        """Test getting loaded models when no models are loaded."""
        self.eject_all_models()

        response = self.client.get("/models")

        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIn("models", data)
        self.assertEqual([], data["models"])

    def test_get_loaded_models_after_loading(self):
        """Test getting loaded models after loading a model via vectorise."""
        # Load a model by making a vectorise request
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        vectorise_response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )
        self.assertEqual(200, vectorise_response.status_code)

        # Now check loaded models
        response = self.client.get("/models")

        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIn("models", data)
        self.assertGreater(len(data["models"]), 0)

        # Verify model is in the list
        model_names = [model["modelName"] for model in data["models"]]
        self.assertTrue(any(model_name in name for name in model_names))

    def test_get_loaded_models_with_detailed_parameter(self):
        """Test getting loaded models with detailed=true shows model properties."""
        # Load a model first
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        # Get models with detailed=false (default)
        response_simple = self.client.get("/models?detailed=false")
        data_simple = response_simple.json()

        # Get models with detailed=true
        response_detailed = self.client.get("/models?detailed=true")
        data_detailed = response_detailed.json()

        self.assertEqual(200, response_simple.status_code)
        self.assertEqual(200, response_detailed.status_code)

        # Both should have models
        self.assertGreater(len(data_simple["models"]), 0)
        self.assertGreater(len(data_detailed["models"]), 0)

        # Detailed should include modelProperties
        if len(data_detailed["models"]) > 0:
            first_model_detailed = data_detailed["models"][0]
            self.assertIn("modelProperties", first_model_detailed)
            # Should be JSON string or object, not "<omitted>"
            self.assertNotEqual("<omitted>", first_model_detailed["modelProperties"])

    def test_eject_model_removes_loaded_model(self):
        """Test that ejecting a model removes it from loaded models."""
        # Load a model
        model_name = "hf/all-MiniLM-L6-v2"
        model_properties = self.get_model_properties_from_registry(model_name)

        inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName=model_name,
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        request_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        # Get loaded models to find the exact model cache key
        models_response = self.client.get("/models")
        models_data = models_response.json()
        model_cache_key = None
        for model in models_data["models"]:
            if model_name in model["modelName"]:
                model_cache_key = model["modelName"]
                break

        self.assertIsNotNone(model_cache_key, "Model should be loaded")

        # Eject the model
        eject_response = self.client.delete(f"/models?model_name={model_cache_key}")

        self.assertEqual(200, eject_response.status_code)
        eject_data = eject_response.json()
        self.assertIn("result", eject_data)
        self.assertEqual("success", eject_data["result"])

        # Verify model is no longer in loaded models
        models_response_after = self.client.get("/models")
        models_data_after = models_response_after.json()

        model_names_after = [
            model["modelName"] for model in models_data_after["models"]
        ]
        self.assertNotIn(model_cache_key, model_names_after)


class TestMainAPIErrorHandling(InferenceTestCase):
    """Test error handling in the API.

    These tests specifically cover error handling in the vectorise endpoint,
    focusing on lines 100-114 of main.py which handle msgpack parsing and
    validation errors.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = TestClient(app)
        cls.eject_all_models()

    def test_vectorise_with_malformed_msgpack_returns_400(self):
        """Test that malformed msgpack data returns 400 with ExtraData error.

        This tests line 102-106 in main.py for msgpack.ExtraData exception.
        """
        # Create valid msgpack data, then append extra bytes to make it malformed
        valid_data = msgpack.packb({"test": "data"}, use_bin_type=True)
        malformed_data = valid_data + b"\x00\x01\x02"  # Extra bytes after valid msgpack

        response = self.client.post(
            "/vectorise",
            content=malformed_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(400, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)
        self.assertIn("Invalid MessagePack format", error_data["detail"])

    def test_vectorise_with_corrupted_msgpack_returns_400(self):
        """Test that corrupted msgpack data returns 400 with UnpackException.

        This tests line 102-106 in main.py for msgpack.UnpackException.
        """
        # Create completely invalid msgpack data
        corrupted_data = b"\xff\xfe\xfd\xfc\xfb"  # Invalid msgpack bytes

        response = self.client.post(
            "/vectorise",
            content=corrupted_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(400, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)
        self.assertIn("Invalid MessagePack format", error_data["detail"])

    def test_vectorise_with_incomplete_msgpack_returns_500(self):
        """Test that incomplete msgpack data returns 500.

        This tests line 111-114 in main.py - incomplete msgpack triggers
        ValueError which is caught by the generic Exception handler.
        """
        # Create truncated msgpack data (incomplete)
        incomplete_data = b"\x81"  # Start of a map but truncated

        response = self.client.post(
            "/vectorise",
            content=incomplete_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(500, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)

    def test_vectorise_with_missing_required_field_returns_422(self):
        """Test that missing required fields returns 422 ValidationError.

        This tests line 107-110 in main.py for Pydantic ValidationError.
        """
        # Missing required 'contents' field
        invalid_request = {
            "modality": "language",
            "embeddingModelConfig": {
                "modelName": "test-model",
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)
        # ValidationError detail should mention missing field
        self.assertIn("Field required", error_data["detail"])

    def test_vectorise_with_wrong_field_type_returns_422(self):
        """Test that wrong field types return 422 ValidationError.

        This tests line 107-110 in main.py for Pydantic ValidationError.
        """
        # 'contents' should be a list, not a string
        invalid_request = {
            "modality": "language",
            "contents": "should be a list not string",
            "embeddingModelConfig": {
                "modelName": "test-model",
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)

    def test_vectorise_with_invalid_modality_returns_422(self):
        """Test that invalid modality enum value returns 422 ValidationError.

        This tests line 107-110 in main.py for Pydantic ValidationError.
        """
        invalid_request = {
            "modality": "invalid_modality",  # Not a valid Modality enum
            "contents": ["test"],
            "embeddingModelConfig": {
                "modelName": "test-model",
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)

    def test_vectorise_with_invalid_embedding_config_returns_422(self):
        """Test that invalid embeddingModelConfig returns 422 ValidationError.

        This tests line 107-110 in main.py for Pydantic ValidationError.
        """
        # Missing required 'modelName' in embeddingModelConfig
        invalid_request = {
            "modality": "language",
            "contents": ["test"],
            "embeddingModelConfig": {
                "normalizeEmbeddings": True,
                # Missing modelName
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)
        self.assertIn("Field required", error_data["detail"])

    def test_vectorise_with_invalid_preprocessing_config_returns_422(self):
        """Test that invalid preprocessingConfig returns 422 ValidationError.

        This tests line 107-110 in main.py for Pydantic ValidationError.
        """
        # Missing required 'modality' in preprocessingConfig
        invalid_request = {
            "modality": "language",
            "contents": ["test"],
            "embeddingModelConfig": {
                "modelName": "test-model",
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {},  # Missing modality
        }

        request_data = msgpack.packb(invalid_request, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        self.assertEqual(422, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)

    def test_vectorise_with_invalid_model_properties_returns_400(self):
        """Test that invalid model properties return 400 error.

        This tests line 124-129 in main.py for ServiceError exception.
        """
        # Create request with invalid model properties
        invalid_model_properties = {
            "type": "invalid_type",
            "dimensions": 512,
        }

        inference_request_dict = {
            "modality": "language",
            "contents": ["test"],
            "embeddingModelConfig": {
                "modelName": "test-model",
                "modelProperties": invalid_model_properties,
                "normalizeEmbeddings": True,
            },
            "preprocessingConfig": {"modality": "language"},
        }

        request_data = msgpack.packb(inference_request_dict, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            content=request_data,
            headers={"Content-Type": "application/msgpack"},
        )

        # Should return 400 for invalid model properties (bad request)
        self.assertEqual(400, response.status_code)
        error_data = response.json()
        self.assertIn("detail", error_data)
        self.assertIn("An error occurred during vectorisation", error_data["detail"])
