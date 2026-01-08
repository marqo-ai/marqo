import json
from unittest import TestCase

from model_management.schemas.api_models import (
    LoadModelRequest,
    LoadModelResponse,
    UnloadModelResponse,
)
from model_management.schemas.triton_model_properties import TritonModelProperties
from pydantic import ValidationError


class TestLoadModelRequest(TestCase):
    """Test class for LoadModelRequest schema."""

    def setUp(self):
        """Set up common test data."""
        self.valid_triton_properties = {
            "name": "test-model",
            "maxBatchSize": 8,
            "sources": ["s3://bucket/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        }

    def test_load_model_request_with_valid_data(self):
        """Test LoadModelRequest creation with valid data."""
        request = LoadModelRequest(tritonModelProperties=self.valid_triton_properties)

        self.assertIsInstance(request.triton_model_properties, TritonModelProperties)
        self.assertEqual("test-model", request.triton_model_properties.name)

    def test_load_model_request_validation_alias(self):
        """Test that LoadModelRequest accepts both tritonModelProperties and triton_model_properties."""
        # Using validation_alias 'tritonModelProperties'
        request1 = LoadModelRequest(tritonModelProperties=self.valid_triton_properties)
        self.assertIsInstance(request1.triton_model_properties, TritonModelProperties)

        # Using field name 'triton_model_properties'
        request2 = LoadModelRequest(
            triton_model_properties=self.valid_triton_properties
        )
        self.assertIsInstance(request2.triton_model_properties, TritonModelProperties)

        self.assertEqual(
            request1.triton_model_properties.name, request2.triton_model_properties.name
        )

    def test_load_model_request_missing_required_field(self):
        """Test that LoadModelRequest raises ValidationError when tritonModelProperties is missing."""
        with self.assertRaises(ValidationError) as context:
            LoadModelRequest()

        self.assertGreater(len(context.exception.errors()), 0)

    def test_load_model_request_with_triton_model_properties_object(self):
        """Test LoadModelRequest accepts TritonModelProperties object instead of dict."""
        triton_props = TritonModelProperties(**self.valid_triton_properties)
        request = LoadModelRequest(triton_model_properties=triton_props)

        self.assertIsInstance(request.triton_model_properties, TritonModelProperties)
        self.assertEqual("test-model", request.triton_model_properties.name)

    def test_load_model_request_serialization(self):
        """Test that LoadModelRequest can be serialized and deserialized."""
        request1 = LoadModelRequest(tritonModelProperties=self.valid_triton_properties)

        # Serialize to dict
        request_dict = request1.model_dump()
        self.assertIn("triton_model_properties", request_dict)
        self.assertIsInstance(request_dict["triton_model_properties"], dict)

        # Deserialize from dict
        request2 = LoadModelRequest(**request_dict)
        self.assertEqual(
            request1.triton_model_properties.name, request2.triton_model_properties.name
        )


class TestLoadModelResponse(TestCase):
    """Test class for LoadModelResponse schema."""

    def test_load_model_response_with_valid_data(self):
        """Test LoadModelResponse creation with valid message."""
        response = LoadModelResponse(message="Model loaded successfully")

        self.assertEqual("Model loaded successfully", response.message)
        self.assertIsInstance(response.message, str)

    def test_load_model_response_missing_required_field(self):
        """Test that LoadModelResponse raises ValidationError when message is missing."""
        with self.assertRaises(ValidationError):
            LoadModelResponse()

    def test_load_model_response_serialization(self):
        """Test that LoadModelResponse can be serialized and deserialized."""
        response1 = LoadModelResponse(message="Success")

        # Serialize to dict
        response_dict = response1.model_dump()
        self.assertIn("message", response_dict)
        self.assertEqual("Success", response_dict["message"])

        # Deserialize from dict
        response2 = LoadModelResponse(**response_dict)
        self.assertEqual(response1.message, response2.message)

    def test_load_model_response_json_serialization(self):
        """Test JSON serialization and deserialization of LoadModelResponse."""
        response1 = LoadModelResponse(message="Model loaded")

        # Serialize to JSON
        json_str = response1.model_dump_json()
        self.assertIsInstance(json_str, str)

        # Verify JSON structure
        json_data = json.loads(json_str)
        self.assertIn("message", json_data)
        self.assertEqual("Model loaded", json_data["message"])

        # Deserialize from JSON
        response2 = LoadModelResponse.model_validate_json(json_str)
        self.assertEqual(response1.message, response2.message)

    def test_load_model_response_with_various_messages(self):
        """Test LoadModelResponse with various message formats."""
        test_cases = [
            "Model loaded successfully",
            "Error: Failed to load model",
            "Model 'test-model' is now available",
            "",  # Empty string should be valid
        ]

        for message in test_cases:
            with self.subTest(message=message):
                response = LoadModelResponse(message=message)
                self.assertEqual(message, response.message)


class TestUnloadModelRequest(TestCase):
    """Test class for UnloadModelRequest schema."""

    def test_unload_model_request_with_valid_data(self):
        """Test UnloadModelRequest creation with valid message."""
        request = UnloadModelResponse(message="Model unloaded successfully")

        self.assertEqual("Model unloaded successfully", request.message)
        self.assertIsInstance(request.message, str)

    def test_unload_model_request_missing_required_field(self):
        """Test that UnloadModelRequest raises ValidationError when message is missing."""
        with self.assertRaises(ValidationError):
            UnloadModelResponse()

    def test_unload_model_request_serialization(self):
        """Test that UnloadModelRequest can be serialized and deserialized."""
        request1 = UnloadModelResponse(message="Unload complete")

        # Serialize to dict
        request_dict = request1.model_dump()
        self.assertIn("message", request_dict)
        self.assertEqual("Unload complete", request_dict["message"])

        # Deserialize from dict
        request2 = UnloadModelResponse(**request_dict)
        self.assertEqual(request1.message, request2.message)

    def test_unload_model_request_json_serialization(self):
        """Test JSON serialization and deserialization of UnloadModelRequest."""
        request1 = UnloadModelResponse(message="Unloading model")

        # Serialize to JSON
        json_str = request1.model_dump_json()
        self.assertIsInstance(json_str, str)

        # Verify JSON structure
        json_data = json.loads(json_str)
        self.assertIn("message", json_data)
        self.assertEqual("Unloading model", json_data["message"])

        # Deserialize from JSON
        request2 = UnloadModelResponse.model_validate_json(json_str)
        self.assertEqual(request1.message, request2.message)

    def test_unload_model_request_with_various_messages(self):
        """Test UnloadModelRequest with various message formats."""
        test_cases = [
            "Model unloaded successfully",
            "Unloading model 'test-model'",
            "Error: Model not found",
            "",  # Empty string should be valid
        ]

        for message in test_cases:
            with self.subTest(message=message):
                request = UnloadModelResponse(message=message)
                self.assertEqual(message, request.message)
