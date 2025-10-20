from unittest import TestCase

from model_management.schemas.triton_model_properties import (
    DataType,
    ModelInput,
    ModelOutput,
    TritonModelProperties,
)
from pydantic import ValidationError


class TestDataType(TestCase):
    """Test class for DataType enum."""

    def test_data_type_enum_values(self):
        """Test that DataType enum has all expected values."""
        expected_types = [
            "TYPE_FP64",
            "TYPE_FP32",
            "TYPE_FP16",
            "TYPE_INT8",
            "TYPE_INT16",
            "TYPE_INT32",
            "TYPE_INT64",
            "TYPE_BF16",
        ]

        for expected_type in expected_types:
            with self.subTest(data_type=expected_type):
                self.assertIn(expected_type, [dt.value for dt in DataType])

    def test_data_type_string_representation(self):
        """Test that DataType values are strings."""
        test_cases = [
            (DataType.TYPE_FP32, "TYPE_FP32"),
            (DataType.TYPE_FP16, "TYPE_FP16"),
            (DataType.TYPE_INT8, "TYPE_INT8"),
        ]

        for data_type, expected_str in test_cases:
            with self.subTest(data_type=expected_str):
                self.assertEqual(expected_str, str(data_type))
                self.assertEqual(expected_str, data_type.value)


class TestModelInput(TestCase):
    """Test class for ModelInput schema."""

    def test_model_input_with_valid_data(self):
        """Test ModelInput creation with valid data."""
        test_cases = [
            (
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"},
                "image input with FP32",
            ),
            (
                {"name": "text_input", "dims": [512], "dataType": "TYPE_INT32"},
                "text input with INT32",
            ),
            (
                {"name": "batch_input", "dims": [1, 768], "dataType": "TYPE_FP16"},
                "batch input with FP16",
            ),
        ]

        for input_data, msg in test_cases:
            with self.subTest(msg=msg):
                model_input = ModelInput(**input_data)
                self.assertEqual(input_data["name"], model_input.name)
                self.assertEqual(input_data["dims"], model_input.dims)
                self.assertEqual(
                    DataType(input_data["dataType"]), model_input.data_type
                )

    def test_model_input_validation_alias(self):
        """Test that ModelInput accepts both dataType and data_type."""
        # Using validation_alias 'dataType'
        input1 = ModelInput(name="input", dims=[3, 224, 224], dataType="TYPE_FP32")
        self.assertEqual(DataType.TYPE_FP32, input1.data_type)

        # Using field name 'data_type'
        input2 = ModelInput(name="input", dims=[3, 224, 224], data_type="TYPE_FP32")
        self.assertEqual(DataType.TYPE_FP32, input2.data_type)

    def test_model_input_missing_required_fields(self):
        """Test that ModelInput raises ValidationError when required fields are missing."""
        test_cases = [
            ({}, "all fields missing"),
            ({"name": "input"}, "dims and dataType missing"),
            ({"name": "input", "dims": [3, 224, 224]}, "dataType missing"),
            ({"dims": [3, 224, 224], "dataType": "TYPE_FP32"}, "name missing"),
        ]

        for invalid_data, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValidationError):
                    ModelInput(**invalid_data)

    def test_model_input_invalid_data_type(self):
        """Test that ModelInput raises ValidationError for invalid data type."""
        invalid_types = [
            "INVALID_TYPE",
            "TYPE_FLOAT",
            "int32",
            "fp32",
            123,
        ]

        for invalid_type in invalid_types:
            with self.subTest(data_type=invalid_type):
                with self.assertRaises(ValidationError):
                    ModelInput(name="input", dims=[3, 224, 224], dataType=invalid_type)

    def test_model_input_empty_dims(self):
        """Test that ModelInput allows empty dims list."""
        model_input = ModelInput(name="input", dims=[], dataType="TYPE_FP32")
        self.assertEqual([], model_input.dims)
        self.assertEqual(0, len(model_input.dims))

    def test_model_input_various_dims(self):
        """Test ModelInput with various dimension configurations."""
        test_cases = [
            ([1], "1D tensor"),
            ([768], "1D embedding"),
            ([3, 224, 224], "3D image tensor"),
            ([1, 512, 768], "3D batch tensor"),
            ([2, 3, 224, 224], "4D batch image tensor"),
        ]

        for dims, msg in test_cases:
            with self.subTest(msg=msg):
                model_input = ModelInput(name="input", dims=dims, dataType="TYPE_FP32")
                self.assertEqual(dims, model_input.dims)
                self.assertEqual(len(dims), len(model_input.dims))


class TestModelOutput(TestCase):
    """Test class for ModelOutput schema."""

    def test_model_output_with_valid_data(self):
        """Test ModelOutput creation with valid data."""
        test_cases = [
            (
                {"name": "output", "dims": [768], "dataType": "TYPE_FP32"},
                "embedding output",
            ),
            (
                {"name": "logits", "dims": [1000], "dataType": "TYPE_FP16"},
                "classification output",
            ),
        ]

        for output_data, msg in test_cases:
            with self.subTest(msg=msg):
                model_output = ModelOutput(**output_data)
                self.assertEqual(output_data["name"], model_output.name)
                self.assertEqual(output_data["dims"], model_output.dims)
                self.assertEqual(
                    DataType(output_data["dataType"]), model_output.data_type
                )

    def test_model_output_validation_alias(self):
        """Test that ModelOutput accepts both dataType and data_type."""
        # Using validation_alias 'dataType'
        output1 = ModelOutput(name="output", dims=[768], dataType="TYPE_FP32")
        self.assertEqual(DataType.TYPE_FP32, output1.data_type)

        # Using field name 'data_type'
        output2 = ModelOutput(name="output", dims=[768], data_type="TYPE_FP32")
        self.assertEqual(DataType.TYPE_FP32, output2.data_type)

    def test_model_output_missing_required_fields(self):
        """Test that ModelOutput raises ValidationError when required fields are missing."""
        test_cases = [
            ({}, "all fields missing"),
            ({"name": "output"}, "dims and dataType missing"),
            ({"name": "output", "dims": [768]}, "dataType missing"),
        ]

        for invalid_data, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValidationError):
                    ModelOutput(**invalid_data)


class TestTritonModelProperties(TestCase):
    """Test class for TritonModelProperties schema."""

    def setUp(self):
        """Set up common test data."""
        self.valid_model_data = {
            "name": "test-model",
            "maxBatchSize": 8,
            "sources": ["s3://bucket/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        }

    def test_triton_model_properties_with_valid_data(self):
        """Test TritonModelProperties creation with valid data."""
        model = TritonModelProperties(**self.valid_model_data)

        self.assertEqual("test-model", model.name)
        self.assertEqual(8, model.max_batch_size)
        self.assertEqual(1, len(model.sources))
        self.assertEqual("s3://bucket/model.onnx", model.sources[0])
        self.assertEqual(1, len(model.input))
        self.assertIsInstance(model.input[0], ModelInput)
        self.assertEqual(1, len(model.output))
        self.assertIsInstance(model.output[0], ModelOutput)

    def test_triton_model_properties_default_max_batch_size(self):
        """Test that max_batch_size defaults to 8 when not provided."""
        data = self.valid_model_data.copy()
        del data["maxBatchSize"]

        model = TritonModelProperties(**data)
        self.assertEqual(8, model.max_batch_size)

    def test_triton_model_properties_validation_aliases(self):
        """Test that TritonModelProperties accepts validation aliases."""
        # Using validation aliases
        model1 = TritonModelProperties(**self.valid_model_data)
        self.assertEqual(8, model1.max_batch_size)

        # Using field names
        data = {
            "name": "test-model",
            "max_batch_size": 16,
            "sources": ["s3://bucket/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "data_type": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "data_type": "TYPE_FP32"}],
        }
        model2 = TritonModelProperties(**data)
        self.assertEqual(16, model2.max_batch_size)

    def test_triton_model_properties_max_batch_size_constraints(self):
        """Test max_batch_size validation constraints (gt=0, le=128)."""
        test_cases = [
            (1, True, "minimum valid batch size"),
            (64, True, "mid-range batch size"),
            (128, True, "maximum valid batch size"),
            (0, False, "zero batch size should fail"),
            (-1, False, "negative batch size should fail"),
            (129, False, "batch size above 128 should fail"),
            (1000, False, "large batch size should fail"),
        ]

        for batch_size, should_pass, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                data["maxBatchSize"] = batch_size

                if should_pass:
                    model = TritonModelProperties(**data)
                    self.assertEqual(batch_size, model.max_batch_size)
                else:
                    with self.assertRaises(ValidationError):
                        TritonModelProperties(**data)

    def test_triton_model_properties_sources_length_constraints(self):
        """Test sources list length constraints (min_length=1, max_length=5)."""
        test_cases = [
            ([], False, "empty sources list should fail"),
            (["s3://bucket/model.onnx"], True, "single source"),
            (
                ["s3://bucket/model.onnx", "s3://bucket/model.onnx.data"],
                True,
                "two sources",
            ),
            (
                ["s3://bucket/model.onnx"]
                + [f"s3://bucket/model.onnx.data_{i}" for i in range(4)],
                True,
                "five sources",
            ),
            (
                ["s3://bucket/model.onnx"]
                + [f"s3://bucket/model.onnx.data_{i}" for i in range(5)],
                False,
                "six sources should fail",
            ),
        ]

        for sources, should_pass, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                data["sources"] = sources

                if should_pass:
                    model = TritonModelProperties(**data)
                    self.assertEqual(len(sources), len(model.sources))
                else:
                    with self.assertRaises(ValidationError):
                        TritonModelProperties(**data)

    def test_triton_model_properties_sources_validation(self):
        """Test that sources must point to model.onnx or model.onnx.data files."""
        test_cases = [
            (["s3://bucket/model.onnx"], True, "model.onnx file"),
            (["http://example.com/model.onnx"], True, "http URL with model.onnx"),
            (["file:///path/to/model.onnx"], True, "file path to model.onnx"),
            (["s3://bucket/model.onnx.data"], True, "model.onnx.data file"),
            (["s3://bucket/model.onnx.data_0"], True, "model.onnx.data with suffix"),
            (["s3://bucket/model.pb"], False, "invalid .pb file"),
            (["s3://bucket/model.pt"], False, "invalid .pt file"),
            (["s3://bucket/weights.bin"], False, "invalid .bin file"),
            (["s3://bucket/random.txt"], False, "invalid .txt file"),
        ]

        for sources, should_pass, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                data["sources"] = sources

                if should_pass:
                    model = TritonModelProperties(**data)
                    self.assertEqual(sources, model.sources)
                else:
                    with self.assertRaises(ValueError) as context:
                        TritonModelProperties(**data)
                    self.assertIn("model.onnx", str(context.exception))

    def test_triton_model_properties_output_length_constraint(self):
        """Test that output list must have exactly 1 element (min_length=1, max_length=1)."""
        test_cases = [
            ([], False, "empty output list should fail"),
            (
                [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
                True,
                "single output",
            ),
            (
                [
                    {"name": "output1", "dims": [768], "dataType": "TYPE_FP32"},
                    {"name": "output2", "dims": [512], "dataType": "TYPE_FP32"},
                ],
                False,
                "two outputs should fail",
            ),
        ]

        for outputs, should_pass, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                data["output"] = outputs

                if should_pass:
                    model = TritonModelProperties(**data)
                    self.assertEqual(len(outputs), len(model.output))
                else:
                    with self.assertRaises(ValidationError):
                        TritonModelProperties(**data)

    def test_triton_model_properties_missing_required_fields(self):
        """Test that TritonModelProperties raises ValidationError when required fields are missing."""
        test_cases = [
            ("name", "name is required"),
            ("sources", "sources is required"),
            ("input", "input is required"),
            ("output", "output is required"),
        ]

        for field_to_remove, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                del data[field_to_remove]

                with self.assertRaises(ValidationError):
                    TritonModelProperties(**data)

    def test_triton_model_properties_multiple_inputs(self):
        """Test TritonModelProperties with multiple input tensors."""
        data = self.valid_model_data.copy()
        data["input"] = [
            {"name": "image", "dims": [3, 224, 224], "dataType": "TYPE_FP32"},
            {"name": "text", "dims": [512], "dataType": "TYPE_INT32"},
        ]

        model = TritonModelProperties(**data)
        self.assertEqual(2, len(model.input))
        self.assertEqual("image", model.input[0].name)
        self.assertEqual("text", model.input[1].name)

    def test_triton_model_properties_serialization(self):
        """Test that TritonModelProperties can be serialized and deserialized."""
        model1 = TritonModelProperties(**self.valid_model_data)

        # Serialize to dict
        model_dict = model1.model_dump()

        # Check key fields are present
        self.assertIn("name", model_dict)
        self.assertIn("max_batch_size", model_dict)
        self.assertIn("sources", model_dict)
        self.assertIn("input", model_dict)
        self.assertIn("output", model_dict)

        # Deserialize from dict
        model2 = TritonModelProperties(**model_dict)

        self.assertEqual(model1.name, model2.name)
        self.assertEqual(model1.max_batch_size, model2.max_batch_size)
        self.assertEqual(model1.sources, model2.sources)

    def test_triton_model_properties_with_various_source_urls(self):
        """Test TritonModelProperties with various valid source URL formats."""
        test_cases = [
            ("s3://bucket/path/to/model.onnx", "S3 path"),
            ("https://example.com/models/model.onnx", "HTTPS URL"),
            ("http://example.com/model.onnx", "HTTP URL"),
            ("file:///local/path/model.onnx", "file URL"),
            ("/absolute/path/model.onnx", "absolute path"),
            ("./relative/path/model.onnx", "relative path"),
        ]

        for source_url, msg in test_cases:
            with self.subTest(msg=msg):
                data = self.valid_model_data.copy()
                data["sources"] = [source_url]

                model = TritonModelProperties(**data)
                self.assertEqual(source_url, model.sources[0])
