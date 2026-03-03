import unittest

from pydantic import ValidationError

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    DataType,
    ModelInput,
    ModelOutput,
)
from inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model_properties import (
    HFTritonModelProperties,
    HuggingFaceModelProperties,
    PoolingMethod,
)


class TestPoolingMethod(unittest.TestCase):
    """Test the PoolingMethod enum."""

    def test_pooling_method_enum_values(self):
        """Test that all expected PoolingMethod enum values exist."""
        expected_values = {"mean", "cls"}

        actual_values = {member.value for member in PoolingMethod}

        self.assertEqual(expected_values, actual_values)

    def test_pooling_method_access(self):
        """Test accessing PoolingMethod enum values."""
        self.assertEqual("mean", PoolingMethod.Mean.value)
        self.assertEqual("cls", PoolingMethod.CLS.value)


class TestHFTritonModelProperties(unittest.TestCase):
    """Test the HFTritonModelProperties validation."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valid_inputs = [
            ModelInput(name="input_ids", dims=[1, 128], data_type=DataType.TYPE_INT64),
            ModelInput(
                name="attention_mask", dims=[1, 128], data_type=DataType.TYPE_INT64
            ),
            ModelInput(
                name="token_type_ids", dims=[1, 128], data_type=DataType.TYPE_INT64
            ),
        ]
        self.valid_output = ModelOutput(
            name="last_hidden_state", dims=[1, 128, 768], data_type=DataType.TYPE_FP32
        )

    def test_valid_hf_triton_properties(self):
        """Test creating valid HFTritonModelProperties."""
        properties = HFTritonModelProperties(
            name="hf-text-encoder",
            max_batch_size=32,
            sources=["s3://bucket/model.onnx"],
            input=self.valid_inputs,
            output=[self.valid_output],
        )

        self.assertEqual("hf-text-encoder", properties.name)
        self.assertEqual(32, properties.max_batch_size)
        self.assertEqual(3, len(properties.input))
        self.assertEqual("input_ids", properties.input[0].name)
        self.assertEqual("attention_mask", properties.input[1].name)
        self.assertEqual("token_type_ids", properties.input[2].name)
        self.assertEqual(1, len(properties.output))
        self.assertEqual("last_hidden_state", properties.output[0].name)

    def test_must_have_exactly_three_inputs(self):
        """Test that there must be exactly three inputs."""
        test_cases = [
            ("zero inputs", []),
            ("one input", [self.valid_inputs[0]]),
            ("two inputs", self.valid_inputs[:2]),
            ("four inputs", self.valid_inputs + [self.valid_inputs[0]]),
        ]

        for test_name, inputs in test_cases:
            with self.subTest(msg=test_name):
                with self.assertRaises(ValidationError) as context:
                    HFTritonModelProperties(
                        name="hf-text-encoder",
                        max_batch_size=32,
                        sources=["s3://bucket/model.onnx"],
                        input=inputs,
                        output=[self.valid_output],
                    )

                error_message = str(context.exception)
                # Check for either the custom validation message or Pydantic's base validation
                self.assertTrue(
                    "exactly 3 inputs" in error_message
                    or "at least" in error_message
                    or "at most" in error_message,
                    f"Expected input count error, got: {error_message}",
                )

    def test_inputs_must_have_correct_names(self):
        """Test that inputs must be named input_ids, attention_mask, token_type_ids."""
        test_cases = [
            (
                "wrong first name",
                [
                    ModelInput(
                        name="wrong_name", dims=[1, 128], data_type=DataType.TYPE_INT64
                    ),
                    self.valid_inputs[1],
                    self.valid_inputs[2],
                ],
            ),
            (
                "wrong second name",
                [
                    self.valid_inputs[0],
                    ModelInput(
                        name="wrong_name", dims=[1, 128], data_type=DataType.TYPE_INT64
                    ),
                    self.valid_inputs[2],
                ],
            ),
            (
                "wrong third name",
                [
                    self.valid_inputs[0],
                    self.valid_inputs[1],
                    ModelInput(
                        name="wrong_name", dims=[1, 128], data_type=DataType.TYPE_INT64
                    ),
                ],
            ),
            (
                "correct names but wrong order",
                [
                    self.valid_inputs[1],  # attention_mask first
                    self.valid_inputs[0],  # input_ids second
                    self.valid_inputs[2],
                ],
            ),
        ]

        for test_name, inputs in test_cases:
            with self.subTest(msg=test_name):
                with self.assertRaises(ValidationError) as context:
                    HFTritonModelProperties(
                        name="hf-text-encoder",
                        max_batch_size=32,
                        sources=["s3://bucket/model.onnx"],
                        input=inputs,
                        output=[self.valid_output],
                    )

                error_message = str(context.exception)
                self.assertTrue(
                    "input_ids" in error_message
                    or "attention_mask" in error_message
                    or "token_type_ids" in error_message,
                    f"Expected input name error, got: {error_message}",
                )

    def test_must_have_exactly_one_output(self):
        """Test that there must be exactly one output."""
        test_cases = [
            ("zero outputs", [], "at least 1 item"),
            ("two outputs", [self.valid_output, self.valid_output], "at most 1 item"),
        ]

        for test_name, outputs, expected_error_fragment in test_cases:
            with self.subTest(msg=test_name):
                with self.assertRaises(ValidationError) as context:
                    HFTritonModelProperties(
                        name="hf-text-encoder",
                        max_batch_size=32,
                        sources=["s3://bucket/model.onnx"],
                        input=self.valid_inputs,
                        output=outputs,
                    )

                error_message = str(context.exception)
                self.assertTrue(
                    expected_error_fragment in error_message
                    or "exactly 1 output" in error_message,
                    f"Expected '{expected_error_fragment}' or 'exactly 1 output', got: {error_message}",
                )

    def test_output_must_be_named_last_hidden_state(self):
        """Test that output must be named 'last_hidden_state'."""
        invalid_output = ModelOutput(
            name="wrong_name", dims=[1, 128, 768], data_type=DataType.TYPE_FP32
        )

        with self.assertRaises(ValidationError) as context:
            HFTritonModelProperties(
                name="hf-text-encoder",
                max_batch_size=32,
                sources=["s3://bucket/model.onnx"],
                input=self.valid_inputs,
                output=[invalid_output],
            )

        error_message = str(context.exception)
        self.assertIn("last_hidden_state", error_message)


class TestHuggingFaceModelProperties(unittest.TestCase):
    """Test the HuggingFaceModelProperties class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.text_encoder_properties = HFTritonModelProperties(
            name="hf-text-encoder",
            max_batch_size=32,
            sources=["s3://bucket/text-encoder/model.onnx"],
            input=[
                ModelInput(
                    name="input_ids", dims=[1, 128], data_type=DataType.TYPE_INT64
                ),
                ModelInput(
                    name="attention_mask", dims=[1, 128], data_type=DataType.TYPE_INT64
                ),
                ModelInput(
                    name="token_type_ids", dims=[1, 128], data_type=DataType.TYPE_INT64
                ),
            ],
            output=[
                ModelOutput(
                    name="last_hidden_state",
                    dims=[1, 128, 768],
                    data_type=DataType.TYPE_FP32,
                )
            ],
        )

    def test_create_valid_hf_model_properties_with_required_fields(self):
        """Test creating valid HuggingFaceModelProperties with required fields only."""
        properties = HuggingFaceModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            type="hf",
            dimensions=384,
            pooling_method=PoolingMethod.Mean,
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", properties.name)
        self.assertEqual("hf", properties.type)
        self.assertEqual(384, properties.dimensions)
        self.assertEqual(PoolingMethod.Mean, properties.pooling_method)
        self.assertEqual(128, properties.tokens)  # Default value
        self.assertIsNone(properties.note)

    def test_create_hf_model_properties_with_all_fields(self):
        """Test creating HuggingFaceModelProperties with all optional fields."""
        properties = HuggingFaceModelProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            type="hf",
            dimensions=768,
            pooling_method=PoolingMethod.CLS,
            tokens=512,
            note="High quality sentence embeddings",
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        self.assertEqual("sentence-transformers/all-mpnet-base-v2", properties.name)
        self.assertEqual("hf", properties.type)
        self.assertEqual(768, properties.dimensions)
        self.assertEqual(PoolingMethod.CLS, properties.pooling_method)
        self.assertEqual(512, properties.tokens)
        self.assertEqual("High quality sentence embeddings", properties.note)

    def test_pooling_method_enum_values(self):
        """Test that different PoolingMethod enum values can be set."""
        test_cases = [
            ("Mean pooling", PoolingMethod.Mean),
            ("CLS pooling", PoolingMethod.CLS),
        ]

        for test_name, pooling_method in test_cases:
            with self.subTest(msg=test_name, pooling_method=pooling_method):
                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=pooling_method,
                    triton_text_encoder_properties=self.text_encoder_properties,
                )

                self.assertEqual(pooling_method, properties.pooling_method)

    def test_default_tokens_value(self):
        """Test that tokens defaults to 128."""
        properties = HuggingFaceModelProperties(
            name="test-model",
            type="hf",
            dimensions=384,
            pooling_method=PoolingMethod.Mean,
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        self.assertEqual(128, properties.tokens)

    def test_custom_tokens_value(self):
        """Test that custom tokens value can be set."""
        test_cases = [
            ("tokens=64", 64),
            ("tokens=256", 256),
            ("tokens=512", 512),
        ]

        for test_name, tokens in test_cases:
            with self.subTest(msg=test_name, tokens=tokens):
                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=PoolingMethod.Mean,
                    tokens=tokens,
                    triton_text_encoder_properties=self.text_encoder_properties,
                )

                self.assertEqual(tokens, properties.tokens)

    def test_dimensions_must_be_positive(self):
        """Test that dimensions must be greater than or equal to 1."""
        test_cases = [
            ("zero dimensions", 0),
            ("negative dimensions", -1),
        ]

        for test_name, dimensions in test_cases:
            with self.subTest(msg=test_name, dimensions=dimensions):
                with self.assertRaises(ValidationError) as context:
                    HuggingFaceModelProperties(
                        name="test-model",
                        type="hf",
                        dimensions=dimensions,
                        pooling_method=PoolingMethod.Mean,
                        triton_text_encoder_properties=self.text_encoder_properties,
                    )

                error_message = str(context.exception)
                self.assertIn("dimensions", error_message.lower())

    def test_type_must_be_hf(self):
        """Test that type must be 'hf'."""
        with self.assertRaises(ValidationError) as context:
            HuggingFaceModelProperties(
                name="test-model",
                type="invalid_type",  # type: ignore
                dimensions=384,
                pooling_method=PoolingMethod.Mean,
                triton_text_encoder_properties=self.text_encoder_properties,
            )

        error_message = str(context.exception)
        self.assertIn("type", error_message.lower())

    def test_pooling_method_is_required(self):
        """Test that pooling_method is required."""
        with self.assertRaises(ValidationError) as context:
            HuggingFaceModelProperties(
                name="test-model",
                type="hf",
                dimensions=384,
                triton_text_encoder_properties=self.text_encoder_properties,
            )  # type: ignore (missing pooling_method)

        error_message = str(context.exception)
        self.assertTrue(
            "pooling_method" in error_message.lower()
            or "poolingmethod" in error_message.lower(),
            f"Expected pooling_method error, got: {error_message}",
        )

    def test_triton_text_encoder_properties_is_required(self):
        """Test that triton_text_encoder_properties is required."""
        with self.assertRaises(ValidationError) as context:
            HuggingFaceModelProperties(
                name="test-model",
                type="hf",
                dimensions=384,
                pooling_method=PoolingMethod.Mean,
            )  # type: ignore (missing triton_text_encoder_properties)

        error_message = str(context.exception)
        self.assertTrue(
            "triton_text_encoder_properties" in error_message.lower()
            or "tritontextencoderproperties" in error_message.lower(),
            f"Expected triton_text_encoder_properties error, got: {error_message}",
        )

    def test_immutability(self):
        """Test that HuggingFaceModelProperties is immutable."""
        properties = HuggingFaceModelProperties(
            name="test-model",
            type="hf",
            dimensions=384,
            pooling_method=PoolingMethod.Mean,
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        # Attempt to modify should raise an error
        with self.assertRaises(ValidationError):
            properties.name = "new-name"  # type: ignore

        with self.assertRaises(ValidationError):
            properties.dimensions = 768  # type: ignore

        with self.assertRaises(ValidationError):
            properties.tokens = 256  # type: ignore

    def test_effective_name_returns_name_when_triton_model_name_absent(self):
        """Test that effective_name falls back to name when tritonModelName is not set."""
        properties = HuggingFaceModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            type="hf",
            dimensions=384,
            pooling_method=PoolingMethod.Mean,
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", properties.effective_name)
        self.assertIsNone(properties.triton_model_name)

    def test_effective_name_returns_triton_model_name_when_set(self):
        """Test that effective_name returns tritonModelName when it is set."""
        properties = HuggingFaceModelProperties(
            name="old-model-name",
            type="hf",
            dimensions=384,
            pooling_method=PoolingMethod.Mean,
            tritonModelName="sentence-transformers/all-MiniLM-L6-v2",
            triton_text_encoder_properties=self.text_encoder_properties,
        )

        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", properties.effective_name)
        self.assertEqual("old-model-name", properties.name)
        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", properties.triton_model_name)

    def test_effective_name_with_various_triton_model_names(self):
        """Test effective_name with different tritonModelName values."""
        test_cases = [
            ("with triton_model_name", "old-name", "new-hf-name", "new-hf-name"),
            ("None triton_model_name", "original-name", None, "original-name"),
        ]

        for test_name, name, triton_model_name, expected_effective in test_cases:
            with self.subTest(msg=test_name):
                kwargs = dict(
                    name=name,
                    type="hf",
                    dimensions=384,
                    pooling_method=PoolingMethod.Mean,
                    triton_text_encoder_properties=self.text_encoder_properties,
                )
                if triton_model_name is not None:
                    kwargs["tritonModelName"] = triton_model_name

                properties = HuggingFaceModelProperties(**kwargs)

                self.assertEqual(expected_effective, properties.effective_name)

    def test_field_aliases(self):
        """Test that field aliases work correctly."""
        properties = HuggingFaceModelProperties(
            name="test-model",
            type="hf",
            dimensions=384,
            poolingMethod="mean",  # Using alias
            tritonTextEncoderProperties=self.text_encoder_properties,  # Using alias
        )

        self.assertEqual(PoolingMethod.Mean, properties.pooling_method)
        self.assertEqual(
            self.text_encoder_properties, properties.triton_text_encoder_properties
        )

    def test_various_batch_sizes(self):
        """Test that different batch sizes can be set for the encoder."""
        test_cases = [
            ("batch_size=8", 8),
            ("batch_size=16", 16),
            ("batch_size=64", 64),
            ("batch_size=128", 128),
        ]

        for test_name, batch_size in test_cases:
            with self.subTest(msg=test_name, batch_size=batch_size):
                encoder_props = HFTritonModelProperties(
                    name="hf-encoder",
                    max_batch_size=batch_size,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(
                            name="input_ids",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                        ModelInput(
                            name="attention_mask",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                        ModelInput(
                            name="token_type_ids",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                    ],
                    output=[
                        ModelOutput(
                            name="last_hidden_state",
                            dims=[1, 128, 768],
                            data_type=DataType.TYPE_FP32,
                        )
                    ],
                )

                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=PoolingMethod.Mean,
                    triton_text_encoder_properties=encoder_props,
                )

                self.assertEqual(
                    batch_size, properties.triton_text_encoder_properties.max_batch_size
                )

    def test_different_input_data_types(self):
        """Test that different data types can be used for inputs."""
        test_cases = [
            ("INT32", DataType.TYPE_INT32),
            ("INT64", DataType.TYPE_INT64),
        ]

        for test_name, data_type in test_cases:
            with self.subTest(msg=test_name, data_type=data_type):
                encoder_props = HFTritonModelProperties(
                    name="hf-encoder",
                    max_batch_size=32,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(
                            name="input_ids", dims=[1, 128], data_type=data_type
                        ),
                        ModelInput(
                            name="attention_mask", dims=[1, 128], data_type=data_type
                        ),
                        ModelInput(
                            name="token_type_ids", dims=[1, 128], data_type=data_type
                        ),
                    ],
                    output=[
                        ModelOutput(
                            name="last_hidden_state",
                            dims=[1, 128, 768],
                            data_type=DataType.TYPE_FP32,
                        )
                    ],
                )

                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=PoolingMethod.Mean,
                    triton_text_encoder_properties=encoder_props,
                )

                self.assertEqual(
                    data_type,
                    properties.triton_text_encoder_properties.input[0].data_type,
                )
                self.assertEqual(
                    data_type,
                    properties.triton_text_encoder_properties.input[1].data_type,
                )
                self.assertEqual(
                    data_type,
                    properties.triton_text_encoder_properties.input[2].data_type,
                )

    def test_different_output_data_types(self):
        """Test that different data types can be used for output."""
        test_cases = [
            ("FP16", DataType.TYPE_FP16),
            ("FP32", DataType.TYPE_FP32),
            ("FP64", DataType.TYPE_FP64),
        ]

        for test_name, data_type in test_cases:
            with self.subTest(msg=test_name, data_type=data_type):
                encoder_props = HFTritonModelProperties(
                    name="hf-encoder",
                    max_batch_size=32,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(
                            name="input_ids",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                        ModelInput(
                            name="attention_mask",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                        ModelInput(
                            name="token_type_ids",
                            dims=[1, 128],
                            data_type=DataType.TYPE_INT64,
                        ),
                    ],
                    output=[
                        ModelOutput(
                            name="last_hidden_state",
                            dims=[1, 128, 768],
                            data_type=data_type,
                        )
                    ],
                )

                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=PoolingMethod.Mean,
                    triton_text_encoder_properties=encoder_props,
                )

                self.assertEqual(
                    data_type,
                    properties.triton_text_encoder_properties.output[0].data_type,
                )

    def test_different_token_dimensions(self):
        """Test that different token sequence lengths can be specified."""
        test_cases = [
            ("dims=77", [1, 77]),
            ("dims=128", [1, 128]),
            ("dims=256", [1, 256]),
            ("dims=512", [1, 512]),
        ]

        for test_name, dims in test_cases:
            with self.subTest(msg=test_name, dims=dims):
                encoder_props = HFTritonModelProperties(
                    name="hf-encoder",
                    max_batch_size=32,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(
                            name="input_ids", dims=dims, data_type=DataType.TYPE_INT64
                        ),
                        ModelInput(
                            name="attention_mask",
                            dims=dims,
                            data_type=DataType.TYPE_INT64,
                        ),
                        ModelInput(
                            name="token_type_ids",
                            dims=dims,
                            data_type=DataType.TYPE_INT64,
                        ),
                    ],
                    output=[
                        ModelOutput(
                            name="last_hidden_state",
                            dims=dims + [768],
                            data_type=DataType.TYPE_FP32,
                        )
                    ],
                )

                properties = HuggingFaceModelProperties(
                    name="test-model",
                    type="hf",
                    dimensions=768,
                    pooling_method=PoolingMethod.Mean,
                    triton_text_encoder_properties=encoder_props,
                )

                self.assertEqual(
                    dims, properties.triton_text_encoder_properties.input[0].dims
                )
                self.assertEqual(
                    dims + [768],
                    properties.triton_text_encoder_properties.output[0].dims,
                )


if __name__ == "__main__":
    unittest.main()
