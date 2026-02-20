import unittest

import numpy as np
from pydantic import ValidationError

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    DataType,
    ModelInput,
    ModelOutput,
    TritonModelProperties,
)
from inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model_properties import (
    ImagePreprocessor,
    OpenCLIPModelProperties,
    OpenCLIPTritonModelProperties,
    Precision,
)


class TestImagePreprocessor(unittest.TestCase):
    """Test the ImagePreprocessor enum."""

    def test_image_preprocessor_enum_values(self):
        """Test that all expected ImagePreprocessor enum values exist."""
        expected_values = {"SigLIP", "OpenAI", "OpenCLIP", "CLIPA"}

        actual_values = {member.value for member in ImagePreprocessor}

        self.assertEqual(expected_values, actual_values)

    def test_image_preprocessor_access(self):
        """Test accessing ImagePreprocessor enum values."""
        self.assertEqual("SigLIP", ImagePreprocessor.SigLIP.value)
        self.assertEqual("OpenAI", ImagePreprocessor.OpenAI.value)
        self.assertEqual("OpenCLIP", ImagePreprocessor.OpenCLIP.value)
        self.assertEqual("CLIPA", ImagePreprocessor.CLIPA.value)


class TestPrecision(unittest.TestCase):
    """Test the Precision enum."""

    def test_precision_enum_values(self):
        """Test that all expected Precision enum values exist."""
        expected_values = {"fp32", "fp16"}

        actual_values = {member.value for member in Precision}

        self.assertEqual(expected_values, actual_values)

    def test_precision_access(self):
        """Test accessing Precision enum values."""
        self.assertEqual("fp32", Precision.FP32.value)
        self.assertEqual("fp16", Precision.FP16.value)


class TestOpenCLIPTritonModelProperties(unittest.TestCase):
    """Test the OpenCLIPTritonModelProperties validation."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valid_input = ModelInput(
            name="input", dims=[1, 77], data_type=DataType.TYPE_INT64
        )
        self.valid_output = ModelOutput(
            name="output", dims=[1, 512], data_type=DataType.TYPE_FP32
        )

    def test_valid_openclip_triton_properties(self):
        """Test creating valid OpenCLIPTritonModelProperties."""
        properties = OpenCLIPTritonModelProperties(
            name="openclip-text-encoder",
            max_batch_size=32,
            sources=["s3://bucket/model.onnx"],
            input=[self.valid_input],
            output=[self.valid_output],
        )

        self.assertEqual("openclip-text-encoder", properties.name)
        self.assertEqual(32, properties.max_batch_size)
        self.assertEqual(1, len(properties.input))
        self.assertEqual("input", properties.input[0].name)
        self.assertEqual(1, len(properties.output))
        self.assertEqual("output", properties.output[0].name)

    def test_input_must_be_named_input(self):
        """Test that input must be named 'input'."""
        invalid_input = ModelInput(
            name="wrong_name", dims=[1, 77], data_type=DataType.TYPE_INT64
        )

        with self.assertRaises(ValidationError) as context:
            OpenCLIPTritonModelProperties(
                name="openclip-text-encoder",
                max_batch_size=32,
                sources=["s3://bucket/model.onnx"],
                input=[invalid_input],
                output=[self.valid_output],
            )

        error_message = str(context.exception)
        self.assertIn("input must be named 'input'", error_message)
        self.assertIn("wrong_name", error_message)

    def test_output_must_be_named_output(self):
        """Test that output must be named 'output'."""
        invalid_output = ModelOutput(
            name="wrong_name", dims=[1, 512], data_type=DataType.TYPE_FP32
        )

        with self.assertRaises(ValidationError) as context:
            OpenCLIPTritonModelProperties(
                name="openclip-text-encoder",
                max_batch_size=32,
                sources=["s3://bucket/model.onnx"],
                input=[self.valid_input],
                output=[invalid_output],
            )

        error_message = str(context.exception)
        self.assertIn("output must be named 'output'", error_message)
        self.assertIn("wrong_name", error_message)

    def test_must_have_exactly_one_input(self):
        """Test that there must be exactly one input."""
        test_cases = [
            ("zero inputs", [], "at least 1 item"),
            ("two inputs", [self.valid_input, self.valid_input], "at most 1 item"),
        ]

        for test_name, inputs, expected_error_fragment in test_cases:
            with self.subTest(msg=test_name):
                with self.assertRaises(ValidationError) as context:
                    OpenCLIPTritonModelProperties(
                        name="openclip-text-encoder",
                        max_batch_size=32,
                        sources=["s3://bucket/model.onnx"],
                        input=inputs,
                        output=[self.valid_output],
                    )

                error_message = str(context.exception)
                # For zero inputs, check base validation first (from TritonModelProperties)
                # For two inputs, check OpenCLIP-specific validation
                self.assertTrue(
                    expected_error_fragment in error_message
                    or "exactly 1 input" in error_message,
                    f"Expected '{expected_error_fragment}' or 'exactly 1 input' in error message, got: {error_message}",
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
                    OpenCLIPTritonModelProperties(
                        name="openclip-text-encoder",
                        max_batch_size=32,
                        sources=["s3://bucket/model.onnx"],
                        input=[self.valid_input],
                        output=outputs,
                    )

                error_message = str(context.exception)
                # For zero outputs, check base validation first (from TritonModelProperties)
                # For two outputs, check OpenCLIP-specific validation
                self.assertTrue(
                    expected_error_fragment in error_message
                    or "exactly 1 output" in error_message,
                    f"Expected '{expected_error_fragment}' or 'exactly 1 output' in error message, got: {error_message}",
                )


class TestOpenCLIPModelProperties(unittest.TestCase):
    """Test the OpenCLIPModelProperties class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.text_encoder_properties = TritonModelProperties(
            name="openclip-text-encoder",
            max_batch_size=32,
            sources=["s3://bucket/text-encoder/model.onnx"],
            input=[
                ModelInput(name="input", dims=[1, 77], data_type=DataType.TYPE_INT64)
            ],
            output=[
                ModelOutput(name="output", dims=[1, 512], data_type=DataType.TYPE_FP32)
            ],
        )

        self.image_encoder_properties = TritonModelProperties(
            name="openclip-image-encoder",
            max_batch_size=16,
            sources=["s3://bucket/image-encoder/model.onnx"],
            input=[
                ModelInput(
                    name="input", dims=[1, 3, 224, 224], data_type=DataType.TYPE_FP32
                )
            ],
            output=[
                ModelOutput(name="output", dims=[1, 512], data_type=DataType.TYPE_FP32)
            ],
        )

    def test_create_valid_openclip_model_properties(self):
        """Test creating valid OpenCLIPModelProperties with required fields only."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-32",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        self.assertEqual("ViT-B-32", properties.name)
        self.assertEqual("open_clip", properties.type)
        self.assertEqual(512, properties.dimensions)
        self.assertEqual(ImagePreprocessor.OpenCLIP, properties.image_preprocessor)
        self.assertIsNone(properties.tokenizer)
        self.assertIsNone(properties.mean)
        self.assertIsNone(properties.std)
        self.assertIsNone(properties.size)
        self.assertIsNone(properties.note)

    def test_create_openclip_model_properties_with_all_fields(self):
        """Test creating OpenCLIPModelProperties with all optional fields."""
        properties = OpenCLIPModelProperties(
            name="ViT-L-14",
            type="open_clip",
            dimensions=768,
            tokenizer="custom-tokenizer",
            image_preprocessor=ImagePreprocessor.SigLIP,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            size=336,
            note="Custom model with SigLIP preprocessor",
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        self.assertEqual("ViT-L-14", properties.name)
        self.assertEqual("open_clip", properties.type)
        self.assertEqual(768, properties.dimensions)
        self.assertEqual("custom-tokenizer", properties.tokenizer)
        self.assertEqual(ImagePreprocessor.SigLIP, properties.image_preprocessor)
        self.assertEqual([0.5, 0.5, 0.5], properties.mean)
        self.assertEqual([0.5, 0.5, 0.5], properties.std)
        self.assertEqual(336, properties.size)
        self.assertEqual("Custom model with SigLIP preprocessor", properties.note)

    def test_image_preprocessor_enum_values(self):
        """Test that different ImagePreprocessor enum values can be set."""
        test_cases = [
            ("SigLIP", ImagePreprocessor.SigLIP),
            ("OpenAI", ImagePreprocessor.OpenAI),
            ("OpenCLIP", ImagePreprocessor.OpenCLIP),
            ("CLIPA", ImagePreprocessor.CLIPA),
        ]

        for test_name, preprocessor in test_cases:
            with self.subTest(msg=test_name, preprocessor=preprocessor):
                properties = OpenCLIPModelProperties(
                    name="test-model",
                    type="open_clip",
                    dimensions=512,
                    image_preprocessor=preprocessor,
                    triton_text_encoder_properties=self.text_encoder_properties,
                    triton_image_encoder_properties=self.image_encoder_properties,
                )

                self.assertEqual(preprocessor, properties.image_preprocessor)

    def test_cached_derived_types_text_encoder(self):
        """Test that derived types are cached correctly for text encoder."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-32",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        # Verify text encoder types
        self.assertEqual(np.int64, properties.text_input_numpy_type)
        self.assertEqual("INT64", properties.text_input_triton_type)

    def test_cached_derived_types_image_encoder(self):
        """Test that derived types are cached correctly for image encoder."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-32",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        # Verify image encoder types
        self.assertEqual(np.float32, properties.image_input_numpy_type)
        self.assertEqual("FP32", properties.image_input_triton_type)

    def test_cached_derived_types_with_different_data_types(self):
        """Test derived type caching with various data types."""
        test_cases = [
            (
                "FP16 text, FP32 image",
                DataType.TYPE_FP16,
                DataType.TYPE_FP32,
                np.float16,
                "FP16",
                np.float32,
                "FP32",
            ),
            (
                "INT32 text, FP16 image",
                DataType.TYPE_INT32,
                DataType.TYPE_FP16,
                np.int32,
                "INT32",
                np.float16,
                "FP16",
            ),
            (
                "INT64 text, FP64 image",
                DataType.TYPE_INT64,
                DataType.TYPE_FP64,
                np.int64,
                "INT64",
                np.float64,
                "FP64",
            ),
        ]

        for (
            test_name,
            text_dtype,
            image_dtype,
            expected_text_np,
            expected_text_triton,
            expected_image_np,
            expected_image_triton,
        ) in test_cases:
            with self.subTest(msg=test_name):
                text_props = TritonModelProperties(
                    name="text-encoder",
                    max_batch_size=32,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(name="input", dims=[1, 77], data_type=text_dtype)
                    ],
                    output=[
                        ModelOutput(
                            name="output", dims=[1, 512], data_type=DataType.TYPE_FP32
                        )
                    ],
                )

                image_props = TritonModelProperties(
                    name="image-encoder",
                    max_batch_size=16,
                    sources=["s3://bucket/model.onnx"],
                    input=[
                        ModelInput(
                            name="input", dims=[1, 3, 224, 224], data_type=image_dtype
                        )
                    ],
                    output=[
                        ModelOutput(
                            name="output", dims=[1, 512], data_type=DataType.TYPE_FP32
                        )
                    ],
                )

                properties = OpenCLIPModelProperties(
                    name="test-model",
                    type="open_clip",
                    dimensions=512,
                    triton_text_encoder_properties=text_props,
                    triton_image_encoder_properties=image_props,
                )

                self.assertEqual(expected_text_np, properties.text_input_numpy_type)
                self.assertEqual(
                    expected_text_triton, properties.text_input_triton_type
                )
                self.assertEqual(expected_image_np, properties.image_input_numpy_type)
                self.assertEqual(
                    expected_image_triton, properties.image_input_triton_type
                )

    def test_dimensions_must_be_positive(self):
        """Test that dimensions must be greater than or equal to 1."""
        test_cases = [
            ("zero dimensions", 0),
            ("negative dimensions", -1),
        ]

        for test_name, dimensions in test_cases:
            with self.subTest(msg=test_name, dimensions=dimensions):
                with self.assertRaises(ValidationError) as context:
                    OpenCLIPModelProperties(
                        name="test-model",
                        type="open_clip",
                        dimensions=dimensions,
                        triton_text_encoder_properties=self.text_encoder_properties,
                        triton_image_encoder_properties=self.image_encoder_properties,
                    )

                error_message = str(context.exception)
                self.assertIn("dimensions", error_message.lower())

    def test_type_must_be_open_clip(self):
        """Test that type must be 'open_clip'."""
        with self.assertRaises(ValidationError) as context:
            OpenCLIPModelProperties(
                name="test-model",
                type="invalid_type",  # type: ignore
                dimensions=512,
                triton_text_encoder_properties=self.text_encoder_properties,
                triton_image_encoder_properties=self.image_encoder_properties,
            )

        error_message = str(context.exception)
        self.assertIn("type", error_message.lower())

    def test_immutability(self):
        """Test that OpenCLIPModelProperties is immutable."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-32",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        # Attempt to modify should raise an error
        with self.assertRaises(ValidationError):
            properties.name = "new-name"  # type: ignore

        with self.assertRaises(ValidationError):
            properties.dimensions = 1024  # type: ignore

    def test_mean_and_std_as_lists(self):
        """Test that mean and std accept lists of floats."""
        properties = OpenCLIPModelProperties(
            name="test-model",
            type="open_clip",
            dimensions=512,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        self.assertEqual([0.485, 0.456, 0.406], properties.mean)
        self.assertEqual([0.229, 0.224, 0.225], properties.std)

    def test_triton_encoder_properties_validation(self):
        """Test that both triton encoder properties are required."""
        # Missing text encoder
        with self.assertRaises(ValidationError) as context:
            OpenCLIPModelProperties(
                name="test-model",
                type="open_clip",
                dimensions=512,
                triton_image_encoder_properties=self.image_encoder_properties,
            )  # type: ignore

        error_message = str(context.exception)
        # Pydantic uses camelCase in error messages when using field aliases
        self.assertTrue(
            "triton_text_encoder_properties" in error_message.lower()
            or "tritontextencoderproperties" in error_message.lower(),
            f"Expected field name in error message, got: {error_message}",
        )

        # Missing image encoder
        with self.assertRaises(ValidationError) as context:
            OpenCLIPModelProperties(
                name="test-model",
                type="open_clip",
                dimensions=512,
                triton_text_encoder_properties=self.text_encoder_properties,
            )  # type: ignore

        error_message = str(context.exception)
        # Pydantic uses camelCase in error messages when using field aliases
        self.assertTrue(
            "triton_image_encoder_properties" in error_message.lower()
            or "tritonimageencoderproperties" in error_message.lower(),
            f"Expected field name in error message, got: {error_message}",
        )

    def test_field_aliases(self):
        """Test that field aliases work correctly."""
        properties = OpenCLIPModelProperties(
            name="test-model",
            type="open_clip",
            dimensions=512,
            imagePreprocessor="SigLIP",  # Using alias
            tritonTextEncoderProperties=self.text_encoder_properties,  # Using alias
            tritonImageEncoderProperties=self.image_encoder_properties,  # Using alias
        )

        self.assertEqual(ImagePreprocessor.SigLIP, properties.image_preprocessor)
        self.assertEqual(
            self.text_encoder_properties, properties.triton_text_encoder_properties
        )
        self.assertEqual(
            self.image_encoder_properties, properties.triton_image_encoder_properties
        )

    def test_effective_name_returns_name_when_triton_model_name_absent(self):
        """Test that effective_name falls back to name when tritonModelName is not set."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-32",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        self.assertEqual("ViT-B-32", properties.effective_name)
        self.assertIsNone(properties.triton_model_name)

    def test_effective_name_returns_triton_model_name_when_set(self):
        """Test that effective_name returns tritonModelName when it is set."""
        properties = OpenCLIPModelProperties(
            name="ViT-B-16-SigLIP",
            type="open_clip",
            dimensions=512,
            tritonModelName="hf-hub:timm/ViT-B-16-SigLIP",
            triton_text_encoder_properties=self.text_encoder_properties,
            triton_image_encoder_properties=self.image_encoder_properties,
        )

        self.assertEqual("hf-hub:timm/ViT-B-16-SigLIP", properties.effective_name)
        self.assertEqual("ViT-B-16-SigLIP", properties.name)
        self.assertEqual("hf-hub:timm/ViT-B-16-SigLIP", properties.triton_model_name)

    def test_effective_name_with_various_triton_model_names(self):
        """Test effective_name with different tritonModelName values."""
        test_cases = [
            ("hf-hub prefix", "old-name", "hf-hub:org/model", "hf-hub:org/model"),
            ("open_clip prefix", "old-name", "open_clip/ViT-B-32/openai", "open_clip/ViT-B-32/openai"),
            ("None triton_model_name", "original-name", None, "original-name"),
        ]

        for test_name, name, triton_model_name, expected_effective in test_cases:
            with self.subTest(msg=test_name):
                kwargs = dict(
                    name=name,
                    type="open_clip",
                    dimensions=512,
                    triton_text_encoder_properties=self.text_encoder_properties,
                    triton_image_encoder_properties=self.image_encoder_properties,
                )
                if triton_model_name is not None:
                    kwargs["tritonModelName"] = triton_model_name

                properties = OpenCLIPModelProperties(**kwargs)

                self.assertEqual(expected_effective, properties.effective_name)

    def test_different_batch_sizes_for_encoders(self):
        """Test that text and image encoders can have different batch sizes."""
        text_props = TritonModelProperties(
            name="text-encoder",
            max_batch_size=64,  # Different from image
            sources=["s3://bucket/model.onnx"],
            input=[
                ModelInput(name="input", dims=[1, 77], data_type=DataType.TYPE_INT64)
            ],
            output=[
                ModelOutput(name="output", dims=[1, 512], data_type=DataType.TYPE_FP32)
            ],
        )

        image_props = TritonModelProperties(
            name="image-encoder",
            max_batch_size=8,  # Different from text
            sources=["s3://bucket/model.onnx"],
            input=[
                ModelInput(
                    name="input", dims=[1, 3, 224, 224], data_type=DataType.TYPE_FP32
                )
            ],
            output=[
                ModelOutput(name="output", dims=[1, 512], data_type=DataType.TYPE_FP32)
            ],
        )

        properties = OpenCLIPModelProperties(
            name="test-model",
            type="open_clip",
            dimensions=512,
            triton_text_encoder_properties=text_props,
            triton_image_encoder_properties=image_props,
        )

        self.assertEqual(64, properties.triton_text_encoder_properties.max_batch_size)
        self.assertEqual(8, properties.triton_image_encoder_properties.max_batch_size)


if __name__ == "__main__":
    unittest.main()
