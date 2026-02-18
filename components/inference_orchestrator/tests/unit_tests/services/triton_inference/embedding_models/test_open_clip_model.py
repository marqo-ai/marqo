import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.errors import (
    InternalServerError,
    InvalidModelPropertiesError,
)
from inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model import (
    OpenCLIPModel,
    OpenCLIPPreprocessor,
)


class TestOpenCLIPPreprocessor(unittest.TestCase):
    """Test suite for OpenCLIPPreprocessor class.

    These tests verify the preprocessing functionality for both text and image modalities
    without loading any real models.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = MagicMock()
        self.mock_image_preprocessor = MagicMock()
        self.preprocessor = OpenCLIPPreprocessor(
            tokenizer=self.mock_tokenizer,
            image_preprocessor=self.mock_image_preprocessor,
        )

    def test_preprocess_text_modality_returns_input_as_is(self):
        """Test that text preprocessing returns input strings unchanged."""
        inputs = ["hello world", "test text", "another string"]

        result = self.preprocessor.preprocess(inputs, Modality.TEXT)

        self.assertEqual(inputs, result)
        # Tokenizer should not be called during preprocess, only during encoding
        self.mock_tokenizer.assert_not_called()

    def test_preprocess_image_modality_applies_image_preprocessor(self):
        """Test that image preprocessing applies the image preprocessor to each image."""
        # Create mock images
        mock_images = [MagicMock(spec=Image.Image) for _ in range(3)]

        # Mock preprocessor returns tensors that need unsqueeze
        mock_tensors = [torch.randn(3, 224, 224) for _ in range(3)]
        self.mock_image_preprocessor.side_effect = mock_tensors

        result = self.preprocessor.preprocess(mock_images, Modality.IMAGE)

        # Verify preprocessor was called for each image
        self.assertEqual(3, self.mock_image_preprocessor.call_count)
        for i, mock_image in enumerate(mock_images):
            self.mock_image_preprocessor.assert_any_call(mock_image)

        # Verify result has correct length
        self.assertEqual(3, len(result))

        # Verify each result is a tensor with batch dimension added
        for i, tensor in enumerate(result):
            self.assertTrue(torch.is_tensor(tensor))
            # Shape should be (1, 3, 224, 224) after unsqueeze(0)
            self.assertEqual(4, len(tensor.shape))

    def test_preprocess_unsupported_modality_raises_error(self):
        """Test that unsupported modality raises InternalServerError."""
        inputs = ["test"]

        with self.assertRaises(InternalServerError) as context:
            self.preprocessor.preprocess(inputs, Modality.AUDIO)

        error_message = str(context.exception).lower()
        self.assertIn("unsupported modality", error_message)
        self.assertIn("audio", error_message)

    def test_tokenize_text_returns_inputs_unchanged(self):
        """Test that _tokenize_text returns inputs as-is."""
        inputs = ["text1", "text2"]

        result = self.preprocessor._tokenize_text(inputs)

        self.assertEqual(inputs, result)

    def test_preprocess_image_with_single_image(self):
        """Test image preprocessing with a single image."""
        mock_image = MagicMock(spec=Image.Image)
        mock_tensor = torch.randn(3, 224, 224)
        self.mock_image_preprocessor.return_value = mock_tensor

        result = self.preprocessor._preprocess_image([mock_image])

        self.assertEqual(1, len(result))
        self.assertTrue(torch.is_tensor(result[0]))
        self.mock_image_preprocessor.assert_called_once_with(mock_image)


class TestOpenCLIPModel(unittest.TestCase):
    """Test suite for OpenCLIPModel class.

    These tests verify the OpenCLIPModel functionality without loading real models.
    All external dependencies (open_clip, model downloads, triton) are mocked.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model_management_client = MagicMock()
        self.mock_triton_client = MagicMock()

        # Valid model properties for testing
        self.valid_hf_model_properties = {
            "name": "hf-hub:test/model",
            "type": "open_clip",
            "dimensions": 512,
            "imagePreprocessor": "SigLIP",
            "tritonTextEncoderProperties": {
                "name": "text-encoder",
                "sources": ["s3://bucket/text-encoder/model.onnx"],
                "input": [{"name": "input", "dims": [1, 77], "dataType": "TYPE_INT64"}],
                "output": [
                    {"name": "output", "dims": [1, 512], "dataType": "TYPE_FP32"}
                ],
                "maxBatchSize": 32,
            },
            "tritonImageEncoderProperties": {
                "name": "image-encoder",
                "sources": ["s3://bucket/image-encoder/model.onnx"],
                "input": [
                    {"name": "input", "dims": [1, 3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [
                    {"name": "output", "dims": [1, 512], "dataType": "TYPE_FP32"}
                ],
                "maxBatchSize": 16,
            },
        }

        self.valid_openclip_model_properties = {
            "name": "open_clip/ViT-B-32/openai",
            "type": "open_clip",
            "dimensions": 512,
            "imagePreprocessor": "SigLIP",
            "tritonTextEncoderProperties": {
                "name": "text-encoder",
                "sources": ["s3://bucket/text-encoder/model.onnx"],
                "input": [{"name": "input", "dims": [1, 77], "dataType": "TYPE_INT64"}],
                "output": [
                    {"name": "output", "dims": [1, 512], "dataType": "TYPE_FP32"}
                ],
                "maxBatchSize": 32,
            },
            "tritonImageEncoderProperties": {
                "name": "image-encoder",
                "sources": ["s3://bucket/image-encoder/model.onnx"],
                "input": [
                    {"name": "input", "dims": [1, 3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [
                    {"name": "output", "dims": [1, 512], "dataType": "TYPE_FP32"}
                ],
                "maxBatchSize": 16,
            },
        }

    def test_init_with_valid_hf_properties(self):
        """Test OpenCLIPModel initialization with valid HF hub properties."""
        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        self.assertIsNotNone(model.model_properties)
        self.assertEqual("hf-hub:test/model", model.model_properties.name)
        self.assertEqual(512, model.model_properties.dimensions)

    def test_init_with_valid_openclip_properties(self):
        """Test OpenCLIPModel initialization with valid OpenCLIP registry properties."""
        model = OpenCLIPModel(
            model_properties=self.valid_openclip_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        self.assertIsNotNone(model.model_properties)
        self.assertEqual("open_clip/ViT-B-32/openai", model.model_properties.name)

    def test_init_with_invalid_properties_raises_error(self):
        """Test that invalid model properties raise InvalidModelPropertiesError."""
        invalid_properties = {
            "name": "hf-hub:test/model",
            "type": "wrong_type",  # Invalid type
            "dimensions": 512,
        }

        with self.assertRaises(InvalidModelPropertiesError) as context:
            OpenCLIPModel(
                model_properties=invalid_properties,
                model_management_client=self.mock_model_management_client,
                triton_client=self.mock_triton_client,
            )

        self.assertIn("Invalid model properties", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_load_from_hf_repo(self, mock_open_clip):
        """Test loading model from HuggingFace hub."""
        # Mock open_clip functions
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        # Execute load
        model.load()

        # Verify model loading was called
        mock_open_clip.create_model_and_transforms.assert_called_once()
        call_kwargs = mock_open_clip.create_model_and_transforms.call_args[1]
        self.assertEqual("hf-hub:test/model", call_kwargs["model_name"])
        self.assertEqual("cpu", call_kwargs["device"])

        # Verify tokenizer loading
        mock_open_clip.get_tokenizer.assert_called_once()

        # Verify Triton models were loaded
        self.assertEqual(2, self.mock_model_management_client.load_model.call_count)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_load_from_openclip_registry(self, mock_open_clip):
        """Test loading model from OpenCLIP registry."""
        # Mock open_clip functions
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_openclip_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        # Execute load
        model.load()

        # Verify model loading was called with parsed name
        mock_open_clip.create_model_and_transforms.assert_called_once()
        call_kwargs = mock_open_clip.create_model_and_transforms.call_args[1]
        self.assertEqual("ViT-B-32", call_kwargs["model_name"])  # architecture
        self.assertEqual("openai", call_kwargs["pretrained"])  # pretrained

        # Verify tokenizer loading with architecture
        mock_open_clip.get_tokenizer.assert_called_once_with("ViT-B-32")

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_load_uses_effective_name_from_triton_model_name(self, mock_open_clip):
        """Test that loading uses tritonModelName (via effective_name) when set.

        During migration, the Vespa-stored properties keep the old 'name' for cache key stability
        and add 'tritonModelName' with the new value for the inference orchestrator.
        """
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Properties with old name but new tritonModelName
        properties_with_triton_name = self.valid_hf_model_properties.copy()
        properties_with_triton_name["name"] = "ViT-B-16-SigLIP"  # Old name
        properties_with_triton_name["tritonModelName"] = "hf-hub:timm/ViT-B-16-SigLIP"  # New name

        model = OpenCLIPModel(
            model_properties=properties_with_triton_name,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model.load()

        # Verify create_model_and_transforms used the tritonModelName (effective_name)
        call_kwargs = mock_open_clip.create_model_and_transforms.call_args[1]
        self.assertEqual("hf-hub:timm/ViT-B-16-SigLIP", call_kwargs["model_name"])

        # Verify tokenizer used the tritonModelName (effective_name)
        mock_open_clip.get_tokenizer.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_load_openclip_registry_uses_effective_name_from_triton_model_name(self, mock_open_clip):
        """Test that loading from open_clip registry uses tritonModelName when set."""
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Properties with old name but new tritonModelName pointing to open_clip registry
        properties_with_triton_name = self.valid_openclip_model_properties.copy()
        properties_with_triton_name["name"] = "old-name"
        properties_with_triton_name["tritonModelName"] = "open_clip/ViT-L-14/laion2b_s32b_b82k"

        model = OpenCLIPModel(
            model_properties=properties_with_triton_name,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model.load()

        # Verify architecture and pretrained were extracted from tritonModelName
        call_kwargs = mock_open_clip.create_model_and_transforms.call_args[1]
        self.assertEqual("ViT-L-14", call_kwargs["model_name"])
        self.assertEqual("laion2b_s32b_b82k", call_kwargs["pretrained"])

        # Verify tokenizer used the architecture from tritonModelName
        mock_open_clip.get_tokenizer.assert_called_once_with("ViT-L-14")

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_load_tokenizer_from_checkpoint_uses_effective_name(self, mock_open_clip):
        """Test that _load_tokenizer_from_checkpoint uses effective_name for tokenizer loading.

        Tests both HF hub prefix and non-HF prefix paths, with and without tritonModelName.
        """
        mock_tokenizer = MagicMock()
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        test_cases = [
            (
                "hf-hub name without tritonModelName",
                {"name": "hf-hub:org/model"},
                "hf-hub:org/model",
            ),
            (
                "hf-hub name via tritonModelName",
                {"name": "old-name", "tritonModelName": "hf-hub:org/new-model"},
                "hf-hub:org/new-model",
            ),
            (
                "non-hf name without tritonModelName",
                {"name": "ViT-B/32"},
                "ViT-B-32",  # '/' replaced with '-'
            ),
            (
                "non-hf name via tritonModelName",
                {"name": "old-name", "tritonModelName": "ViT-L/14"},
                "ViT-L-14",  # '/' replaced with '-'
            ),
        ]

        for test_name, name_overrides, expected_tokenizer_arg in test_cases:
            with self.subTest(msg=test_name):
                mock_open_clip.get_tokenizer.reset_mock()

                props = self.valid_hf_model_properties.copy()
                props.update(name_overrides)

                model = OpenCLIPModel(
                    model_properties=props,
                    model_management_client=self.mock_model_management_client,
                    triton_client=self.mock_triton_client,
                )

                result = model._load_tokenizer_from_checkpoint()

                mock_open_clip.get_tokenizer.assert_called_once_with(expected_tokenizer_arg)
                self.assertIs(mock_tokenizer, result)

    def test_load_with_invalid_prefix_raises_error(self):
        """Test that loading model with invalid prefix raises error."""
        invalid_properties = self.valid_hf_model_properties.copy()
        invalid_properties["name"] = "invalid-prefix:test/model"

        model = OpenCLIPModel(
            model_properties=invalid_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        with self.assertRaises(InvalidModelPropertiesError) as context:
            model.load()

        self.assertIn("cannot load", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_check_loaded_components_raises_error_if_model_not_loaded(
        self, mock_open_clip
    ):
        """Test that _check_loaded_components raises error if model is None."""
        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        # Don't load the model
        model.model = None
        model.tokenizer = MagicMock()
        model.image_preprocessor = MagicMock()

        with self.assertRaises(RuntimeError) as context:
            model._check_loaded_components()

        self.assertIn("model is not loaded", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_check_loaded_components_raises_error_if_tokenizer_not_loaded(
        self, mock_open_clip
    ):
        """Test that _check_loaded_components raises error if tokenizer is None."""
        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model.model = True
        model.tokenizer = None
        model.image_preprocessor = MagicMock()

        with self.assertRaises(RuntimeError) as context:
            model._check_loaded_components()

        self.assertIn("tokenizer is not loaded", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_check_loaded_components_raises_error_if_image_preprocessor_not_loaded(
        self, mock_open_clip
    ):
        """Test that _check_loaded_components raises error if image_preprocessor is None."""
        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model.model = True
        model.tokenizer = MagicMock()
        model.image_preprocessor = None

        with self.assertRaises(RuntimeError) as context:
            model._check_loaded_components()

        self.assertIn(
            "image preprocessor is not loaded", str(context.exception).lower()
        )

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_get_preprocessor_returns_openclip_preprocessor(self, mock_open_clip):
        """Test that get_preprocessor returns the correct preprocessor."""
        # Mock open_clip functions
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        preprocessor = model.get_preprocessor()

        self.assertIsInstance(preprocessor, OpenCLIPPreprocessor)
        self.assertIs(preprocessor.tokenizer, mock_tokenizer)
        self.assertIs(preprocessor.image_preprocessor, mock_preprocess)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_text(self, mock_open_clip):
        """Test encoding text inputs."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Mock tokenizer output
        mock_tokenized = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mock_tokenizer.return_value = mock_tokenized

        # Mock triton response
        mock_response = MagicMock()
        embeddings_output = np.random.rand(2, 512).astype(np.float32)
        mock_response.as_numpy.return_value = embeddings_output
        self.mock_triton_client.encode.return_value = mock_response

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute
        text_inputs = ["hello", "world"]
        result = model.encode_text(text_inputs, normalize=True)

        # Verify tokenizer was called
        mock_tokenizer.assert_called_once_with(text_inputs)

        # Verify triton client was called
        self.mock_triton_client.encode.assert_called_once()
        call_kwargs = self.mock_triton_client.encode.call_args[1]
        self.assertEqual("text-encoder", call_kwargs["model_name"])

        # Verify result
        self.assertEqual(2, len(result))
        for embedding in result:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(512, len(embedding))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_image(self, mock_open_clip):
        """Test encoding image inputs."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Mock triton response
        mock_response = MagicMock()
        embeddings_output = np.random.rand(2, 512).astype(np.float32)
        mock_response.as_numpy.return_value = embeddings_output
        self.mock_triton_client.encode.return_value = mock_response

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute with mock tensors
        image_tensors = [
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
        ]
        result = model.encode_image(image_tensors, normalize=True)

        # Verify triton client was called
        self.mock_triton_client.encode.assert_called_once()
        call_kwargs = self.mock_triton_client.encode.call_args[1]
        self.assertEqual("image-encoder", call_kwargs["model_name"])

        # Verify result
        self.assertEqual(2, len(result))
        for embedding in result:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(512, len(embedding))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_routes_to_correct_method_for_text(self, mock_open_clip):
        """Test that encode() routes to encode_text for TEXT modality."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Mock encode_text
        model.encode_text = MagicMock(return_value=[np.array([1.0] * 512)])

        # Execute
        _ = model.encode(["test"], Modality.TEXT, normalize=True)

        # Verify encode_text was called
        model.encode_text.assert_called_once_with(["test"], normalize=True)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_routes_to_correct_method_for_image(self, mock_open_clip):
        """Test that encode() routes to encode_image for IMAGE modality."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Mock encode_image
        model.encode_image = MagicMock(return_value=[np.array([1.0] * 512)])

        # Execute
        images = [torch.randn(1, 3, 224, 224)]
        _ = model.encode(images, Modality.IMAGE, normalize=True)

        # Verify encode_image was called
        model.encode_image.assert_called_once_with(images, normalize=True)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_with_unsupported_modality_raises_error(self, mock_open_clip):
        """Test that encode raises error for unsupported modality."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute and verify
        with self.assertRaises(InternalServerError) as context:
            model.encode(["test"], Modality.AUDIO, normalize=True)

        self.assertIn("unsupported modality", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_text_without_normalization(self, mock_open_clip):
        """Test encoding text without normalization."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Mock tokenizer output
        mock_tokenized = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = mock_tokenized

        # Mock triton response with unnormalized embeddings
        mock_response = MagicMock()
        embeddings_output = np.array([[2.0] * 512], dtype=np.float32)
        mock_response.as_numpy.return_value = embeddings_output.copy()
        self.mock_triton_client.encode.return_value = mock_response

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute without normalization
        result = model.encode_text(["test"], normalize=False)

        # Verify result is not normalized (values should match original)
        self.assertEqual(1, len(result))
        # Since normalize=False, values should be larger than 1 in magnitude
        self.assertTrue(np.any(np.abs(result[0]) > 1.0))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_encode_text_shape_mismatch_raises_error(self, mock_open_clip):
        """Test that shape mismatch in encode_text raises InternalServerError."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # Mock tokenizer output
        mock_tokenized = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = mock_tokenized

        # Mock triton response with wrong shape
        mock_response = MagicMock()
        embeddings_output = np.array(
            [[0.1, 0.2, 0.3]]
        )  # Wrong dimensions (3 instead of 512)
        mock_response.as_numpy.return_value = embeddings_output
        self.mock_triton_client.encode.return_value = mock_response

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute and verify error
        with self.assertRaises(InternalServerError) as context:
            model.encode_text(["test"], normalize=True)

        self.assertIn("shape", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model.open_clip"
    )
    def test_unload_calls_model_management_client(self, mock_open_clip):
        """Test that unload calls model_management_client for both encoders."""
        # Setup
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        model = OpenCLIPModel(
            model_properties=self.valid_hf_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute unload
        model.unload(remove_files=True)

        # Verify unload_model was called for both encoders
        self.assertEqual(2, self.mock_model_management_client.unload_model.call_count)

        # Verify both text and image encoders were unloaded
        calls = self.mock_model_management_client.unload_model.call_args_list
        encoder_names = {call[0][0] for call in calls}
        self.assertIn("text-encoder", encoder_names)
        self.assertIn("image-encoder", encoder_names)

        # Verify remove_files was passed
        for call in calls:
            self.assertEqual(True, call[1]["remove_files"])


if __name__ == "__main__":
    unittest.main()
