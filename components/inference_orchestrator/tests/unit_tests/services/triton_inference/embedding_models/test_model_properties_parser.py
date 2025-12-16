import unittest

from inference_orchestrator.services.errors import InvalidModelPropertiesError
from inference_orchestrator.services.triton_inference.embedding_models import (
    HuggingFaceModel,
    OpenCLIPModel,
    RandomModel,
)
from inference_orchestrator.services.triton_inference.embedding_models.model_properties_parser import (
    get_model_loader,
)


class TestGetModelLoader(unittest.TestCase):
    """Tests for get_model_loader function."""

    def test_get_model_loader_hugging_face(self):
        """Test that 'hf' type returns HuggingFaceModel."""
        model_properties = {"type": "hf"}
        result = get_model_loader(model_properties)
        self.assertEqual(HuggingFaceModel, result)

    def test_get_model_loader_open_clip(self):
        """Test that 'open_clip' type returns OpenCLIPModel."""
        model_properties = {"type": "open_clip"}
        result = get_model_loader(model_properties)
        self.assertEqual(OpenCLIPModel, result)

    def test_get_model_loader_random(self):
        """Test that 'random' type returns RandomModel."""
        model_properties = {"type": "random"}
        result = get_model_loader(model_properties)
        self.assertEqual(RandomModel, result)

    def test_get_model_loader_invalid_type_raises_error(self):
        """Test that unsupported model type raises InvalidModelPropertiesError."""
        test_cases = [
            ("invalid_type", {"type": "invalid_type"}),
            ("unknown", {"type": "unknown"}),
            ("bert", {"type": "bert"}),
        ]

        for msg, model_properties in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    get_model_loader(model_properties)
                self.assertIn("Unsupported model type", str(context.exception))
                self.assertIn(model_properties["type"], str(context.exception))

    def test_get_model_loader_missing_type_raises_error(self):
        """Test that missing 'type' key raises InvalidModelPropertiesError."""
        model_properties = {}
        with self.assertRaises(InvalidModelPropertiesError) as context:
            get_model_loader(model_properties)
        self.assertIn("Unsupported model type", str(context.exception))

    def test_get_model_loader_none_type_raises_error(self):
        """Test that None type raises InvalidModelPropertiesError."""
        model_properties = {"type": None}
        with self.assertRaises(InvalidModelPropertiesError) as context:
            get_model_loader(model_properties)
        self.assertIn("Unsupported model type", str(context.exception))
