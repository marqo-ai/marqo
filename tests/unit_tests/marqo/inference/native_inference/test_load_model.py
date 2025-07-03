from marqo.inference.native_inference.load_model import load_model, _load_model
from unittest import TestCase
from unittest.mock import patch, Mock, MagicMock
from marqo.core.inference.api.exceptions import ModelError
from marqo.s2_inference.errors import InvalidModelPropertiesError, ModelLoadError


class TestLoadModel(TestCase):

    def test_supported_model_types(self):
        supported_model_types = [
            "multilingual_clip",
            "open_clip",
            "hf_stella",
            "hf",
            "languagebind",
            "no_model",
            "random"
        ]

        for model_type in supported_model_types:
            with self.subTest(model_type=model_type):
                with patch("marqo.inference.native_inference.load_model._get_model_loader") as mock_get_model_loader:
                    mock_get_model_loader.return_value = MagicMock()
                    _ = _load_model(
                        model_name="test", model_properties={"type": model_type},
                        device="cpu", model_auth=None, calling_func="unit_test"
                    )
                mock_get_model_loader.assert_called_once()

    def test_unsupported_model_types(self):
        unsupported_model_types = [
            "sbert",
            "test",
            "clip",
            "sbert_onnx",
            "..."
        ]

        for model_type in unsupported_model_types:
            with self.subTest(model_type=model_type):
                with patch("marqo.inference.native_inference.load_model._get_model_loader") as mock_get_model_loader:
                    mock_get_model_loader.return_value = MagicMock()
                    with self.assertRaises(ModelError):
                        _ = _load_model(
                            model_name="test", model_properties={"type": model_type},
                            device="cpu", model_auth=None, calling_func="unit_test"
                        )
                mock_get_model_loader.assert_not_called()

    def test_retry_mechanism_for_model_loading(self):
        with (patch("marqo.inference.native_inference.embedding_models.hugging_face_model.HuggingFaceModel.load")
              as mock_load_model, \
             patch("time.sleep") as mock_sleep):
            mock_load_model.side_effect = InvalidModelPropertiesError("Can't load model")
            with self.assertRaises(ModelLoadError):
                _ = load_model(
                    model_name= "hf/all-MiniLM-L6-v2",
                    model_properties={
                        "name": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimensions": 384,
                        "tokens": 256,
                        "type": "hf",
                        "notes": ""
                    },
                    model_auth=None,
                    device="cpu"
                )
            self.assertEqual(mock_load_model.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)  # Should sleep 2 times (between 3 attempts)