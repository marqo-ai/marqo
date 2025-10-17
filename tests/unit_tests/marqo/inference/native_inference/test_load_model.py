from unittest import TestCase
from unittest.mock import MagicMock, patch

from marqo.core.inference.api.exceptions import ModelError
from marqo.inference.native_inference.load_model import _load_model


class TestLoadModel(TestCase):
    def test_supported_model_types(self):
        supported_model_types = [
            "multilingual_clip",
            "open_clip",
            "hf_stella",
            "hf",
            "languagebind",
            "no_model",
            "random",
        ]

        for model_type in supported_model_types:
            with self.subTest(model_type=model_type):
                with patch(
                    "marqo.inference.native_inference.load_model._get_model_loader"
                ) as mock_get_model_loader:
                    mock_get_model_loader.return_value = MagicMock()
                    _ = _load_model(
                        model_name="test",
                        model_properties={"type": model_type},
                        device="cpu",
                        model_auth=None,
                        calling_func="unit_test",
                    )
                mock_get_model_loader.assert_called_once()

    def test_unsupported_model_types(self):
        unsupported_model_types = ["sbert", "test", "clip", "sbert_onnx", "..."]

        for model_type in unsupported_model_types:
            with self.subTest(model_type=model_type):
                with patch(
                    "marqo.inference.native_inference.load_model._get_model_loader"
                ) as mock_get_model_loader:
                    mock_get_model_loader.return_value = MagicMock()
                    with self.assertRaises(ModelError):
                        _ = _load_model(
                            model_name="test",
                            model_properties={"type": model_type},
                            device="cpu",
                            model_auth=None,
                            calling_func="unit_test",
                        )
                mock_get_model_loader.assert_not_called()
