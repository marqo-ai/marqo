from marqo.s2_inference.s2_inference import _load_model
from unittest import TestCase
from unittest.mock import patch, MagicMock
from urllib3.exceptions import ReadTimeoutError


class TestModelLoad(TestCase):
    @patch("marqo.s2_inference.s2_inference._get_model_loader")
    def test_load_model_successfully(self, mock_get_model_loader):
        mock_loader = MagicMock()
        mock_get_model_loader.return_value = mock_loader
        model_properties = {
            "name": "dummy_model",
            "dimensions": 512,
        }
        _load_model("dummy_model", model_properties, "cpu", calling_func="unit_test")

        mock_get_model_loader.assert_called_once_with("dummy_model", model_properties)
        mock_loader.return_value.load.assert_called_once()

    @patch("marqo.s2_inference.s2_inference._get_model_loader")
    @patch("time.sleep")
    def test_load_model_after_retry(self, mock_sleep, mock_get_model_loader):
        mock_loader = MagicMock()
        mock_loader.return_value.load.side_effect = [
            ReadTimeoutError(pool=None, url=None, message=None),
            None
        ]
        mock_get_model_loader.return_value = mock_loader
        model_properties = {
            "name": "dummy_model",
            "dimensions": 512,
        }
        _load_model("dummy_model", model_properties, "cpu", calling_func="unit_test")

        mock_get_model_loader.assert_called_once_with("dummy_model", model_properties)
        mock_sleep.assert_called_once()
        self.assertEqual(mock_loader.return_value.load.call_count, 2)

    @patch("marqo.s2_inference.s2_inference._get_model_loader")
    @patch("time.sleep")
    def test_load_model_retries_exhausted_error_raised(self, mock_sleep, mock_get_model_loader):
        mock_loader = MagicMock()
        mock_loader.return_value.load.side_effect = ReadTimeoutError(pool=None, url=None, message=None)
        mock_get_model_loader.return_value = mock_loader
        model_properties = {
            "name": "dummy_model",
            "dimensions": 512,
        }
        with self.assertRaises(ReadTimeoutError):
            _load_model("dummy_model", model_properties, "cpu", calling_func="unit_test")

        mock_get_model_loader.assert_called_once_with("dummy_model", model_properties)
        self.assertEqual(mock_loader.return_value.load.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("marqo.s2_inference.s2_inference._get_model_loader")
    @patch("time.sleep")
    def test_load_model_unexpected_error_no_retries(self, mock_sleep, mock_get_model_loader):
        mock_loader = MagicMock()
        mock_loader.return_value.load.side_effect = ValueError("Unexpected error")
        mock_get_model_loader.return_value = mock_loader
        model_properties = {
            "name": "dummy_model",
            "dimensions": 512,
        }
        with self.assertRaises(ValueError):
            _load_model("dummy_model", model_properties, "cpu", calling_func="unit_test")

        mock_get_model_loader.assert_called_once_with("dummy_model", model_properties)
        mock_loader.return_value.load.assert_called_once()
        self.assertEqual(mock_sleep.call_count, 0)
