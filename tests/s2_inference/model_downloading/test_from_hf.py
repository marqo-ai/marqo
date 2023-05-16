import unittest
from unittest.mock import MagicMock, patch
from marqo.s2_inference.errors import ModelDownloadError
from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from marqo.s2_inference.model_downloading.from_hf import download_model_from_hf
from huggingface_hub.utils._errors import RepositoryNotFoundError
from marqo.s2_inference.configs import ModelCache


class TestDownloadModelFromHF(unittest.TestCase):
    def setUp(self):
        self.hf_location = HfModelLocation(repo_id="test-repo-id", filename="test-filename")
        self.hf_auth = HfAuth(token="test-token")

    def test_download_model_from_hf_success(self):
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   return_value="model_path") as hf_hub_download_mock:
            result = download_model_from_hf(self.hf_location, self.hf_auth)
        self.assertEqual(result, "model_path")
        hf_hub_download_mock.assert_called_once_with(repo_id="test-repo-id", filename="test-filename", token="test-token", cache_dir=None)

    def test_download_model_from_hf_no_auth(self):
        with patch(
                "marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                return_value="model_path") as hf_hub_download_mock:
            result = download_model_from_hf(self.hf_location)
        self.assertEqual(result, "model_path")
        hf_hub_download_mock.assert_called_once_with(repo_id="test-repo-id", filename="test-filename", cache_dir=None)

    def test_download_model_from_hf_repository_not_found_error(self):
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   side_effect=RepositoryNotFoundError("repo not found")):
            with self.assertRaises(ModelDownloadError):
                download_model_from_hf(self.hf_location, self.hf_auth)

    def test_download_model_from_hf_invalid_location(self):
        invalid_location = HfModelLocation(repo_id="", filename="test-filename")
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   side_effect=RepositoryNotFoundError("repo not found")):
            with self.assertRaises(ModelDownloadError):
                download_model_from_hf(invalid_location, self.hf_auth)

    def test_download_model_from_hf_invalid_auth(self):
        invalid_auth = HfAuth(token="")
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   side_effect=RepositoryNotFoundError("repo not found")):
            with self.assertRaises(ModelDownloadError):
                download_model_from_hf(self.hf_location, invalid_auth)

    def test_download_model_from_hf_unexpected_error(self):
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   side_effect=Exception("Unexpected error")):
            with self.assertRaises(Exception):
                download_model_from_hf(self.hf_location, self.hf_auth)

    def test_download_model_from_hf_with_download_dir(self):
        with patch("marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                   return_value="model_path") as hf_hub_download_mock:
            with patch("marqo.s2_inference.model_downloading.from_hf.logger.warning") as logger_warning_mock:
                result = download_model_from_hf(self.hf_location, self.hf_auth, download_dir="custom_download_dir")
        self.assertEqual(result, "model_path")
        hf_hub_download_mock.assert_called_once_with(repo_id="test-repo-id", filename="test-filename", token="test-token", cache_dir="custom_download_dir")

    def test_download_model_from_hf_no_auth_with_hf_dir(self):
        with patch(
                "marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                return_value="model_path") as hf_hub_download_mock:
            result = download_model_from_hf(self.hf_location, download_dir=ModelCache.hf_cache_path)
        self.assertEqual(result, "model_path")
        hf_hub_download_mock.assert_called_once_with(repo_id="test-repo-id", filename="test-filename",
                                                     cache_dir=ModelCache.hf_cache_path)

    def test_download_model_from_hf_no_auth_with_openclip_dir(self):
        with patch(
                "marqo.s2_inference.model_downloading.from_hf.hf_hub_download",
                return_value="model_path") as hf_hub_download_mock:
            result = download_model_from_hf(self.hf_location, download_dir=ModelCache.clip_cache_path)
        self.assertEqual(result, "model_path")
        hf_hub_download_mock.assert_called_once_with(repo_id="test-repo-id", filename="test-filename",
                                                     cache_dir=ModelCache.clip_cache_path)

