import unittest
import urllib
from unittest.mock import patch, MagicMock
from marqo.s2_inference.processing.custom_clip_utils import (
    download_pretrained_from_s3, download_model, download_pretrained_from_url,
    ModelDownloadError, S3Auth, S3Location, ModelAuth, ModelLocation
)
from marqo.s2_inference.errors import InvalidModelPropertiesError
import tempfile
import os

class TestDownloadModel(unittest.TestCase):
    def test_both_location_and_url_provided(self):
        with self.assertRaises(InvalidModelPropertiesError):
            download_model(repo_location=ModelLocation(s3=S3Location(Bucket="test_bucket", Key="test_key")), url="http://example.com/model.pt")

    def test_neither_location_nor_url_provided(self):
        with self.assertRaises(InvalidModelPropertiesError):
            download_model()

    @patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_s3")
    def test_download_from_s3(self, mock_download_s3):
        mock_download_s3.return_value = "/path/to/model.pt"
        repo_location = ModelLocation(s3=S3Location(Bucket="test_bucket", Key="test_key"))
        auth = ModelAuth(s3=S3Auth(aws_access_key_id="test_access_key", aws_secret_access_key="test_secret_key"))
        model_path = download_model(repo_location=repo_location, auth=auth)

        self.assertEqual(model_path, "/path/to/model.pt")
        mock_download_s3.assert_called_once_with(location=repo_location.s3, auth=auth.s3, download_dir=None)

    @patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url")
    def test_download_from_url(self, mock_download_url):
        mock_download_url.return_value = "/path/to/model.pt"
        url = "http://example.com/model.pt"
        model_path = download_model(url=url)

        self.assertEqual(model_path, "/path/to/model.pt")
        mock_download_url.assert_called_once_with(url=url, cache_dir=None)


class TestDownloadPretrainedFromS3(unittest.TestCase):
    def setUp(self):
        self.s3_location = S3Location(Bucket="test_bucket", Key="test_key")
        self.s3_auth = S3Auth(aws_access_key_id="test_access_key", aws_secret_access_key="test_secret_key")

    @patch("marqo.s2_inference.processing.custom_clip_utils.check_s3_model_already_exists")
    def test_model_exists_locally(self, mock_check_s3_model):
        mock_check_s3_model.return_value = True

        with patch("marqo.s2_inference.processing.custom_clip_utils.get_s3_model_absolute_cache_path"
                   ) as mock_get_abs_path:
            with patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url"
                       ) as mock_download_pretrained_from_url:
                mock_get_abs_path.return_value = "/path/to/model.pt"
                result = download_pretrained_from_s3(location=self.s3_location, auth=self.s3_auth)

        self.assertEqual(result, "/path/to/model.pt")
        mock_download_pretrained_from_url.assert_not_called()
        mock_check_s3_model.assert_called_once_with(location=self.s3_location)

    @patch("marqo.s2_inference.processing.custom_clip_utils.check_s3_model_already_exists")
    @patch("marqo.s2_inference.processing.custom_clip_utils.get_presigned_s3_url")
    def test_model_does_not_exist_locally(self, mock_get_presigned_url, mock_check_s3_model):
        mock_check_s3_model.return_value = False
        mock_get_presigned_url.return_value = "http://example.com/model.pt"

        with patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url"
                   ) as mock_download_pretrained_from_url:
            mock_download_pretrained_from_url.return_value = "/path/to/model.pt"
            result = download_pretrained_from_s3(location=self.s3_location, auth=self.s3_auth)

        self.assertEqual(result, "/path/to/model.pt")
        mock_download_pretrained_from_url.assert_called()
        mock_get_presigned_url.assert_called_once_with(location=self.s3_location, auth=self.s3_auth)

        mock_download_pretrained_from_url.assert_called_once_with(
            url="http://example.com/model.pt",
            cache_dir=None,
            cache_file_name=self.s3_location.Key
        )

    @patch("marqo.s2_inference.processing.custom_clip_utils.check_s3_model_already_exists")
    @patch("marqo.s2_inference.processing.custom_clip_utils.get_presigned_s3_url")
    def test_model_download_raises_403_error(self, mock_get_presigned_url, mock_check_s3_model):
        mock_check_s3_model.return_value = False
        mock_get_presigned_url.return_value = "http://example.com/model.pt"

        with patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url") as mock_download_url:
            mock_download_url.side_effect = urllib.error.HTTPError(url=None, code=403, msg=None, hdrs=None, fp=None)

            with self.assertRaises(ModelDownloadError):
                download_pretrained_from_s3(location=self.s3_location, auth=self.s3_auth)

class TestDownloadPretrainedFromURL(unittest.TestCase):
    def setUp(self):
        self.url = "http://example.com/model.pt"

    @patch("urllib.request.urlopen")
    @patch("os.path.isfile")
    def test_file_exists_locally(self, mock_isfile, mock_urlopen):
        mock_isfile.return_value = True
        with patch("builtins.open", unittest.mock.mock_open()) as mock_open:
            with patch("marqo.s2_inference.processing.custom_clip_utils.tqdm") as mock_tqdm:
                with patch("marqo.s2_inference.processing.custom_clip_utils.ModelCache") as mock_cache:
                    with tempfile.TemporaryDirectory() as temp_cache_dir:
                        mock_cache.clip_cache_path = temp_cache_dir
                        result = download_pretrained_from_url(self.url)

        self.assertEqual(result, os.path.join(temp_cache_dir, 'model.pt'))
        mock_urlopen.assert_not_called()
        mock_isfile.assert_called_once()

    @patch("os.path.isfile")
    @patch("urllib.request.urlopen")
    def test_file_does_not_exist_locally(self, mock_urlopen, mock_isfile):
        mock_isfile.return_value = False
        mock_source = MagicMock()
        mock_source.headers.get.return_value = 0
        mock_source.read.return_value = b''
        mock_urlopen.return_value.__enter__.return_value = mock_source

        with patch("builtins.open", unittest.mock.mock_open()) as mock_open:
            with patch("marqo.s2_inference.processing.custom_clip_utils.tqdm") as mock_tqdm:
                with patch("marqo.s2_inference.processing.custom_clip_utils.ModelCache") as mock_cache:
                    with tempfile.TemporaryDirectory() as temp_cache_dir:
                        mock_cache.clip_cache_path = temp_cache_dir
                        result = download_pretrained_from_url(self.url)

        self.assertEqual(result, os.path.join(temp_cache_dir, 'model.pt'))
        mock_isfile.assert_called_once()
        mock_urlopen.assert_called_once_with(self.url)
        mock_open.assert_called_once_with(os.path.join(temp_cache_dir, 'model.pt'), "wb")
