from marqo.s2_inference.model_downloading.from_s3 import (
    get_presigned_s3_url,
    get_s3_model_absolute_cache_path,
    check_s3_model_already_exists,
    get_s3_model_cache_filename,
)
from botocore.exceptions import NoCredentialsError
from marqo.s2_inference.configs import ModelCache
import unittest
import botocore
from unittest.mock import patch
from marqo.s2_inference.errors import ModelDownloadError
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location


class TestModelAuthEdgeCases(unittest.TestCase):
    def setUp(self):
        self.s3_location = S3Location(Bucket="test-bucket", Key="test-key")
        self.s3_auth = S3Auth(aws_access_key_id="test-access-key", aws_secret_access_key="test-secret-key")

    def test_get_presigned_s3_url_no_credentials_error(self):
        with patch("boto3.client") as boto3_client_mock:
            boto3_client_mock.return_value.generate_presigned_url.side_effect = NoCredentialsError
            with self.assertRaises(ModelDownloadError):
                get_presigned_s3_url(self.s3_location, self.s3_auth)

    def test_get_presigned_s3_url_invalid_location(self):
        invalid_location = S3Location(Bucket="", Key="")
        with self.assertRaises(botocore.exceptions.ParamValidationError):
            get_presigned_s3_url(invalid_location, self.s3_auth)

    def test_get_s3_model_absolute_cache_path_empty_key(self):
        empty_key_location = S3Location(Bucket="test-bucket", Key="")
        with patch("os.path.expanduser", return_value="some_cache_path"):
            result = get_s3_model_absolute_cache_path(empty_key_location)
        self.assertEqual(result, "some_cache_path/")

    def test_check_s3_model_already_exists_empty_key(self):
        empty_key_location = S3Location(Bucket="test-bucket", Key="")
        with patch("os.path.isfile", return_value=True):
            result = check_s3_model_already_exists(empty_key_location)
        self.assertTrue(result)

    def test_check_s3_model_already_exists_no_file(self):
        with patch("os.path.isfile", return_value=False):
            result = check_s3_model_already_exists(self.s3_location)
        self.assertFalse(result)

    def test_get_s3_model_cache_filename_empty_key(self):
        empty_key_location = S3Location(Bucket="test-bucket", Key="")
        result = get_s3_model_cache_filename(empty_key_location)
        self.assertEqual(result, "")

    def test_get_s3_model_absolute_cache_path_invalid_cache_dir(self):
        with patch("os.path.expanduser", return_value=""):
            result = get_s3_model_absolute_cache_path(self.s3_location)
        self.assertEqual(result, "test-key")

    def test_get_s3_model_absolute_cache_path_cache_dir_not_expanded(self):
        with patch("os.path.expanduser", side_effect=lambda x: x):
            with patch("os.path.join", side_effect=lambda x, y: f"{x}/{y}"):
                result = get_s3_model_absolute_cache_path(self.s3_location)
        self.assertEqual(result, f"{ModelCache.clip_cache_path}/test-key")

    def test_check_s3_model_already_exists_os_error(self):
        with patch("os.path.isfile", side_effect=OSError("Test OSError")):
            with self.assertRaises(OSError):
                check_s3_model_already_exists(self.s3_location)

    def test_get_s3_model_cache_filename_with_directory(self):
        location_with_directory = S3Location(Bucket="test-bucket", Key="models/test-key")
        result = get_s3_model_cache_filename(location_with_directory)
        self.assertEqual(result, "test-key")
