import logging
from unittest.mock import patch

import pytest
from botocore.config import Config
from botocore.exceptions import NoCredentialsError

from marqo.inference.model_download.download_model_from_s3 import get_presigned_s3_url
from marqo.s2_inference.errors import ModelDownloadError
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location


class TestGetPresignedS3Url:
    @pytest.mark.parametrize(
        "env_value,expected",
        [
            ("TRUE", True),
            ("true", True),
            ("False", False),
            ("", False),
        ],
    )
    def test_dualstack_flag_is_parsed_case_insensitive(self, caplog, env_value, expected):
        location = S3Location(Bucket="test-bucket", Key="path/to/model.pt")
        auth = S3Auth(aws_access_key_id="akid", aws_secret_access_key="secret")

        with patch("marqo.tensor_search.utils.read_env_vars_and_defaults", return_value=env_value):
            with patch("boto3.client") as boto3_client_mock:
                boto3_client_mock.return_value.generate_presigned_url.return_value = "https://example"

                with caplog.at_level(logging.INFO):
                    get_presigned_s3_url(location=location, auth=auth)

                _, kwargs = boto3_client_mock.call_args
                assert isinstance(kwargs.get("config"), Config)
                assert kwargs["config"].use_dualstack_endpoint is expected
                assert "Using dual stack endpoint for S3" in caplog.text

    def test_raises_model_download_error_on_no_credentials(self):
        location = S3Location(Bucket="test-bucket", Key="path/to/model.pt")
        auth = S3Auth(aws_access_key_id="akid", aws_secret_access_key="secret")

        with patch("marqo.tensor_search.utils.read_env_vars_and_defaults", return_value="FALSE"):
            with patch("boto3.client") as boto3_client_mock:
                boto3_client_mock.return_value.generate_presigned_url.side_effect = NoCredentialsError()

                with pytest.raises(ModelDownloadError):
                    get_presigned_s3_url(location=location, auth=auth)


