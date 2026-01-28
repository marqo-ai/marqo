import os
import unittest
from unittest.mock import patch

from pytest import mark

from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.tensor_search.models.external_apis.hf import HfAuth
from marqo.tensor_search.models.external_apis.s3 import S3Auth
from marqo.tensor_search.models.private_models import ModelAuth


@mark.largemodel
class TestLanguagebindModelLoad(unittest.TestCase):
    """
    Test different loading methods of LanguagebindModel.
    """
    AUDIO_HF_REPO_NAME = "Marqo/LanguageBind_Audio_FT"
    IMAGE_HF_REPO_NAME = "Marqo/LanguageBind_Image"
    VIDEO_HF_REPO_NAME = "Marqo/LanguageBind_Video_V1.5_FT"
    PRIVATE_VIDEO_HF_REPO = "Marqo/private-LanguageBind_Video_V1.5_FT"
    AUDIO_URL = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/LanguageBind_Audio_FT.zip"

    aws_access_key_id = os.getenv("PRIVATE_MODEL_TESTS_AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.getenv("PRIVATE_MODEL_TESTS_AWS_SECRET_ACCESS_KEY", None)
    hf_token = os.getenv("PRIVATE_MODEL_TESTS_HF_TOKEN", None)

    def test_loading_languagebind_model_from_a_hf_repo(self):
        """A test for loading a LanguagebindModel from a public Hugging Face repo."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"hf": {"repoId": self.IMAGE_HF_REPO_NAME}},
                "audio": {"hf": {"repoId": self.AUDIO_HF_REPO_NAME}},
                "video": {"hf": {"repoId": self.VIDEO_HF_REPO_NAME}}
            }
        }

        model = LanguagebindModel(device="cuda", model_properties=model_properties)
        model.load()

    @unittest.skip(reason="we are hitting limit hf private repos, no user is using models in private hf repo")
    def test_loading_languagebind_model_from_a_private_hf_repo(self):
        """A test for loading a LanguagebindModel from a private Hugging Face repo."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "video"],
            "modelLocation": {
                "video": {"hf": {"repoId": self.PRIVATE_VIDEO_HF_REPO}}
            }
        }

        mode_auth = ModelAuth(hf=HfAuth(token=self.hf_token))

        model = LanguagebindModel(device="cuda", model_properties=model_properties, model_auth=mode_auth)
        model.load()

    def test_loading_languagebind_model_from_a_url(self):
        """A test for loading a LanguagebindModel from a URL."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "audio"],
            "modelLocation": {
                "audio": {"url": self.AUDIO_URL}
            }
        }

        model = LanguagebindModel(device="cuda", model_properties=model_properties)
        model.load()

    def test_loading_languagebind_model_from_a_zip_on_s3(self):
        """A test for loading a LanguagebindModel from a zip file on S3."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Image.zip"}},
                "audio": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Audio_FT.zip"}},
                "video": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Video_V1.5_FT.zip"}},
            }
        }

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_secret_access_key=self.aws_secret_access_key,
                aws_access_key_id=self.aws_access_key_id)
        )
        model = LanguagebindModel(
            device="cuda", model_properties=model_properties, model_auth=model_auth
        )
        model.load()

    def test_loading_languagebind_model_from_a_zip_on_s3_with_role(self):
        """A test for loading a LanguagebindModel from a zip file on S3 using a role."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Image.zip"}},
                "audio": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Audio_FT.zip"}},
                "video": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Video_V1.5_FT.zip"}},
            }
        }

        model = LanguagebindModel(
            device="cuda", model_properties=model_properties
        )
        raised_exception = RuntimeError("Stop here")
        with (patch("marqo.inference.model_download.model_download.get_presigned_s3_url",side_effect=raised_exception)
              as mock_presigned_url):
            with patch("marqo.inference.model_download.model_download.check_s3_model_already_exists", return_value=False):
                with self.assertRaises(RuntimeError) as context:
                    model.load()

        # Ensure that the get_presigned_s3_url function was called thus role based access was attempted
        mock_presigned_url.assert_called_once()