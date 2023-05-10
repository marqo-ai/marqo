import copy
import itertools
import PIL
import requests.exceptions
from marqo.s2_inference import clip_utils, types
import unittest
from unittest import mock
import requests
from marqo.s2_inference.clip_utils import CLIP, download_model
from marqo.tensor_search.enums import ModelProperties
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
from unittest.mock import patch
import pytest
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
from marqo.tensor_search.models.private_models import S3Auth, S3Location


class TestEncoding(unittest.TestCase):

    def test_load_image_from_path_timeout(self):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # should be fine on regular timeout:
        img = clip_utils.load_image_from_path(good_url, {})
        assert isinstance(img, types.ImageType)
        try:
            # should definitely timeout:
            clip_utils.load_image_from_path(good_url, {}, timeout=0.0000001)
            raise AssertionError
        except PIL.UnidentifiedImageError:
            pass

    def test_load_image_from_path_all_req_errors(self):
        """Do we catch other download errors?
        The errors tested inherit from requests.exceptions.RequestException
        """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # it should be fine normally
        clip_utils.load_image_from_path(good_url, {})

        for err in [requests.exceptions.ReadTimeout, requests.exceptions.HTTPError]:
            mock_get = mock.MagicMock()
            mock_get.side_effect = err

            @mock.patch('requests.get', mock_get)
            def run():
                try:
                    clip_utils.load_image_from_path(good_url, {})
                    raise AssertionError
                except PIL.UnidentifiedImageError:
                    pass
                return True

            run()

    def test_load_image_from_path_http_error(self):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # it should be fine normally
        clip_utils.load_image_from_path(good_url, {})
        #
        normal_response = requests.get(good_url)
        assert normal_response.status_code == 200

        for status_code in itertools.chain(range(400, 452), range(500, 512)):
            mock_get = mock.MagicMock()
            bad_response = copy.deepcopy(normal_response)
            bad_response.status_code = status_code
            mock_get.return_value = bad_response

            @mock.patch('requests.get', mock_get)
            def run():
                try:
                    clip_utils.load_image_from_path(good_url, {})
                    raise AssertionError
                except PIL.UnidentifiedImageError as e:
                    assert str(status_code) in str(e)
                return True

            run()

class TestDownloadFromRepo(unittest.TestCase):

    @patch('marqo.s2_inference.clip_utils.download_model')
    def test__download_from_repo_with_auth(self, mock_download_model, ):
        mock_download_model.return_value = 'model.pth'
        location = ModelLocation(
            s3=S3Location(Bucket='some_bucket', Key='some_key'), auth_required=True)
        s3_auth = S3Auth(aws_access_key_id='some_key_id', aws_secret_access_key='some_secret')

        model_props = {
            ModelProperties.model_location: location.dict(),
        }
        auth = {
            's3': s3_auth.dict()
        }

        clip = CLIP(model_properties=model_props, model_auth=auth)
        assert clip._download_from_repo() == 'model.pth'
        mock_download_model.assert_called_once_with(repo_location=location, auth=auth)

    @patch('marqo.s2_inference.clip_utils.download_model')
    def test__download_from_repo_without_auth(self, mock_download_model, ):
        mock_download_model.return_value = 'model.pth'
        location = ModelLocation(
            s3=S3Location(Bucket='some_bucket', Key='some_key'), auth_required=False)

        model_props = {
            ModelProperties.model_location: location.dict(),
        }

        clip = CLIP(model_properties=model_props)
        assert clip._download_from_repo() == 'model.pth'
        mock_download_model.assert_called_once_with(repo_location=location)

    @patch('marqo.s2_inference.clip_utils.download_model')
    def test__download_from_repo_with_empty_filepath(self, mock_download_model):
        mock_download_model.return_value = None
        location = ModelLocation(
            s3=S3Location(Bucket='some_bucket', Key='some_key'), auth_required=False)

        model_props = {
            ModelProperties.model_location: location.dict(),
        }

        clip = CLIP(model_properties=model_props)

        with pytest.raises(RuntimeError):
            clip._download_from_repo()

        mock_download_model.assert_called_once_with(repo_location=location)

