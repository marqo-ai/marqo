import unittest
from unittest import mock
from unittest.mock import patch

import PIL
import pycurl
import pytest

from marqo.api.exceptions import InternalError
from marqo.s2_inference import clip_utils, types
from marqo.s2_inference.clip_utils import CLIP, OPEN_CLIP, FP16_CLIP, MULTILINGUAL_CLIP
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import ImageDownloadError
from marqo.tensor_search.enums import ModelProperties
from marqo.tensor_search.models.private_models import ModelLocation
from marqo.tensor_search.models.private_models import S3Auth, S3Location, HfModelLocation


class TestImageDownloading(unittest.TestCase):

    def test_loadImageFromPathTimeout(self):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # should be fine on regular timeout:
        img = clip_utils.load_image_from_path(good_url, {})
        assert isinstance(img, types.ImageType)
        with self.assertRaises(PIL.UnidentifiedImageError):
            # should definitely timeout:
            clip_utils.load_image_from_path(good_url, {}, timeout_ms=1)


    def test_loadImageFromPathAllRequestErrors(self):
        """Do we catch other download errors?
        The errors tested inherit from requests.exceptions.RequestException
        """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        clip_utils.load_image_from_path(good_url, {})
        for err in [pycurl.error]:
            with mock.patch('pycurl.Curl') as MockCurl:
                mock_curl = MockCurl.return_value
                mock_curl.perform.side_effect = err
                with self.assertRaises(PIL.UnidentifiedImageError):
                    clip_utils.load_image_from_path(good_url, {})

    @patch('pycurl.Curl')
    def test_downloadImageFromRrlCloseCalled(self, MockCurl):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'

        mock_curl_instance = MockCurl.return_value
        mock_curl_instance.getinfo.return_value = 200

        try:
            clip_utils.load_image_from_path(good_url, {})
        except (ImageDownloadError, PIL.UnidentifiedImageError):
            pass  # This test is only for checking if close() is called

        # Check if c.close() was called
        mock_curl_instance.close.assert_called_once()


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

        clip = CLIP(model_properties=model_props, model_auth=auth, device="cpu")
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

        clip = CLIP(model_properties=model_props, device="cpu")
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

        clip = CLIP(model_properties=model_props, device="cpu")

        with pytest.raises(RuntimeError):
            clip._download_from_repo()

        mock_download_model.assert_called_once_with(repo_location=location)

class TestLoad(unittest.TestCase):
    """tests the CLIP.load() method"""
    @patch('marqo.s2_inference.clip_utils.clip.load', return_value=(mock.Mock(), mock.Mock()))
    def test_load_without_model_properties(self, mock_clip_load):
        clip = CLIP(device="cpu")
        clip.load()
        mock_clip_load.assert_called_once_with('ViT-B/32', device='cpu', jit=False, download_root=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.clip.load', return_value=(mock.Mock(), mock.Mock()))
    @patch('os.path.isfile', return_value=True)
    def test_load_with_local_file(self, mock_isfile, mock_clip_load):
        model_path = 'localfile.pth'
        clip = CLIP(model_properties={'localpath': model_path}, device="cpu")
        clip.load()
        mock_clip_load.assert_called_once_with(name=model_path, device='cpu', jit=False, download_root=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.download_model', return_value='downloaded_model.pth')
    @patch('marqo.s2_inference.clip_utils.clip.load', return_value=(mock.Mock(), mock.Mock()))
    @patch('os.path.isfile', return_value=False)
    @patch('validators.url', return_value=True)
    def test_load_with_url(self, mock_url_valid, mock_isfile, mock_clip_load, mock_download_model):
        model_url = 'http://example.com/model.pth'
        clip = CLIP(model_properties={'url': model_url}, device="cpu")
        clip.load()
        mock_download_model.assert_called_once_with(url=model_url)
        mock_clip_load.assert_called_once_with(name='downloaded_model.pth', device='cpu', jit=False, download_root=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.CLIP._download_from_repo', return_value='downloaded_model.pth')
    @patch('marqo.s2_inference.clip_utils.clip.load', return_value=(mock.Mock(), mock.Mock()))
    def test_load_with_model_location(self, mock_clip_load, mock_download_from_repo):
        model_location = ModelLocation(s3=S3Location(Bucket='some_bucket', Key='some_key'))
        clip = CLIP(model_properties={ModelProperties.model_location: model_location.dict()}, device="cpu")
        clip.load()
        mock_download_from_repo.assert_called_once()
        mock_clip_load.assert_called_once_with(name='downloaded_model.pth', device='cpu', jit=False, download_root=ModelCache.clip_cache_path)
    
    def test_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            clip = CLIP(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
    
    def test_fp16_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            clip = FP16_CLIP(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
    
    def test_multilingual_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            clip = MULTILINGUAL_CLIP(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
        

class TestOpenClipLoad(unittest.TestCase):

    @patch('marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms', 
           return_value=(mock.Mock(), mock.Mock(), mock.Mock()))
    def test_load_without_model_properties(self, mock_open_clip_create_model_and_transforms):
        """By default laion400m_e32 is loaded..."""
        open_clip = OPEN_CLIP(device="cpu")
        open_clip.load()
        mock_open_clip_create_model_and_transforms.assert_called_once_with(
            'ViT-B-32-quickgelu', pretrained='laion400m_e32', 
            device='cpu', jit=False, cache_dir=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms', 
           return_value=(mock.Mock(), mock.Mock(), mock.Mock()))
    @patch('os.path.isfile', return_value=True)
    def test_load_with_local_file(self, mock_isfile, mock_open_clip_create_model_and_transforms):
        model_path = 'localfile.pth'
        open_clip = OPEN_CLIP(model_properties={'localpath': model_path}, device="cpu")
        open_clip.load()
        mock_open_clip_create_model_and_transforms.assert_called_once_with(
            model_name=open_clip.model_name, jit=False, pretrained=model_path,
            precision='fp32', image_mean=None, image_std=None, 
            device='cpu', cache_dir=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms', 
           return_value=(mock.Mock(), mock.Mock(), mock.Mock()))
    @patch('validators.url', return_value=True)
    @patch('marqo.s2_inference.clip_utils.download_model', return_value='model.pth')
    def test_load_with_url(self, mock_download_model, mock_validators_url, mock_open_clip_create_model_and_transforms):
        model_url = 'http://model.com/model.pth'
        open_clip = OPEN_CLIP(model_properties={'url': model_url}, device="cpu")
        open_clip.load()
        mock_download_model.assert_called_once_with(url=model_url)
        mock_open_clip_create_model_and_transforms.assert_called_once_with(
            model_name=open_clip.model_name, jit=False, pretrained='model.pth', precision='fp32',
            image_mean=None, image_std=None, device='cpu', cache_dir=ModelCache.clip_cache_path)

    @patch('marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms', 
           return_value=(mock.Mock(), mock.Mock(), mock.Mock()))
    @patch('marqo.s2_inference.clip_utils.CLIP._download_from_repo', 
           return_value='model.pth')
    def test_load_with_model_location(self, mock_download_from_repo, mock_open_clip_create_model_and_transforms):
        open_clip = OPEN_CLIP(model_properties={
            ModelProperties.model_location: ModelLocation(
                auth_required=True, hf=HfModelLocation(repo_id='someId', filename='some_file.pt')).dict()}, device="cpu")
        open_clip.load()
        mock_download_from_repo.assert_called_once()
        mock_open_clip_create_model_and_transforms.assert_called_once_with(
            model_name=open_clip.model_name, jit=False, pretrained='model.pth', precision='fp32',
            image_mean=None, image_std=None, device='cpu', cache_dir=ModelCache.clip_cache_path)
    
    def test_open_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            clip = OPEN_CLIP(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
