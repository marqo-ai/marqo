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
from tests.marqo_test import TestImageUrls


class TestImageDownloading(unittest.TestCase):

    def test_loadImageFromPathTimeout(self):
        good_url = TestImageUrls.HIPPO_REALISTIC.value
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
        good_url = TestImageUrls.HIPPO_REALISTIC.value
        clip_utils.load_image_from_path(good_url, {})
        for err in [pycurl.error]:
            with mock.patch('pycurl.Curl') as MockCurl:
                mock_curl = MockCurl.return_value
                mock_curl.perform.side_effect = err
                with self.assertRaises(PIL.UnidentifiedImageError):
                    clip_utils.load_image_from_path(good_url, {})

    @patch('pycurl.Curl')
    def test_downloadImageFromRrlCloseCalled(self, MockCurl):
        good_url = TestImageUrls.HIPPO_REALISTIC.value

        mock_curl_instance = MockCurl.return_value
        mock_curl_instance.getinfo.return_value = 200

        try:
            clip_utils.load_image_from_path(good_url, {})
        except (ImageDownloadError, PIL.UnidentifiedImageError):
            pass  # This test is only for checking if close() is called

        # Check if c.close() was called
        mock_curl_instance.close.assert_called_once()

class TestIsImage(unittest.TestCase):
    def test_is_image(self):
        test_cases = [
            ("Valid JPG", 'image.jpg', True),
            ("Valid PNG", 'image.png', True),
            ("Valid JPEG", 'image.jpeg', True),
            ("Valid BMP", 'image.bmp', True),
            ("Uppercase JPG", 'image.JPG', True),
            ("Uppercase PNG", 'image.PNG', True),
            ("Valid URL", 'https://example.com/image.jpg', True),
            ("Invalid PDF", 'document.pdf', False),
            ("Invalid TXT", 'text.txt', False),
            ("No extension", 'imagewithoutextension', False),
            ("Valid URL with unencoded characters(space) ",
             "http://dummy.dummy.com/is/image/dummy/dummy (1)?wid=123&hei=321&qlt=123&fmt=png-alpha", True)
        ]

        for description, input_value, expected in test_cases:
            with self.subTest(description=description, input=input_value):
                self.assertEqual(clip_utils._is_image(input_value), expected,
                                f"Failed for {description}: input '{input_value}' expected {expected}")

        # Test with PIL Image
        with self.subTest("PIL Image"):
            pil_image = PIL.Image.new('RGB', (100, 100))
            self.assertTrue(clip_utils._is_image(pil_image))

        # Test with list of images
        with self.subTest("List of images"):
            image_list = ['image1.jpg', 'image2.png', 'https://example.com/image3.jpeg']
            self.assertTrue(clip_utils._is_image(image_list))

        # Test with empty list
        with self.subTest("Empty list"):
            with self.assertRaises(clip_utils.UnidentifiedImageError):
                clip_utils._is_image([])

        # Test with invalid type
        with self.subTest("Invalid type"):
            with self.assertRaises(clip_utils.UnidentifiedImageError):
                clip_utils._is_image(123)


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