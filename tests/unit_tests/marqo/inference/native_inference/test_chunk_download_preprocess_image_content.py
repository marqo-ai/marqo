from unittest import TestCase
from unittest.mock import patch

import PIL.Image
import torch
import numpy as np

from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.inference.type import *
from integ_tests.marqo_test import TestImageUrls
from marqo.inference.native_inference.content_preprocessing import download_and_preprocess_image
from unit_tests.marqo_test import MarqoTestCase


class CLIPPreprocessor(AbstractPreprocessor):
    """A mock preprocessor for testing."""

    def preprocess(self, inputs, modality):
        return [torch.rand(size=(1, 12)) for _ in range(len(inputs))]


def preprocess_side_effect(inputs, *args, **kwargs):
    """
    A side effect function for preprocess mock.
    Returns a list of tensors where each tensor depends on the length of the text.
    """
    return [torch.ones(size=(1, 12)), ] * len(inputs)


def faulty_preprocess_side_effect(inputs, *args, **kwargs):
    """Return fewer tensors than the number of inputs to trigger the ValueError."""
    return [torch.ones(size=(1, 12))] * (len(inputs) - 1)  # One less output


def mock_load_image_from_path(url, *args, **kwargs):
    if url in {e for e in TestImageUrls}:
        return PIL.Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    else:
        raise PIL.UnidentifiedImageError("This image does not exist in the test data")


class TestChunkDownloadPreprocessImage(MarqoTestCase):

    def setUp(self):
        self.preprocessing_config = ImagePreprocessingConfig(
            modality=Modality.IMAGE,
            download_timeout_ms=3000,
            download_thread_count=1,
            download_header=None
        )

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def testdownload_and_preprocess_image_valid_url(self, mock_preprocess, mock_download_image):
        content = [TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value]
        preprocessor = CLIPPreprocessor()
        results = download_and_preprocess_image(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=self.preprocessing_config,
        )

        self.assertEqual(2, len(results))
        self.assertEqual(1, len(results[0]))
        self.assertTrue(isinstance(results[0][0], tuple))
        self.assertEqual(TestImageUrls.IMAGE1.value, results[0][0][0])
        self.assertTrue(torch.eq(torch.ones(size=(1, 12)), results[0][0][1]).all())

        self.assertEqual(1, len(results[1]))
        self.assertTrue(isinstance(results[1][0], tuple))
        self.assertEqual(TestImageUrls.IMAGE2.value, results[1][0][0])
        self.assertTrue(torch.eq(torch.ones(size=(1, 12)), results[1][0][1]).all())

        mock_preprocess.assert_called()
        mock_download_image.assert_called()

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    def testdownload_and_preprocess_image_invalid_url_returns_individual_error(self, mock_download_image):
        """Check behavior when a non-existent image URL is provided."""
        content = ["http://invalid-url.com/does-not-exist.jpg"]
        preprocessor = CLIPPreprocessor()
        results = download_and_preprocess_image(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=self.preprocessing_config,
            return_individual_error=True
        )

        self.assertEqual(1, len(results))
        self.assertTrue(isinstance(results[0], InferenceErrorModel))
        self.assertIn("This image does not exist in the test data", results[0].error_message)

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    def testdownload_and_preprocess_image_invalid_url_raises_error_when_return_individual_error_is_false(self, mock_download_image):
        """Check behavior when a non-existent image URL is provided."""
        content = ["http://invalid-url.com/does-not-exist.jpg"]
        preprocessor = CLIPPreprocessor()

        with self.assertRaises(MediaDownloadError) as context:
            download_and_preprocess_image(
                content=content,
                preprocessor=preprocessor,
                preprocessing_config=self.preprocessing_config,
                return_individual_error=False
            )
        self.assertIn("This image does not exist in the test data", str(context.exception))

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def testdownload_and_preprocess_image_partial_failure(self, mock_preprocess, mock_download_image):
        """One image succeeds, one fails, returned list reflects both outcomes."""
        content = [TestImageUrls.IMAGE1.value, "http://invalid-url.com/does-not-exist.jpg"]
        preprocessor = CLIPPreprocessor()

        results = download_and_preprocess_image(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=self.preprocessing_config,
            return_individual_error=True
        )

        self.assertEqual(len(results), 2)

        # First image success
        self.assertIsInstance(results[0], list)
        self.assertEqual(len(results[0]), 1)
        url_1, tensor_1 = results[0][0]
        self.assertEqual(url_1, TestImageUrls.IMAGE1.value)
        self.assertIsInstance(tensor_1, torch.Tensor)

        # Second image failure
        self.assertIsInstance(results[1], InferenceErrorModel)
        self.assertIn("This image does not exist in the test data", results[1].error_message)

        mock_download_image.assert_called()
        self.assertEqual(2, mock_download_image.call_count)
        mock_preprocess.assert_called()

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def testdownload_and_preprocess_image_all_failures(self, mock_preprocess, mock_download_image):
        """All images fail, returns list of MediaDownloadErrors."""
        content = [
            "http://invalid-url.com/does-not-exist1.jpg",
            "http://invalid-url.com/does-not-exist2.jpg"
        ]
        preprocessor = CLIPPreprocessor()

        results = download_and_preprocess_image(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=self.preprocessing_config,
            return_individual_error=True
        )

        self.assertEqual(len(results), 2)

        for i, result in enumerate(results):
            self.assertIsInstance(result, InferenceErrorModel)
            self.assertIn("This image does not exist in the test data", result.error_message)

        mock_download_image.assert_called()
        self.assertEqual(2, mock_download_image.call_count)
        mock_preprocess.assert_not_called()

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def testdownload_and_preprocess_image_first_fails_rest_succeed(self, mock_preprocess, mock_download_image):
        """First URL fails, others succeed."""
        content = [
            "http://invalid-url.com/fail-first.jpg",
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value
        ]
        preprocessor = CLIPPreprocessor()

        results = download_and_preprocess_image(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=self.preprocessing_config,
            return_individual_error=True
        )

        self.assertEqual(len(results), 3)

        # First image: fail
        self.assertIsInstance(results[0], InferenceErrorModel)
        self.assertIn("This image does not exist in the test data", results[0].error_message)

        # Second and third: success
        for i in [1, 2]:
            self.assertIsInstance(results[i], list)
            self.assertEqual(len(results[i]), 1)
            url, tensor = results[i][0]
            self.assertEqual(url, content[i])
            self.assertIsInstance(tensor, torch.Tensor)

        mock_download_image.assert_called()
        self.assertEqual(3, mock_download_image.call_count)
        mock_preprocess.assert_called()

    @patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.load_image_from_path",
           side_effect=mock_load_image_from_path)
    @patch.object(CLIPPreprocessor, 'preprocess', side_effect=preprocess_side_effect)
    def testdownload_and_preprocess_image_first_fails_raise_error_when_not_returning_individual_error(
            self, mock_preprocess, mock_download_image):
        """First URL fails, skip the others."""
        content = [
            "http://invalid-url.com/fail-first.jpg",
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value
        ]
        preprocessor = CLIPPreprocessor()

        with self.assertRaises(MediaDownloadError) as context:
            download_and_preprocess_image(
                content=content,
                preprocessor=preprocessor,
                preprocessing_config=self.preprocessing_config,
                return_individual_error=False
            )
        self.assertIn("This image does not exist in the test data", str(context.exception))

        mock_download_image.assert_called()
        self.assertEqual(1, mock_download_image.call_count)
        mock_preprocess.assert_not_called()
