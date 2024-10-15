import unittest
from unittest.mock import patch

import pytest
import torch
from PIL.Image import Image
from torch import tensor

from marqo.core.inference.tensor_fields_container import TextChunker, ImageChunker, AudioVideoChunker
from marqo.core.exceptions import AddDocumentsError
from marqo.core.models.marqo_index import TextPreProcessing, TextSplitMethod, ImagePreProcessing, PatchMethod
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.s2_inference import errors as s2_inference_errors
from tests.marqo_test import TestImageUrls


@pytest.mark.unittest
class TestTensorFieldChunkers(unittest.TestCase):
    def test_text_chunker_should_chunk_text(self):
        text_chunker = TextChunker(text_chunk_prefix='prefix: ', text_preprocessing=TextPreProcessing(
            split_length=3, split_overlap=1, split_method=TextSplitMethod.Word))

        chunks, content_chunks = text_chunker.chunk('some text to test chunking')
        self.assertEqual(['some text to', 'to test chunking'], chunks)
        self.assertEqual(['prefix: some text to', 'prefix: to test chunking'], content_chunks)

    def test_text_chunker_should_return_single_chunk_when_single_chunk_flag_is_true(self):
        text_chunker = TextChunker(text_chunk_prefix='prefix: ', text_preprocessing=TextPreProcessing(
            split_length=3, split_overlap=1, split_method=TextSplitMethod.Word))

        chunks, content_chunks = text_chunker.chunk('some text to test chunking', single_chunk=True)
        self.assertEqual(['some text to test chunking'], chunks)
        self.assertEqual(['prefix: some text to test chunking'], content_chunks)

    def test_image_chunker_should_chunk_image(self):
        image_url = TestImageUrls.HIPPO_REALISTIC.value
        media_repo = {image_url: load_image_from_path(image_url, {})}
        image_chunker = ImageChunker(media_repo=media_repo, device='cpu',
                                     image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.DinoV1))

        chunks, content_chunks = image_chunker.chunk(image_url)
        self.assertEquals([[0.0, 0.0, 512.0, 512.0], [0.0, 102.4, 409.6, 375.46666666666664]], chunks)
        self.assertEquals(2, len(content_chunks))
        self.assertIsInstance(content_chunks[0], Image)
        self.assertIsInstance(content_chunks[1], Image)

    def test_image_chunker_should_return_single_chunk_when_patch_method_is_none(self):
        image_url = TestImageUrls.HIPPO_REALISTIC.value
        media_repo = {image_url: load_image_from_path(image_url, {})}
        image_chunker = ImageChunker(media_repo=media_repo, device='cpu',
                                     image_preprocessing=ImagePreProcessing(patch_method=None))

        chunks, content_chunks = image_chunker.chunk(image_url)
        self.assertEquals([TestImageUrls.HIPPO_REALISTIC.value], chunks)
        self.assertEquals(1, len(content_chunks))
        self.assertIsInstance(content_chunks[0], Image)

    def test_image_chunker_should_return_single_chunk_when_single_chunk_flag_is_true(self):
        image_url = TestImageUrls.HIPPO_REALISTIC.value
        media_repo = {image_url: load_image_from_path(image_url, {})}
        image_chunker = ImageChunker(media_repo=media_repo, device='cpu',
                                     image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.DinoV1))

        chunks, content_chunks = image_chunker.chunk(image_url, single_chunk=True)
        self.assertEquals([TestImageUrls.HIPPO_REALISTIC.value], chunks)
        self.assertEquals(1, len(content_chunks))
        self.assertIsInstance(content_chunks[0], Image)

    @patch('marqo.core.inference.tensor_fields_container.image_processor.chunk_image')
    def test_image_chunker_should_raise_error_when_chunking_fails(self, mock_chunk_image):
        image_url = TestImageUrls.HIPPO_REALISTIC.value
        media_repo = {image_url: load_image_from_path(image_url, {})}

        mock_chunk_image.side_effect = [s2_inference_errors.S2InferenceError('BOOM!')]

        image_chunker = ImageChunker(media_repo=media_repo, device='cpu',
                                     image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.DinoV1))

        with self.assertRaises(AddDocumentsError) as err_context:
            image_chunker.chunk(image_url)

        self.assertEquals('BOOM!', err_context.exception.error_message)

    def test_audio_video_chunker_should_chunk_audio_and_video(self):
        media_repo = {'url': [
            {'start_time': 5, 'end_time': 15, 'tensor': tensor([1.0, 2.0])},
            {'start_time': 20, 'end_time': 25, 'tensor': tensor([4.0, 6.0])},
        ]}
        audio_video_chunker = AudioVideoChunker(media_repo=media_repo)

        chunks, content_chunks = audio_video_chunker.chunk('url')
        self.assertEquals(['[5, 15]', '[20, 25]'], chunks)
        self.assertTrue(torch.equal(tensor([1.0, 2.0]), content_chunks[0]))
        self.assertTrue(torch.equal(tensor([4.0, 6.0]), content_chunks[1]))

    def test_audio_video_chunker_should_raise_error_when_single_chunk_flag_is_true(self):
        audio_video_chunker = AudioVideoChunker(media_repo={'url': []})

        with self.assertRaises(RuntimeError) as err_context:
            audio_video_chunker.chunk('url', single_chunk=True)

        self.assertEquals('Video and Audio chunker does not support single_chunk', str(err_context.exception))