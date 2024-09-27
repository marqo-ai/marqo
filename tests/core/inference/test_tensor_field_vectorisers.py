from typing import Dict
from unittest.mock import patch

import pytest
import torch
from PIL.Image import Image

from marqo.core.inference.tensor_fields_container import SingleVectoriser, ModelConfig, BatchCachingVectoriser
from marqo.core.exceptions import AddDocumentsError, ModelError
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.s2_inference.errors import ModelDownloadError
from marqo.s2_inference.multimodal_model_load import Modality
from tests.marqo_test import MarqoTestCase, TestImageUrls
from marqo.s2_inference import errors as s2_inference_errors


@pytest.mark.unittest
class TestTensorFieldVectorisers(MarqoTestCase):
    def setUp(self):
        self.model_config = ModelConfig(
            model_name='random',
            normalize_embeddings=True
        )

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_single_vectoriser_should_vectorise_chunks_passed_in_in_one_go(self, mock_vectorise):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                chunks = ['chunk1', 'chunk2']
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = [[[1.0, 2.0], [3.0, 4.0]]]

                embeddings = SingleVectoriser(modality, self.model_config).vectorise(chunks)

                self.assertEquals([[1.0, 2.0], [3.0, 4.0]], embeddings)
                self.assertEquals(1, mock_vectorise.call_count)
                self.assertEquals(chunks, mock_vectorise.call_args_list[0].kwargs['content'])
                self.assertEquals(modality, mock_vectorise.call_args_list[0].kwargs['modality'])

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_single_vectoriser_should_vectorise_audio_and_video_chunks_one_at_a_time(self, mock_vectorise):
        for modality in [Modality.AUDIO, Modality.VIDEO]:
            with self.subTest(modality=modality):
                chunks = ['chunk1', 'chunk2']
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = [[[1.0, 2.0]], [[3.0, 4.0]]]

                embeddings = SingleVectoriser(modality, self.model_config).vectorise(chunks)

                self.assertEquals([[1.0, 2.0], [3.0, 4.0]], embeddings)
                self.assertEquals(2, mock_vectorise.call_count)
                self.assertEquals(['chunk1'], mock_vectorise.call_args_list[0].kwargs['content'])
                self.assertEquals(['chunk2'], mock_vectorise.call_args_list[1].kwargs['content'])

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_batch_vectoriser_should_vectorise_chunks_in_one_batch(self, mock_vectorise):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                all_chunks = ['chunk1', 'chunk2']
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = [[[1.0, 2.0], [3.0, 4.0]]]

                batch_vectoriser = BatchCachingVectoriser(modality, all_chunks, self.model_config)

                self.assertEquals(1, mock_vectorise.call_count)
                self.assertEquals(all_chunks, mock_vectorise.call_args_list[0].kwargs['content'])

                # verify if the embeddings are cached
                mock_vectorise.reset_mock()
                embeddings = batch_vectoriser.vectorise(['chunk2', 'chunk1'])
                self.assertEquals([[3.0, 4.0], [1.0, 2.0]], embeddings)
                self.assertEquals(0, mock_vectorise.call_count)

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_batch_vectoriser_should_vectorise_audio_and_video_chunks_one_at_a_time(self, mock_vectorise):
        for modality in [Modality.AUDIO, Modality.VIDEO]:
            with self.subTest(modality=modality):
                all_chunks = ['chunk1', 'chunk2']
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = [[[1.0, 2.0]], [[3.0, 4.0]]]

                batch_vectoriser = BatchCachingVectoriser(modality, all_chunks, self.model_config)

                self.assertEquals(2, mock_vectorise.call_count)
                self.assertEquals(['chunk1'], mock_vectorise.call_args_list[0].kwargs['content'])
                self.assertEquals(['chunk2'], mock_vectorise.call_args_list[1].kwargs['content'])

                # verify if the embeddings are cached
                mock_vectorise.reset_mock()
                embeddings = batch_vectoriser.vectorise(['chunk2'])
                self.assertEquals([[3.0, 4.0]], embeddings)
                self.assertEquals(0, mock_vectorise.call_count)

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_batch_vectoriser_should_support_different_content_chunk_types(self, mock_vectorise):
        image0 = load_image_from_path(TestImageUrls.IMAGE0.value, {})
        image1 = load_image_from_path(TestImageUrls.IMAGE1.value, {})

        for modality, chunk_type, content_chunks, side_effect in [
            (Modality.TEXT, str, ['chunk1', 'chunk2'], [[[1.0, 2.0], [3.0, 4.0]]]),
            (Modality.IMAGE, Image, [image0, image1], [[[1.0, 2.0], [3.0, 4.0]]]),
            (Modality.IMAGE, torch.Tensor, [torch.tensor([1., 2.]), torch.tensor([2., 3.])], [[[1.0, 2.0], [3.0, 4.0]]]),
            (Modality.AUDIO, Dict[str, torch.Tensor], [{'audio_chunk1': torch.tensor([1., 2.])},
                                                       {'audio_chunk2': torch.tensor([2., 3.])}], [[[1.0, 2.0]], [[3.0, 4.0]]]),
            (Modality.VIDEO, Dict[str, torch.Tensor], [{'video_chunk1': torch.tensor([1., 2.])},
                                                       {'video_chunk2': torch.tensor([2., 3.])}], [[[1.0, 2.0]], [[3.0, 4.0]]]),
        ]:
            with self.subTest(modality=modality, chunk_type=chunk_type):
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = side_effect
                batch_vectoriser = BatchCachingVectoriser(modality, content_chunks, self.model_config)

                embeddings = batch_vectoriser.vectorise([content_chunks[1]])
                self.assertEquals([[3.0, 4.0]], embeddings)

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_vectoriser_should_raise_error_when_model_fails(self, mock_vectorise):
        for error in [
            s2_inference_errors.UnknownModelError('unknown model'),
            s2_inference_errors.InvalidModelPropertiesError('invalid model properties'),
            s2_inference_errors.ModelLoadError('model load error'),
            ModelDownloadError('model download error'),
        ]:
            with self.subTest(error=error):
                mock_vectorise.reset_mock()
                mock_vectorise.side_effect = error

                with self.assertRaises(ModelError) as context:
                    SingleVectoriser(Modality.TEXT, self.model_config).vectorise(['chunk'])

                self.assertIn('Problem vectorising query. Reason:', str(context.exception))

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_vectoriser_should_wrap_inference_error_as_add_docs_error(self, mock_vectorise):

        mock_vectorise.side_effect = s2_inference_errors.S2InferenceError('inference error')

        with self.assertRaises(AddDocumentsError) as context:
            SingleVectoriser(Modality.TEXT, self.model_config).vectorise(['chunk'])

        self.assertIn('inference error', str(context.exception.error_message))

