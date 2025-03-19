import unittest
from typing import Optional, List
from unittest.mock import MagicMock, patch

import numpy as np

# TODO should I move utility method from integ_tests.MarqoTestCase to MarqoTestCase?
from integ_tests.marqo_test import MarqoTestCase
from marqo.core.exceptions import AddDocumentsError, InternalError
from marqo.core.inference.api import Inference, Modality, MediaDownloadError, InferenceRequest, InferenceResult, \
    InferenceError, InferenceErrorModel
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import PatchMethod, ImagePreProcessing
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient


class TestUnstructuredAddDocumentsHandler(unittest.TestCase):
    IMAGE_URL = 'https://sample.com/abcd.png'
    AUDIO_URL = 'https://sample.com/abcd.wav'
    VIDEO_URL = 'https://sample.com/abcd.mp4'
    INVALID_URL = 'https://invalid_url'

    @classmethod
    def setUpClass(cls) -> None:
        MarqoTestCase.configure_request_metrics()

    def setUp(self):
        self.vespa_client = MagicMock(spec=VespaClient)
        self.inference = MagicMock(spec=Inference)

        def vectorise_side_effect(request: InferenceRequest) -> InferenceResult:
            result = []
            for content in request.contents:
                if content.startswith('error:'):
                    result.append(InferenceErrorModel(error_message=content))
                elif request.preprocessing_config.should_chunk:
                    result.append([('chunk1', np.array([1.0, 2.0])), ('chunk2', np.array([2.0, 4.0]))])
                else:
                    result.append([(content, np.array([1.0, 2.0]))])
            return InferenceResult(result=result)

        self.inference.vectorise.side_effect = vectorise_side_effect

        # patch infer_modality
        patcher = patch("marqo.core.unstructured_vespa_index.unstructured_add_document_handler.infer_modality")
        self.mock_infer_modality = patcher.start()
        self.addCleanup(patcher.stop)

        def infer_modality_side_effect(url: str, media_download_header) -> Modality:
            if url == self.IMAGE_URL:
                return Modality.IMAGE
            elif url == self.AUDIO_URL:
                return Modality.AUDIO
            elif url == self.VIDEO_URL:
                return Modality.VIDEO
            elif url == self.INVALID_URL:
                raise MediaDownloadError(f"Error downloading media file {url}")
            else:
                return Modality.TEXT

        self.mock_infer_modality.side_effect = infer_modality_side_effect

    def _get_handler(self, treat_as_images: bool, treat_as_media: bool,
                     add_docs_params: Optional[AddDocsParams] = None,
                     patch_method: Optional[PatchMethod] = None,
                     ) -> UnstructuredAddDocumentsHandler:
        return UnstructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=treat_as_images,
                treat_urls_and_pointers_as_media=treat_as_media,
                image_preprocessing=ImagePreProcessing(patch_method=patch_method),
            ),
            add_docs_params=add_docs_params or AddDocsParams(
                index_name='index1', tensor_fields=['field1'], docs=[{'_id': '1', 'field1': 'hello'}]
            ),
        )

    def test_infer_modality_logic_image_false_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        both treat_urls_and_pointers_as_images and treat_urls_and_pointers_as_media are False."""
        handler = self._get_handler(treat_as_images=False, treat_as_media=False)

        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as text"),
            (self.VIDEO_URL, "video url should be treated as text"),
            (self.IMAGE_URL, "image url should be treated as text"),
            ('text', "text should be treated as text"),
        ]
        for url, test_case in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(Modality.TEXT, modality)
                self.mock_infer_modality.assert_not_called()

    def test_infer_modality_logic_image_true_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=False."""
        handler = self._get_handler(treat_as_images=True, treat_as_media=False)
        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as text", Modality.TEXT),
            (self.VIDEO_URL, "video url should be treated as text", Modality.TEXT),
            (self.IMAGE_URL, "image url should be treated as image", Modality.IMAGE),
            ('text', "text should be treated as text", Modality.TEXT),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality,modality)

    def test_infer_modality_logic_image_true_and_media_true(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=True."""
        handler = self._get_handler(treat_as_images=True, treat_as_media=True)

        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as audio", Modality.AUDIO),
            (self.VIDEO_URL, "video url should be treated as video", Modality.VIDEO),
            (self.IMAGE_URL, "image url should be treated as image", Modality.IMAGE),
            ('text', "text should be treated as text", Modality.TEXT),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality, modality)

    def test_infer_modality_should_raise_error_when_fails_to_download(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=True)

        with self.assertRaises(AddDocumentsError) as context:
            handler._infer_modality(
                TensorField(doc_id='id', field_name='field1', field_content=self.INVALID_URL,
                            is_top_level_tensor_field=True))
        self.assertIn(f'Error processing field1: Error downloading media file {self.INVALID_URL}',
                      str(context.exception))

    def test_vectorise_tensor_fields_should_call_vectorise_with_all_modalities(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=True, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field', 'image_field', 'audio_field', 'video_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello', 'image_field': self.IMAGE_URL},
                {'_id': '2', 'text_field': 'hello2', 'audio_field': self.AUDIO_URL, 'video_field': self.VIDEO_URL},
            ]
        ))

        handler.add_documents()

        self.assertEqual(self.inference.vectorise.call_count, 4)
        modality_collected = set([self.inference.vectorise.call_args_list[i][0][0].modality for i in range(4)])
        self.assertSetEqual(modality_collected, {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})

    def test_vectorise_tensor_fields_should_group_contents_for_the_same_modality_in_a_batch(self):
        handler = self._get_handler(treat_as_images=False, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field', 'image_field', 'audio_field', 'video_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello', 'image_field': self.IMAGE_URL},
                {'_id': '2', 'text_field': 'hello2', 'audio_field': self.AUDIO_URL, 'video_field': self.VIDEO_URL},
            ]
        ))

        handler.add_documents()

        self.assertEqual(self.inference.vectorise.call_count, 1)
        contents = self.inference.vectorise.call_args_list[0][0][0].contents
        # we don't treat urls as images or media in this test
        self.assertSetEqual(set(contents), {'hello', 'hello2', self.IMAGE_URL, self.AUDIO_URL, self.VIDEO_URL})

    def test_vectorise_tensor_fields_should_skip_docs_with_issue_infer_modality(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field', 'image_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello', 'image_field': self.INVALID_URL},
                {'_id': '2', 'text_field': 'hello2'},
            ]
        ))

        res = handler.add_documents()

        self.assertEqual(self.inference.vectorise.call_count, 1)
        self.assertEqual(self.inference.vectorise.call_args_list[0][0][0].contents, ['hello2'])

        self.assertTrue(res.errors)
        self.assertEqual(res.items[0].status, 400)
        self.assertEqual(res.items[0].message, f'Error processing image_field: '
                                               f'Error downloading media file {self.INVALID_URL}')

    def test_vectorise_tensor_fields_should_collect_individual_inference_error(self):
        handler = self._get_handler(treat_as_images=False, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field', 'text_field2'],
            docs=[
                {'_id': '1', 'text_field': 'error:oops', 'text_field2': 'hello'},
                {'_id': '2', 'text_field': 'hello2'},
            ]
        ))

        res = handler.add_documents()

        self.assertTrue(res.errors)
        self.assertEqual(res.items[0].status, 400)
        self.assertEqual(res.items[0].message, 'error:oops')

    def test_vectorise_tensor_fields_should_populate_chunks_and_embeddings(self):
        handler = self._get_handler(treat_as_images=False, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello'},
            ]
        ))

        handler.add_documents()

        self.assertEqual(self.vespa_client.feed_batch.call_count, 1)
        vespa_doc: VespaDocument = self.vespa_client.feed_batch.call_args_list[0][0][0][0]
        self.assertEqual(vespa_doc.fields['marqo__chunks'], ['text_field::chunk1', 'text_field::chunk2'])
        self.assertEqual(vespa_doc.fields['marqo__embeddings'], {'0': [1.0, 2.0], '1': [2.0, 4.0]})
        
    def test_vectorise_tensor_fields_should_call_vectorise_for_multimodal_subfields(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['combo_field'],
            mappings={
                'captioned_image': {
                    'type': 'multimodal_combination',
                    'weights': {'text_field': 0.3, 'image_field': 0.7}
                }
            },
            docs=[
                {'_id': '1', 'text_field': 'hello', 'image_field': self.IMAGE_URL},
            ]
        ))

        handler.add_documents()
        self.assertEqual(self.inference.vectorise.call_count, 2)
        modality_collected = set([self.inference.vectorise.call_args_list[i][0][0].modality for i in range(2)])
        self.assertSetEqual(modality_collected, {Modality.TEXT, Modality.IMAGE})

    def test_vectorise_tensor_fields_should_call_vectorise_for_top_level_tensor_fields_and_subfields_image_text(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['combo_field', 'text_field', 'image_field'],
            mappings={
                'captioned_image': {
                    'type': 'multimodal_combination',
                    'weights': {'text_field': 0.3, 'image_field': 0.7}
                }
            },
            docs=[
                {'_id': '1', 'text_field': 'hello', 'image_field': self.IMAGE_URL},
            ]
        ), patch_method=PatchMethod.Simple)  # we also chunk image

        handler.add_documents()

        # There should be four calls to the inference in total, for image and text field, chunked and not chunked.
        self.assertEqual(self.inference.vectorise.call_count, 4)
        requests: List[InferenceRequest] = [self.inference.vectorise.call_args_list[i][0][0] for i in range(4)]
        self.assertTrue(any([r.modality == Modality.TEXT and r.preprocessing_config.should_chunk for r in requests]))
        self.assertTrue(any([r.modality == Modality.TEXT and not r.preprocessing_config.should_chunk for r in requests]))
        self.assertTrue(any([r.modality == Modality.IMAGE and r.preprocessing_config.should_chunk for r in requests]))
        self.assertTrue(any([r.modality == Modality.IMAGE and not r.preprocessing_config.should_chunk for r in requests]))

    def test_vectorise_tensor_fields_should_call_vectorise_for_top_level_tensor_fields_and_subfields_audio_video(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=True, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['combo_field', 'video_field', 'audio_field'],
            mappings={
                'captioned_image': {
                    'type': 'multimodal_combination',
                    'weights': {'video_field': 0.3, 'audio_field': 0.7}
                }
            },
            docs=[
                {'_id': '1', 'video_field': self.VIDEO_URL, 'audio_field': self.AUDIO_URL},
            ]
        ))

        handler.add_documents()

        # There should be only two calls to the inference in total, since audio and video will always be chunked
        self.assertEqual(self.inference.vectorise.call_count, 2)
        requests: List[InferenceRequest] = [self.inference.vectorise.call_args_list[i][0][0] for i in range(2)]
        self.assertTrue(any([r.modality == Modality.AUDIO and r.preprocessing_config.should_chunk for r in requests]))
        self.assertTrue(any([r.modality == Modality.VIDEO and r.preprocessing_config.should_chunk for r in requests]))

    def test_vectorise_tensor_fields_should_propagate_inference_errors_for_the_batch(self):
        self.inference.vectorise.side_effect = InferenceError('oops')

        handler = self._get_handler(treat_as_images=False, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello'},
            ]
        ))

        with self.assertRaises(InferenceError) as context:
            handler.add_documents()
        self.assertEqual('oops', str(context.exception))

    def test_vectorise_tensor_fields_should_raise_inference_error_if_result_size_does_not_match_contents(self):
        self.inference.vectorise.side_effect = [InferenceResult(result=[[('chunk', np.array([1.0, 2.0]))]])]

        handler = self._get_handler(treat_as_images=False, treat_as_media=False, add_docs_params=AddDocsParams(
            index_name='index1', tensor_fields=['text_field'],
            docs=[
                {'_id': '1', 'text_field': 'hello'},
                {'_id': '2', 'text_field': 'hello2'},
            ]
        ))

        with self.assertRaises(InternalError) as context:
            handler.add_documents()
        self.assertEqual('Inference result contains chunks and embeddings for 1 fields, but 2 are expected',
                         str(context.exception))
