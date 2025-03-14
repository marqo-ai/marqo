import unittest
from unittest.mock import MagicMock, patch

# TODO should I move utility method from integ_tests.MarqoTestCase to MarqoTestCase?
from integ_tests.marqo_test import MarqoTestCase
from marqo.core.exceptions import AddDocumentsError
from marqo.core.inference.api import Inference, Modality, MediaDownloadError
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.vespa.vespa_client import VespaClient


class TestUnstructuredAddDocumentsHandler(unittest.TestCase):
    IMAGE_URL = 'https://sample.com/abcd.png'
    AUDIO_URL = 'https://sample.com/abcd.wav'
    VIDEO_URL = 'https://sample.com/abcd.mp4'
    INVALID_URL = 'https://invalid_url'

    @classmethod
    def setUpClass(cls) -> None:
        cls.vespa_client = MagicMock(spec=VespaClient)
        cls.inference = MagicMock(spec=Inference)
        MarqoTestCase.configure_request_metrics()

    def setUp(self):
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
            else:
                raise MediaDownloadError(f"Error downloading media file {url}")

        self.mock_infer_modality.side_effect = infer_modality_side_effect

    def _get_handler(self, treat_as_images: bool, treat_as_media: bool):
        return UnstructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=treat_as_images,
                treat_urls_and_pointers_as_media=treat_as_media
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'], docs=[{'_id': '1', 'field1': 'hello'}]
            ),
        )

    def test_unstructured_add_documents_handler_infer_modality_logic_image_false_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        both treat_urls_and_pointers_as_images and treat_urls_and_pointers_as_media are False."""
        handler = self._get_handler(treat_as_images=False, treat_as_media=False)

        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as text"),
            (self.VIDEO_URL, "video url should be treated as text"),
            (self.IMAGE_URL, "image url should be treated as text"),
        ]
        for url, test_case in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(Modality.TEXT, modality)
                self.mock_infer_modality.assert_not_called()

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=False."""
        handler = self._get_handler(treat_as_images=True, treat_as_media=False)
        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as text", Modality.TEXT),
            (self.VIDEO_URL, "video url should be treated as text", Modality.TEXT),
            (self.IMAGE_URL, "image url should be treated as image", Modality.IMAGE),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality,modality)

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_true(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=True."""
        handler = self._get_handler(treat_as_images=True, treat_as_media=True)

        test_cases = [
            (self.AUDIO_URL, "audio url should be treated as audio", Modality.AUDIO),
            (self.VIDEO_URL, "video url should be treated as video", Modality.VIDEO),
            (self.IMAGE_URL, "image url should be treated as image", Modality.IMAGE),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality, modality)

    def test_unstructured_add_documents_handler_infer_modality_should_raise_error_when_fails_to_download(self):
        handler = self._get_handler(treat_as_images=True, treat_as_media=True)

        with self.assertRaises(AddDocumentsError) as context:
            handler._infer_modality(
                TensorField(doc_id='id', field_name='dummy_field_name', field_content=self.INVALID_URL,
                            is_top_level_tensor_field=True))
        self.assertIn('Error downloading media file', str(context.exception))