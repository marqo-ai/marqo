import unittest
from unittest.mock import MagicMock, patch

# TODO should I move utility method from integ_tests.MarqoTestCase to MarqoTestCase?
from integ_tests.marqo_test import MarqoTestCase
from marqo.core.exceptions import AddDocumentsError
from marqo.core.inference.api import Inference, Modality, MediaDownloadError
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import FieldType
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
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
        patcher = patch("marqo.core.structured_vespa_index.structured_add_document_handler.infer_modality")
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

    def _get_handler(self):
        return StructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.structured_marqo_index(
                'index1', 'index1',
                fields=[], tensor_fields=[]
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', docs=[{'_id': '1', 'field1': 'hello'}]
            ),
        )

    def test_infer_modality_should_return_text_when_field_type_is_text(self):
        handler = self._get_handler()

        test_cases = [
            (self.AUDIO_URL, FieldType.Text, "audio url should be treated as text", Modality.TEXT),
            (self.VIDEO_URL, FieldType.Text, "video url should be treated as text", Modality.TEXT),
            (self.IMAGE_URL, FieldType.Text, "image url should be treated as text", Modality.TEXT),
        ]
        for url, field_type, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='1', field_name='field1', field_content=url,
                                is_top_level_tensor_field=True, field_type=field_type))
                self.assertEqual(expected_modality, modality)
                self.mock_infer_modality.assert_not_called()

    def test_infer_modality_should_succeed_when_modality_matches_field_type(self):
        handler = self._get_handler()

        test_cases = [
            (self.AUDIO_URL, FieldType.AudioPointer, "audio url should be treated as audio", Modality.AUDIO),
            (self.VIDEO_URL, FieldType.VideoPointer, "video url should be treated as video", Modality.VIDEO),
            (self.IMAGE_URL, FieldType.ImagePointer, "image url should be treated as image", Modality.IMAGE),
        ]
        for url, field_type, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = handler._infer_modality(
                    TensorField(doc_id='1', field_name='field1', field_content=url,
                                is_top_level_tensor_field=True, field_type=field_type))
                self.assertEqual(expected_modality, modality)

    def test_infer_modality_should_raise_error_when_fails_to_download_file(self):
        handler = self._get_handler()

        with self.assertRaises(AddDocumentsError) as context:
            handler._infer_modality(
                TensorField(doc_id='1', field_name='field1', field_content=self.INVALID_URL,
                            is_top_level_tensor_field=True, field_type=FieldType.ImagePointer))
        self.assertIn(f'Error processing field1: Error downloading media file {self.INVALID_URL}',
                      str(context.exception))

    def test_infer_modality_should_raise_error_when_inferred_modality_does_not_match_field_type(self):
        handler = self._get_handler()

        test_cases = [
            (self.AUDIO_URL, FieldType.ImagePointer, "audio url does not match image modality", Modality.AUDIO),
            (self.AUDIO_URL, FieldType.VideoPointer, "audio url does not match video modality", Modality.AUDIO),

            (self.VIDEO_URL, FieldType.ImagePointer, "video url does not match image modality", Modality.VIDEO),
            (self.VIDEO_URL, FieldType.AudioPointer, "video url does not match audio modality", Modality.VIDEO),

            (self.IMAGE_URL, FieldType.VideoPointer, "image url does not match video modality", Modality.IMAGE),
            (self.IMAGE_URL, FieldType.AudioPointer, "image url does not match audio modality", Modality.IMAGE),

            ('some text', FieldType.VideoPointer, "text does not match video modality", Modality.TEXT),
            ('some text', FieldType.AudioPointer, "text does not match video modality", Modality.TEXT),
            ('some text', FieldType.ImagePointer, "text does not match video modality", Modality.TEXT),

        ]
        for url, field_type, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                with self.assertRaises(AddDocumentsError) as context:
                    handler._infer_modality(
                        TensorField(doc_id='1', field_name='field1', field_content=url,
                                    is_top_level_tensor_field=True, field_type=field_type))

                self.assertIn(f'Error processing field1, detected as {expected_modality.value}, '
                              f'but expected field type is {field_type.value}', str(context.exception))
