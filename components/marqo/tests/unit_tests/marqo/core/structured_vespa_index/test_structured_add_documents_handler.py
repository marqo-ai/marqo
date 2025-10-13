from unittest.mock import MagicMock

from marqo.core.exceptions import AddDocumentsError
from marqo.core.inference.api import Inference, Modality
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import FieldType
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase


class TestStructuredAddDocumentsHandler(MarqoTestCase):
    IMAGE_URL = 'https://sample.com/abcd.png'
    AUDIO_URL = 'https://sample.com/abcd.wav'
    VIDEO_URL = 'https://sample.com/abcd.mp4'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.handler = StructuredAddDocumentsHandler(
            vespa_client=MagicMock(spec=VespaClient),
            inference=MagicMock(spec=Inference),
            marqo_index=cls.structured_marqo_index(
                'index1', 'index1',
                fields=[], tensor_fields=[]
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', docs=[{'_id': '1', 'field1': 'hello'}]
            ),
        )

    def test_infer_modality_should_return_modality_based_on_field_type(self):
        test_cases = [
            ('Hello World', FieldType.Text, "string should be inferred as text", Modality.TEXT),
            (self.AUDIO_URL, FieldType.AudioPointer, "audio url should be inferred as audio", Modality.AUDIO),
            (self.VIDEO_URL, FieldType.VideoPointer, "video url should be inferred as video", Modality.VIDEO),
            (self.IMAGE_URL, FieldType.ImagePointer, "image url should be inferred as image", Modality.IMAGE),
        ]
        for content, field_type, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = self.handler._infer_modality(
                    TensorField(doc_id='1', field_name='field1', field_content=content,
                                is_top_level_tensor_field=True, field_type=field_type))
                self.assertEqual(expected_modality, modality)

    def test_infer_modality_should_ignore_content(self):
        test_cases = [
            (self.AUDIO_URL, FieldType.Text, "audio url should be inferred as text", Modality.TEXT),
            (self.IMAGE_URL, FieldType.Text, "image url should be inferred as text", Modality.TEXT),
            (self.VIDEO_URL, FieldType.Text, "audio url should be inferred as text", Modality.TEXT),
            (self.IMAGE_URL, FieldType.AudioPointer, "image url should be inferred as audio", Modality.AUDIO),
            (self.VIDEO_URL, FieldType.AudioPointer, "video url should be inferred as audio", Modality.AUDIO),
            (self.IMAGE_URL, FieldType.VideoPointer, "image url should be inferred as video", Modality.VIDEO),
            (self.AUDIO_URL, FieldType.VideoPointer, "audio url should be inferred as video", Modality.VIDEO),
            (self.VIDEO_URL, FieldType.ImagePointer, "video url should be inferred as image", Modality.IMAGE),
            (self.AUDIO_URL, FieldType.ImagePointer, "audio url should be inferred as image", Modality.IMAGE),
        ]
        for content, field_type, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = self.handler._infer_modality(
                    TensorField(doc_id='1', field_name='field1', field_content=content,
                                is_top_level_tensor_field=True, field_type=field_type))
                self.assertEqual(expected_modality, modality)

    def test_infer_modality_should_raise_error_for_unsupported_field_type(self):
        supported_types = [FieldType.Text, FieldType.ImagePointer, FieldType.AudioPointer, FieldType.VideoPointer]

        test_cases = [field_type for field_type in FieldType if field_type not in supported_types] + [None]

        for unsupported_type in test_cases:
            with self.subTest(msg=unsupported_type):
                with self.assertRaises(AddDocumentsError) as context:
                    self.handler._infer_modality(
                        TensorField(doc_id='1', field_name='field1', field_content='Some content',
                                    is_top_level_tensor_field=True, field_type=unsupported_type))

                self.assertIn(f'Error processing field1, tensor field type {unsupported_type} '
                              f'is not supported', str(context.exception))
