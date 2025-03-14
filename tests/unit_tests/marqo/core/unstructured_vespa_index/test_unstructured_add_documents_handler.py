import unittest
from unittest.mock import MagicMock, patch

# TODO should I move utility method from integ_tests.MarqoTestCase to MarqoTestCase?
from integ_tests.marqo_test import MarqoTestCase, TestAudioUrls, TestVideoUrls, TestImageUrls
from marqo.core.inference.api import Inference, Modality
from marqo.core.inference.tensor_fields_container import TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.vespa.vespa_client import VespaClient


class TestUnstructuredAddDocumentsHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vespa_client = MagicMock(spec=VespaClient)
        cls.inference = MagicMock(spec=Inference)
        MarqoTestCase.configure_request_metrics()

    def test_unstructured_add_documents_handler_infer_modality_logic_image_false_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        both treat_urls_and_pointers_as_images and treat_urls_and_pointers_as_media are False."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=False,
                treat_urls_and_pointers_as_media=False
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as text"),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as text"),
            (TestImageUrls.IMAGE1.value, "image url should be treated as text"),
        ]
        for url, test_case in test_cases:
            with self.subTest(msg=test_case):
                with patch("marqo.core.inference.modality_utils.infer_modality") as mock_infer_modality:
                    modality = unstructured_add_documents_handler._infer_modality(
                        TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                    is_top_level_tensor_field=True))
                    self.assertEqual(Modality.TEXT, modality)
                mock_infer_modality.assert_not_called()

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=False."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=True,
                treat_urls_and_pointers_as_media=False
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as text", Modality.TEXT),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as text", Modality.TEXT),
            (TestImageUrls.IMAGE1.value, "image url should be treated as image", Modality.IMAGE),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = unstructured_add_documents_handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality,modality)

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_true(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=True."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=MarqoTestCase.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=True,
                treat_urls_and_pointers_as_media=True
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as audio", Modality.AUDIO),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as video", Modality.VIDEO),
            (TestImageUrls.IMAGE1.value, "image url should be treated as image", Modality.IMAGE),
        ]

        for url, test_case, expected_modality in test_cases:
            with self.subTest(msg=test_case):
                modality = unstructured_add_documents_handler._infer_modality(
                    TensorField(doc_id='id', field_name='dummy_field_name', field_content=url,
                                is_top_level_tensor_field=True))
                self.assertEqual(expected_modality, modality)