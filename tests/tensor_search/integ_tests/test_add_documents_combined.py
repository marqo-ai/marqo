import os
import uuid
from unittest import mock
from unittest.mock import patch

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from urllib3.exceptions import ProtocolError


class TestAddDocumentsStructured(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_image_index_request = cls.structured_marqo_index_request(
            name="structured_image_index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch])
            ],
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            tensor_fields=["image_field_1", "text_field_1"]
        )

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )

        cls.indexes = cls.create_indexes([
            structured_image_index_request,
            unstructured_image_index_request
        ])

        cls.structured_marqo_index_name = structured_image_index_request.name
        cls.unstructured_marqo_index_name = unstructured_image_index_request.name

    def setUp(self) -> None:
        super().setUp()

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_add_documents_with_truncated_image(self):
        """Test to ensure that the add_documents API can properly return 400 for the document with a truncated image."""
        truncated_image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/truncated_image.jpg"

        documents = [
            {
                "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png",
                "text_field_1": "This is a valid image",
                "_id": "1"
            },
            {
                "image_field_1": truncated_image_url,
                "text_field_1": "This is a truncated image",
                "_id": "2"
            }
        ]

        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(f"test add documents with truncated image for {index_name}"):
                r = tensor_search.add_documents(config=self.config,
                                                add_docs_params=AddDocsParams(index_name=index_name,
                                                                              docs=documents,
                                                                              tensor_fields=tensor_fields))
                self.assertEqual(True, r["errors"])
                self.assertEqual(2, len(r["items"]))
                self.assertEqual(200, r["items"][0]["status"])
                self.assertEqual(400, r["items"][1]["status"])
                self.assertIn("image file is truncated", r["items"][1]["error"])

    def test_add_document_callVectoriseWithoutPassingEnableCache(self):
        """Ensure vectorise does not receive enable_cache when calling add_documents."""
        documents = [
            {
                "text_field_1": "Test test",
                "_id": "1"
            }
        ]
        dummy_return = [[1.0, ] * 512, ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                with patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return) as mock_vectorise:
                    r = tensor_search.add_documents(config=self.config,
                                                    add_docs_params=AddDocsParams(index_name=index_name,
                                                                                  docs=documents,
                                                                                  tensor_fields=tensor_fields))
                    mock_vectorise.assert_called_once()
                    args, kwargs = mock_vectorise.call_args
                    self.assertFalse("enable_cache" in kwargs, "enable_cache should not be passed to "
                                                               "vectorise for add_documents")
                mock_vectorise.reset_mock()

    def test_imageRepoHandleConnectionError_successfully(self):
        """Ensure ConnectionResetError is not causing 500 error, but 400 for the document in add_documents."""
        documents = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/"
                                 "marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "1"
            }
        ]

        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            for error in (ConnectionResetError, ProtocolError):
                tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                    else None
                with (self.subTest(f"{index_name}-{error}")):
                    with patch("marqo.s2_inference.clip_utils.requests.get", side_effect=error) \
                            as mock_requests_get:
                        r = tensor_search.add_documents(config=self.config,
                                                        add_docs_params=AddDocsParams(index_name=index_name,
                                                                                      docs=documents,
                                                                                      tensor_fields=tensor_fields))
                    mock_requests_get.assert_called_once()
                    self.assertEqual(True, r["errors"])
                    self.assertEqual(1, len(r["items"]))
                    self.assertEqual(400, r["items"][0]["status"])
                    self.assertIn(str(error.__name__), r["items"][0]["error"])

    def test_imageRepoHandleThreadHandleError_successfully(self):
        """Ensure image_repo can catch an unexpected error right in thread."""
        documents = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/"
                                 "marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "1"
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.structured_marqo_index_name]:
            error = Exception("Unexpected error during image download")
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with (self.subTest(f"{index_name}-{error}")):
                with patch("marqo.s2_inference.clip_utils.requests.get", side_effect=error) \
                        as mock_requests_get:
                    with self.assertRaises(Exception) as e:
                        r = tensor_search.add_documents(config=self.config,
                                                        add_docs_params=AddDocsParams(index_name=index_name,
                                                                                      docs=documents,
                                                                                      tensor_fields=tensor_fields))
                        self.assertIn("Unexpected error during image download", str(e.exception))
