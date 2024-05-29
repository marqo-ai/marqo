import os
import uuid
from unittest import mock
from unittest.mock import patch

import PIL
import requests
import torch
from torch import Tensor
from urllib3.exceptions import ProtocolError

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference import types
from marqo.tensor_search import add_docs
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestAddDocumentsCombined(MarqoTestCase):
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

    def test_addDocumentsPassTensorToVectorise(self):
        """Ensure vectorise receives tensor from add_documents when the model is OpenCLIP or CLIP."""
        documents = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/"
                                 "marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "1"
            }
        ]
        dummy_return = [[1.0, ] * 512, ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                with patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return) as mock_vectorise:
                    r = tensor_search.add_documents(config=self.config,
                                                    add_docs_params=AddDocsParams(index_name=index_name,
                                                                                  docs=documents,
                                                                                  tensor_fields=tensor_fields))
                    mock_vectorise.assert_called_once()
                    args, kwargs = mock_vectorise.call_args
                    self.assertIn("content", kwargs)
                    content = kwargs["content"]
                    self.assertEqual(1, len(content))
                    self.assertEqual((3, 224, 224), content[0].shape)

    def test_downloadImagesThreadCount(self):
        """
        Test that image download thread count is respected
        """
        docs = [
            {"_id": str(i),
             "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/"
                            "assets/ai_hippo_realistic.png"
             } for i in range(10)
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for thread_count in [2, 5]:
                    with patch.object(
                            add_docs, 'threaded_download_and_preprocess_images',
                            wraps=add_docs.threaded_download_and_preprocess_images
                    ) as mock_download_images:
                        tensor_search.add_documents(
                            config=self.config, add_docs_params=AddDocsParams(
                                index_name=index_name, docs=docs, device="cpu",
                                image_download_thread_count=thread_count,
                                tensor_fields=tensor_fields
                            )
                        )

                        self.assertEqual(thread_count, mock_download_images.call_count)

    def test_imageDownloadWithoutPreprocessor(self):
        image_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_images(
            allocated_docs=[test_doc],
            image_repo=image_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
        )
        assert len(image_repo) == 2
        assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(image_repo[good_url], types.ImageType)

    def test_imageDownloadWithPreprocessor(self):
        image_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_images(
            allocated_docs=[test_doc],
            image_repo=image_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
            preprocessor=lambda x: torch.randn(3, 224, 224),
            device='cpu'
        )
        assert len(image_repo) == 2
        assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(image_repo[good_url], Tensor)

    def test_image_download_timeout(self):
        mock_get = mock.MagicMock()
        mock_get.side_effect = requests.exceptions.RequestException

        @mock.patch('requests.get', mock_get)
        def run():
            image_repo = dict()
            add_docs.threaded_download_and_preprocess_images(
                allocated_docs=[
                    {"Title": "frog", "Desc": "blah"}, {"Title": "Dog", "Loc": "https://google.com/my_dog.png"}],
                image_repo=image_repo,
                tensor_fields=['Title', 'Desc', 'Loc'],
                image_download_headers={}
            )
            assert list(image_repo.keys()) == ['https://google.com/my_dog.png']
            assert isinstance(image_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
            return True

        assert run()

    def test_resilient_add_images(self):
        """
        Various image URLs are handled correctly
        """
        docs_results = [
            ([{"_id": "123",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", 200), ("789", 200), ("456", 400)]
             ),
            ([{"_id": "123",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("123", 200), ("789", 200), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist3.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400)]
             ),
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for docs, expected_results in docs_results:
                    with self.subTest(f'{expected_results} - {index_name}'):
                        add_res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                            index_name=index_name, docs=docs, device="cpu", tensor_fields=tensor_fields))
                        self.assertEqual(len(expected_results), len(add_res['items']))
                        for i, res_dict in enumerate(add_res['items']):
                            self.assertEqual(expected_results[i][0], res_dict["_id"], res_dict)
                            self.assertEqual(expected_results[i][1], res_dict['status'], res_dict)

    def test_threaded_download_images_non_tensor_field(self):
        """Tests add_docs.threaded_download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: types.ImageType
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: types.ImageType
             }),
        ]
        for docs, expected_repo_structure in examples:
            image_repo = dict()
            add_docs.threaded_download_and_preprocess_images(
                allocated_docs=docs,
                image_repo=image_repo,
                tensor_fields=['field_1', 'field_2'],
                image_download_headers={}
            )
            assert len(expected_repo_structure) == len(image_repo)
            for k in expected_repo_structure:
                assert isinstance(image_repo[k], expected_repo_structure[k])

    def test_download_images_non_tensor_field(self):
        """tests add_docs.download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: Tensor
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: Tensor
             }),
        ]
        for docs, expected_repo_structure in examples:
            with mock.patch('PIL.Image.Image.close') as mock_close:
                with add_docs.download_and_preprocess_images(
                    docs=docs,
                    thread_count=20,
                    tensor_fields=['field_1', 'field_2'],
                    image_download_headers={},
                    model_name="ViT-B/32",
                    normalize_embeddings=True,
                    model_properties=None,
                    device="cpu"
                ) as image_repo:
                    self.assertEqual(len(expected_repo_structure), len(image_repo))
                    for k in expected_repo_structure:
                        self.assertIsInstance(image_repo[k], expected_repo_structure[k])

            # Images should not be closed as they are Tensor instead of ImageType
            mock_close.assert_not_called()