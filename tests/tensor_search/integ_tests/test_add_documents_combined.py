import os
import uuid
from unittest import mock
from unittest.mock import patch
import pytest
import torch


import PIL
import requests
import torch
from torch import Tensor
from urllib3.exceptions import ProtocolError
import unittest.mock


from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference import types
from marqo.tensor_search import add_docs
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.multimodal_model_load import infer_modality, Modality
from marqo.tensor_search import streaming_media_processor


class TestAddDocumentsCombined(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_image_index_request = cls.structured_marqo_index_request(
            name="structured_image_index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
                FieldRequest(
                    name="multimodal_field", 
                    type=FieldType.MultimodalCombination,
                    dependent_fields={
                        "image_field_1": 1.0,
                        "text_field_1": 0.0
                    }
                )
            ],
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            tensor_fields=["image_field_1", "text_field_1", "multimodal_field"]
        )

        structured_languagebind_index_request = cls.structured_marqo_index_request(
            name="my-multimodal-index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="text_field_2", type=FieldType.Text),
                FieldRequest(name="text_field_3", type=FieldType.Text),
                FieldRequest(name="video_field_1", type=FieldType.VideoPointer),
                FieldRequest(name="video_field_2", type=FieldType.VideoPointer),
                FieldRequest(name="video_field_3", type=FieldType.VideoPointer),
                FieldRequest(name="audio_field_1", type=FieldType.AudioPointer),
                FieldRequest(name="audio_field_2", type=FieldType.AudioPointer),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(
                    name="multimodal_field",
                    type=FieldType.MultimodalCombination,
                    dependent_fields={
                        "text_field_1": 0.1,
                        "text_field_2": 0.1,
                        "image_field_1": 0.5,
                        "video_field_1": 0.1,
                        "video_field_2": 0.1,
                        "audio_field_1": 0.1
                    }
                )
            ],
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            tensor_fields=["multimodal_field", "text_field_3",
                        "video_field_3", "audio_field_2", "image_field_2"],
            normalize_embeddings=True,
        )

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_languagebind_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_languagebind_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True
        )

        cls.indexes = cls.create_indexes([
            structured_image_index_request,
            structured_languagebind_index_request,
            unstructured_image_index_request,
            unstructured_languagebind_index_request
        ])

        cls.structured_marqo_index_name = structured_image_index_request.name
        cls.structured_languagebind_index_name = structured_languagebind_index_request.name
        cls.unstructured_marqo_index_name = unstructured_image_index_request.name
        cls.unstructured_languagebind_index_name = unstructured_languagebind_index_request.name

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
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image2.jpg",
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
                r = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=documents,
                        tensor_fields=tensor_fields,)
                ).dict(exclude_none=True, by_alias=True)
                print(f"response: {r}")
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
                    r = tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=documents,
                            tensor_fields=tensor_fields)
                    ).dict(exclude_none=True, by_alias=True)
                    self.assertTrue(mock_vectorise.called)
                    args, kwargs = mock_vectorise.call_args
                    self.assertFalse("enable_cache" in kwargs, "enable_cache should not be passed to "
                                                               "vectorise for add_documents")
                mock_vectorise.reset_mock()

    @pytest.mark.skipif(torch.cuda.is_available() is True, reason="GPU testing device needs to be investigated")
    def test_add_multimodal_single_documents(self):
        """ """
        documents = [
            {
                "video_field_3": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
                "_id": "1"
            },
            {
                "audio_field_2": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3",
                "_id": "2"
            },
            {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "3"
            },
            {
                "text_field_3": "hello there padawan. Today you will begin your training to be a Jedi",
                "_id": "4"
            },
        ]
        for index_name in [self.structured_languagebind_index_name, self.unstructured_languagebind_index_name]:
            with self.subTest(index_name):
                res = tensor_search.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=["text_field_3", "image_field_2", "video_field_3",
                                       "audio_field_2"] if "unstructured" in index_name else None
                    )
                )
                print(res)
                for item in res.dict(exclude_none=True, by_alias=True)['items']:
                    self.assertEqual(200, item['status'])

                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index_name,
                    document_ids=["1", "2", "3", "4"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                for i, doc in enumerate(get_res['results']):
                    i += 1
                    tensor_facets = doc['_tensor_facets']
                    print("tensor_facets count", i, len(tensor_facets), doc.keys())
                    # Check the length of tensor facets
                    if i in [1, 3, 4]:
                        self.assertEqual(len(tensor_facets), 1, f"Document {i} should have 1 tensor facet")
                    elif i == 2:
                        # print(tensor_facets)
                        self.assertEqual(len(tensor_facets), 10, f"Document 2 should have 10 tensor facets")

                    # Check embedding length and uniqueness
                    embeddings = []
                    for facet in tensor_facets:
                        embedding = facet['_embedding']
                        self.assertEqual(len(embedding), 768, f"Embedding length should be 768 for document {i}")
                        self.assertNotIn(embedding, embeddings, f"Duplicate embedding found in document {i}")
                        embeddings.append(embedding)

    @pytest.mark.skipif(torch.cuda.is_available() is True, reason="GPU testing device needs to be investigated")
    def test_add_multimodal_field_document(self):
        multimodal_document = {
            "_id": "1_multimodal",
            "text_field_1": "New York",
            "text_field_2": "Los Angeles",
            "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "video_field_1": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "video_field_2": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "audio_field_1": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3",
        },
        for index_name in [self.structured_languagebind_index_name, self.unstructured_languagebind_index_name]:
            mappings = {
                "multimodal_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "text_field_1": 0.1,
                        "text_field_2": 0.1,
                        "image_field": 0.5,
                        "video_field_1": 0.1,
                        "video_field_2": 0.1,
                        "audio_field_1": 0.1
                    },
                }
            } if "unstructured" in index_name else None
            res = tensor_search.add_documents(
                self.config,
                add_docs_params=AddDocsParams(
                    docs=multimodal_document,
                    index_name=index_name,
                    tensor_fields=["multimodal_field"] if "unstructured" in index_name else None,
                    mappings=mappings
                )
            )

            print(res)

            doc = tensor_search.get_documents_by_ids(
                config=self.config,
                index_name=index_name,
                document_ids=["1_multimodal"],
                show_vectors=True
            ).dict(exclude_none=True, by_alias=True)

            print(doc)

            for item in res.dict(exclude_none=True, by_alias=True)['items']:
                self.assertEqual(200, item['status'])


    def test_imageRepoHandleThreadHandleError_successfully(self):
        """Ensure media_repo can catch an unexpected error right in thread."""
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
                        r = tensor_search.add_documents(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=index_name,
                                docs=documents,
                                tensor_fields=tensor_fields)
                        ).dict(exclude_none=True, by_alias=True)
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
                    r = tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=documents,
                            tensor_fields=tensor_fields)
                    ).dict(exclude_none=True, by_alias=True)
                    # Check that vectorise was called at least once
                    self.assertTrue(mock_vectorise.called)
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
             "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
             } for i in range(10)
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for thread_count in [2, 5]:
                    with patch.object(
                            add_docs, 'threaded_download_and_preprocess_content',
                            wraps=add_docs.threaded_download_and_preprocess_content
                    ) as mock_download_images:
                        tensor_search.add_documents(
                            config=self.config, add_docs_params=AddDocsParams(
                                index_name=index_name, docs=docs, device="cpu",
                                image_download_thread_count=thread_count,
                                tensor_fields=tensor_fields
                            )
                        ).dict(exclude_none=True, by_alias=True)

                        self.assertEqual(thread_count, mock_download_images.call_count)

    def test_image_url_is_embedded_as_image_not_text(self):
        """
        Ensure that the image URL is embedded as an image and not as text
        """
        docs = [
            {"_id": "1",
             "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
             }
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )

                doc = tensor_search.get_documents_by_ids(
                    config=self.config,
                    index_name=index_name,
                    document_ids=["1"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                # Assert that the vector is similar to expected_vector
                expected_vector = [-0.06504671275615692, -0.03672310709953308, -0.06603428721427917,
                                   -0.032505638897418976, -0.06116769462823868, -0.03929287940263748]
                actual_vector = doc['results'][0]['_tensor_facets'][0]['_embedding']
                
                for i, expected_value in enumerate(expected_vector):
                    self.assertAlmostEqual(actual_vector[i], expected_value, places=5)

    def test_multimodal_image_url_is_embedded_as_image_not_text(self):
        """
        Ensure that the image URL in a multimodal field is embedded as an image and not as text
        """
        docs = [
            {
                "_id": "1",
                "text_field_1": "This text should be ignored",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            }
        ]

        # Expected vector for the LanguageBind model (adjust these values based on actual output)
        expected_vector = [-0.06504671275615692, -0.03672310709953308, -0.06603428721427917,
                           -0.032505638897418976, -0.06116769462823868, -0.03929287940263748]

        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            with self.subTest(index_name):
                # For unstructured index, we need to define the multimodal field and its weights
                if "unstructured" in index_name:
                    tensor_fields = ["multimodal_field"]
                    mappings = {
                        "multimodal_field": {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field_1": 0.0,
                                "image_field_1": 1.0,  # Only consider the image
                            }
                        }
                    }
                else:
                    tensor_fields = None
                    mappings = None

                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=docs,
                        tensor_fields=tensor_fields,
                        mappings=mappings
                    )
                )

                doc = tensor_search.get_documents_by_ids(
                    config=self.config,
                    index_name=index_name,
                    document_ids=["1"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                # Get the actual vector
                actual_vector = doc['results'][0]['_tensor_facets'][0]['_embedding']

                # Assert that the vector is similar to expected_vector
                for i, expected_value in enumerate(expected_vector):
                    self.assertAlmostEqual(actual_vector[i], expected_value, places=4,
                                        msg=f"Mismatch at index {i} for {index_name}")

    def test_imageDownloadWithoutPreprocessor(self):
        media_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            media_repo=media_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
            marqo_index_type=IndexType.Unstructured,
            marqo_index_model=Model(name="test", properties={}),
        )
        assert len(media_repo) == 2
        assert isinstance(media_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(media_repo[good_url], types.ImageType)

    def test_imageDownloadWithPreprocessor(self):
        media_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            media_repo=media_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
            preprocessors={'image': lambda x: torch.randn(3, 224, 224)},
            device='cpu',
            marqo_index_type=IndexType.Unstructured,
            marqo_index_model=Model(name="test", properties={}),
        )
        assert len(media_repo) == 2
        assert isinstance(media_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(media_repo[good_url], Tensor)

    def test_image_download_timeout(self):
        mock_get = mock.MagicMock()
        mock_get.side_effect = requests.exceptions.RequestException

        @mock.patch('requests.get', mock_get)
        def run():
            media_repo = dict()
            add_docs.threaded_download_and_preprocess_content(
                allocated_docs=[
                    {"Title": "frog", "Desc": "blah"}, {"Title": "Dog", "Loc": "https://google.com/my_dog.png"}],
                media_repo=media_repo,
                tensor_fields=['Title', 'Desc', 'Loc'],
                image_download_headers={},
                marqo_index_type=IndexType.Unstructured,
                marqo_index_model=Model(name="test", properties={}),
            )
            assert list(media_repo.keys()) == ['https://google.com/my_dog.png']
            assert isinstance(media_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
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
                            index_name=index_name, docs=docs, device="cpu", tensor_fields=tensor_fields)).dict(
                            exclude_none=True, by_alias=True)
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
            media_repo = dict()
            add_docs.threaded_download_and_preprocess_content(
                allocated_docs=docs,
                media_repo=media_repo,
                tensor_fields=['field_1', 'field_2'],
                image_download_headers={},
                marqo_index_type=IndexType.Unstructured,
                marqo_index_model=Model(name="test", properties={}),
            )
            assert len(expected_repo_structure) == len(media_repo)
            for k in expected_repo_structure:
                assert isinstance(media_repo[k], expected_repo_structure[k])

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
        model_properties = (
            {
                "name": "ViT-B/32",
                "dimensions": 512,
                "notes": "CLIP ViT-B/32",
                "type": "clip",
            }
        )

        for docs, expected_repo_structure in examples:
            with mock.patch('PIL.Image.Image.close') as mock_close:
                with add_docs.download_and_preprocess_content(
                    docs=docs,
                    thread_count=20,
                    tensor_fields=['field_1', 'field_2'],
                    image_download_headers={},
                    model_name="ViT-B/32",
                    normalize_embeddings=True,
                    model_properties=model_properties,
                    media_field_types_mapping=None,
                    device="cpu",
                    marqo_index_type=IndexType.Unstructured,
                    marqo_index_model=Model(name="test", properties={}),
                ) as media_repo:
                    self.assertEqual(len(expected_repo_structure), len(media_repo))
                    for k in expected_repo_structure:
                        print(f"expected_repo_structure[k] = {expected_repo_structure[k]}")
                        print(f"media_repo[k] = {media_repo[k]}")
                        self.assertIsInstance(media_repo[k], expected_repo_structure[k])

            # Images should not be closed as they are Tensor instead of ImageType
            mock_close.assert_not_called()

    def test_idErrorWhenImageDownloading(self):
        """A test ensure image download is not raising 500 error when there is an invalid _id.

        Image download use the document _id to generate a unique thread id.
        However, the image download happens before validate the document _id.
        This test ensures that the image download does not raise a 500 error when the document _id is invalid.
        """
        test_docs = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image1.jpg",
                 "text_field_1": "this is a valid image",
                 "_id": "1"
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image2.jpg",
                "text_field_1": "this is a invalid image due to int id",
                "_id": 2
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image3.jpg",
                "text_field_1": "this is a invalid image due to None",
                "_id": None
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image4.jpg",
                "text_field_1": "this is a invalid image due to ",
                "_id": []
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.structured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                r = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=test_docs,
                        tensor_fields=tensor_fields)
                ).dict(exclude_none=True, by_alias=True)
                self.assertEqual(True, r["errors"])
                self.assertEqual(4, len(r["items"]))
                self.assertEqual(200, r["items"][0]["status"])
                for i in range(1, 4):
                    self.assertEqual(400, r["items"][i]["status"])
                    self.assertIn("Document _id must be a string", r["items"][i]["error"])


    @unittest.mock.patch('marqo.tensor_search.streaming_media_processor.ffmpeg')
    @unittest.mock.patch('marqo.tensor_search.streaming_media_processor.tempfile.TemporaryDirectory')
    def test_process_media_chunk_calculation(self, mock_temp_dir, mock_ffmpeg):
        # Mock the TemporaryDirectory context manager
        mock_temp_dir.return_value.__enter__.return_value = '/tmp/mock_dir'

        # Create a mock MarqoIndex
        mock_index = unittest.mock.Mock()
        mock_index.video_preprocessing = unittest.mock.Mock(split_length=10, split_overlap=2)

        # Create a StreamingMediaProcessor instance with mocked values
        processor = streaming_media_processor.StreamingMediaProcessor(
            url='http://example.com/video.mp4',
            device='cpu',
            headers={},
            modality=streaming_media_processor.Modality.VIDEO,
            marqo_index_type=IndexType.Unstructured,
            marqo_index_model=Model(name="test", properties={}),
            audio_preprocessing=unittest.mock.Mock(),
            video_preprocessing=unittest.mock.Mock(),
            preprocessors={'video': unittest.mock.Mock()}
        )

        # Set arbitrary values
        processor.duration = 25  # 25 seconds video
        processor.split_length = 10  # 10 seconds per chunk
        processor.split_overlap = 2  # 2 seconds overlap

        # Mock the preprocessor to return a dummy tensor
        processor.preprocessor = unittest.mock.Mock(return_value={'pixel_values': unittest.mock.Mock()})

        # Call the process_media method
        result = processor.process_media()

        # Expected chunk calculations
        expected_chunks = [
            {'start_time': 0, 'end_time': 10},
            {'start_time': 8, 'end_time': 18},
            {'start_time': 15, 'end_time': 25}  # Last chunk adjusted to video end
        ]

        # Assert the number of chunks
        self.assertEqual(len(result), len(expected_chunks))

        # Assert the start and end times of each chunk
        for i, chunk in enumerate(result):
            self.assertEqual(chunk['start_time'], expected_chunks[i]['start_time'])
            self.assertEqual(chunk['end_time'], expected_chunks[i]['end_time'])

        # Verify that ffmpeg.input was called for each chunk
        self.assertEqual(mock_ffmpeg.input.call_count, len(expected_chunks))

        # Verify the ffmpeg.input calls
        for i, expected_chunk in enumerate(expected_chunks):
            mock_ffmpeg.input.assert_any_call(
                'http://example.com/video.mp4',
                ss=expected_chunk['start_time'],
                t=expected_chunk['end_time'] - expected_chunk['start_time']
            )

        # Verify that ffmpeg.run was called for each chunk
        self.assertEqual(mock_ffmpeg.run.call_count, len(expected_chunks))

    def test_webp_image_download_infer_modality(self):
        """the webp extension is not predefined among the extensions in infer_modality.
        this test ensures that the webp extension is correctly inferred as an image"""
        webp_image_url = "https://i.ebayimg.com/images/g/UawAAOSwpd5iR9Bs/s-l1600.webp"
        modality = infer_modality(webp_image_url)
        self.assertEqual(modality, streaming_media_processor.Modality.IMAGE)

    def test_no_extension_image_url_infer_modality(self):
        """this test ensures that the image url with no extension is correctly inferred as an image"""
        image_url_no_extension = "https://il.redbubble.net/catalogue/image/by-rb-work/157037551/simple-preview"
        modality = infer_modality(image_url_no_extension)
        self.assertEqual(modality, streaming_media_processor.Modality.IMAGE)