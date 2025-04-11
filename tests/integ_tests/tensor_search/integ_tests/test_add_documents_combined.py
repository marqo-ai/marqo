import os
import unittest
import uuid
from unittest import mock
from unittest.mock import patch

import PIL
import numpy as np
import pytest
import requests
import torch
from torch import Tensor

from integ_tests.marqo_test import MarqoTestCase, TestImageUrls, TestAudioUrls, TestVideoUrls
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference import types
from marqo.core.inference.modality_utils import infer_modality
from marqo.tensor_search import add_docs
from marqo.tensor_search import streaming_media_processor
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.preprocessors_model import Preprocessors


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
                FieldRequest(name="text_field_2", type=FieldType.Text,
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
            tensor_fields=["image_field_1", "text_field_1", "text_field_2", "multimodal_field"]
        )

        structured_image_index_request_unnormalized = cls.structured_marqo_index_request(
            name="structured_image_index_unnormalised" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
            ],
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            tensor_fields=["image_field_1", "text_field_1"],
            normalize_embeddings=False,
            distance_metric=DistanceMetric.DotProduct
        )

        structured_text_index_request_unnormalized = cls.structured_marqo_index_request(
            name="structured_image_index_unnormalised" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(
                    name="text_field_1", type=FieldType.Text,
                    features=[FieldFeature.Filter, FieldFeature.LexicalSearch]
                ),
            ],
            model=Model(name="hf/e5-base-v2"),
            tensor_fields=["text_field_1"],
            normalize_embeddings=False,
            distance_metric=DistanceMetric.DotProduct
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

        semi_structured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )

        semi_structured_languagebind_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_languagebind_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True
        )

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True,
            marqo_version='2.12.0'
        )

        unstructured_languagebind_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_languagebind_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True,
            marqo_version='2.12.0'
        )

        unstructured_image_index_request_unnormalized = cls.unstructured_marqo_index_request(
            name="unstructured_image_index_unnormalised" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            normalize_embeddings=False,
            distance_metric=DistanceMetric.DotProduct
        )

        unstructured_text_index_request_unnormalized = cls.unstructured_marqo_index_request(
            name="unstructured_image_index_unnormalised" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/e5-base-v2"),
            normalize_embeddings=False,
            distance_metric=DistanceMetric.DotProduct
        )

        cls.indexes = cls.create_indexes([
            structured_image_index_request,
            semi_structured_image_index_request,
            unstructured_image_index_request,

            structured_languagebind_index_request,
            semi_structured_languagebind_index_request,
            unstructured_languagebind_index_request,

            unstructured_image_index_request_unnormalized,
            unstructured_text_index_request_unnormalized,
            structured_image_index_request_unnormalized,
            structured_text_index_request_unnormalized
        ])

        cls.structured_marqo_index_name = structured_image_index_request.name
        cls.structured_languagebind_index_name = structured_languagebind_index_request.name
        cls.semi_structured_marqo_index_name = semi_structured_image_index_request.name
        cls.semi_structured_languagebind_index_name = semi_structured_languagebind_index_request.name
        cls.structured_image_index_unnormalized_name = structured_image_index_request_unnormalized.name
        cls.structured_text_index_unnormalized_name = structured_text_index_request_unnormalized.name

        cls.unstructured_marqo_index_name = unstructured_image_index_request.name
        cls.unstructured_languagebind_index_name = unstructured_languagebind_index_request.name
        cls.unstructured_image_index_unnormalized_name = unstructured_image_index_request_unnormalized.name
        cls.unstructured_text_index_unnormalized_name = unstructured_text_index_request_unnormalized.name

        cls.image_indexes = cls.indexes[:3]
        cls.languagebind_indexes = cls.indexes[3:6]

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
                "image_field_1": TestImageUrls.IMAGE2.value,
                "text_field_1": "This is a valid image",
                "_id": "1"
            },
            {
                "image_field_1": truncated_image_url,
                "text_field_1": "This is a truncated image",
                "_id": "2"
            }
        ]

        for index_name in [self.structured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name != self.structured_marqo_index_name \
                else None
            with self.subTest(f"test add documents with truncated image for {index_name}"):
                r = self.add_documents(
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
                self.assertIn("Image file is truncated", r["items"][1]["error"])

    @unittest.skip(reason="Temporarily skipped due to no support for languagebind model")
    @pytest.mark.largemodel
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
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
                "image_field_2": TestImageUrls.HIPPO_REALISTIC_LARGE.value,
                "_id": "3"
            },
            {
                "text_field_3": "hello there padawan. Today you will begin your training to be a Jedi",
                "_id": "4"
            },
        ]
        for index_name in [self.structured_languagebind_index_name, self.semi_structured_languagebind_index_name,
                           self.unstructured_languagebind_index_name]:
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=["text_field_3", "image_field_2", "video_field_3",
                                       "audio_field_2"] if index_name != self.structured_languagebind_index_name else None
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

    @unittest.skip(reason="Temporarily skipped due to no support for languagebind model")
    @pytest.mark.largemodel
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
    def test_add_multimodal_field_document(self):
        multimodal_document = {
            "_id": "1_multimodal",
            "text_field_1": "New York",
            "text_field_2": "Los Angeles",
            "image_field_1": TestImageUrls.HIPPO_REALISTIC_LARGE.value,
            # "image_field_2": TestImageUrls.HIPPO_REALISTIC.value, # png image with palette is not supported
            "video_field_1": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "video_field_2": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "audio_field_1": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3",
        },
        for index_name in [self.structured_languagebind_index_name, self.semi_structured_languagebind_index_name,
                           self.unstructured_languagebind_index_name]:
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
            res = self.add_documents(
                self.config,
                add_docs_params=AddDocsParams(
                    docs=multimodal_document,
                    index_name=index_name,
                    tensor_fields=["multimodal_field"] if index_name != self.structured_languagebind_index_name else None,
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
                "image_field_1": TestImageUrls.HIPPO_REALISTIC.value,
                "_id": "1"
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.structured_marqo_index_name]:
            error = Exception("Unexpected error during image download")
            tensor_fields = ["image_field_1"] if index_name != self.structured_marqo_index_name \
                else None
            with (self.subTest(f"{index_name}-{error}")):
                with patch("marqo.s2_inference.clip_utils.requests.get", side_effect=error) \
                        as mock_requests_get:
                    with self.assertRaises(Exception) as e:
                        r = self.add_documents(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=index_name,
                                docs=documents,
                                tensor_fields=tensor_fields)
                        ).dict(exclude_none=True, by_alias=True)
                        self.assertIn("Unexpected error during image download", str(e.exception))

    def test_image_url_is_embedded_as_image_not_text(self):
        """
        Ensure that the image URL is embedded as an image and not as text
        """
        docs = [
            {"_id": "1",
             "image_field_1": TestImageUrls.IMAGE2.value
             }
        ]
        for index_name in [self.structured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name != self.structured_marqo_index_name \
                else None
            with self.subTest(index_name):
                res = self.add_documents(
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
                "image_field_1": TestImageUrls.IMAGE2.value,
            }
        ]

        # Expected vector for the LanguageBind model (adjust these values based on actual output)
        expected_vector = [-0.06504671275615692, -0.03672310709953308, -0.06603428721427917,
                           -0.032505638897418976, -0.06116769462823868, -0.03929287940263748]

        for index_name in [self.structured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.unstructured_marqo_index_name]:
            with self.subTest(index_name):
                # For unstructured index, we need to define the multimodal field and its weights
                if index_name != self.structured_marqo_index_name:
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

                res = self.add_documents(
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
        good_url = TestImageUrls.HIPPO_REALISTIC.value
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            media_repo=media_repo,
            tensor_fields=['field_1', 'field_2'],
            media_download_headers={},
            marqo_index_type=IndexType.Unstructured,
            marqo_index_model=Model(name="test", properties={}),
        )
        assert len(media_repo) == 2
        assert isinstance(media_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(media_repo[good_url], types.ImageType)

    def test_imageDownloadWithPreprocessor(self):
        media_repo = dict()
        good_url = TestImageUrls.HIPPO_REALISTIC.value
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            media_repo=media_repo,
            tensor_fields=['field_1', 'field_2'],
            media_download_headers={},
            preprocessors=Preprocessors(**{'image': lambda x: torch.randn(3, 224, 224)}),
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
                media_download_headers={},
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
               "image_field_1": TestImageUrls.HIPPO_REALISTIC.value},
              {"_id": "789",
               "image_field_1": TestImageUrls.HIPPO_STATUE.value},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", 200), ("789", 200), ("456", 400)]
             ),
            ([{"_id": "123",
               "image_field_1": TestImageUrls.HIPPO_REALISTIC.value},
              {"_id": "789",
               "image_field_1": TestImageUrls.HIPPO_STATUE.value},
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
        for index_name in [self.structured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name != self.structured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for docs, expected_results in docs_results:
                    with self.subTest(f'{expected_results} - {index_name}'):
                        add_res = self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                            index_name=index_name, docs=docs, device="cpu", tensor_fields=tensor_fields)).dict(
                            exclude_none=True, by_alias=True)
                        self.assertEqual(len(expected_results), len(add_res['items']))
                        for i, res_dict in enumerate(add_res['items']):
                            self.assertEqual(expected_results[i][0], res_dict["_id"], res_dict)
                            self.assertEqual(expected_results[i][1], res_dict['status'], res_dict)

    def test_threaded_download_images_non_tensor_field(self):
        """Tests add_docs.threaded_download_images(). URLs not in tensor fields should not be downloaded """
        good_url = TestImageUrls.HIPPO_REALISTIC.value
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
                media_download_headers={},
                marqo_index_type=IndexType.Unstructured,
                marqo_index_model=Model(name="test", properties={}),
            )
            assert len(expected_repo_structure) == len(media_repo)
            for k in expected_repo_structure:
                assert isinstance(media_repo[k], expected_repo_structure[k])

    def test_idErrorWhenImageDownloading(self):
        """A test ensure image download is not raising 500 error when there is an invalid _id.

        Image download use the document _id to generate a unique thread id.
        However, the image download happens before validate the document _id.
        This test ensures that the image download does not raise a 500 error when the document _id is invalid.
        """
        test_docs = [
            {
                "image_field_1": TestImageUrls.IMAGE1.value,
                 "text_field_1": "this is a valid image",
                 "_id": "1"
            },
            {
                "image_field_1": TestImageUrls.IMAGE2.value,
                "text_field_1": "this is a invalid image due to int id",
                "_id": 2
            },
            {
                "image_field_1": TestImageUrls.IMAGE3.value,
                "text_field_1": "this is a invalid image due to None",
                "_id": None
            },
            {
                "image_field_1": TestImageUrls.IMAGE4.value,
                "text_field_1": "this is a invalid image due to ",
                "_id": []
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.semi_structured_marqo_index_name,
                           self.structured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name != self.structured_marqo_index_name \
                else None
            with self.subTest(index_name):
                r = self.add_documents(
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

    def test_imageIndexEmbeddingsUnnormalised(self):
        """Test to ensure that the image embeddings are unnormalised when the index is unnormalised"""
        documents = [
            {
                "image_field_1": TestImageUrls.HIPPO_REALISTIC.value,
                "_id": "1"
            }
        ]
        for index_name in [self.unstructured_image_index_unnormalized_name, self.structured_image_index_unnormalized_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_image_index_unnormalized_name \
                else None
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields
                    )
                )
                for item in res.dict(exclude_none=True, by_alias=True)['items']:
                    self.assertEqual(200, item['status'])

                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index_name,
                    document_ids=["1"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                embeddings = get_res['results'][0]['_tensor_facets'][0]['_embedding']
                norm = np.linalg.norm(np.array(embeddings))
                self.assertTrue(norm - 1.0 > 1e-5, f"Embedding norm is {norm}")

    def test_imageIndexEmbeddingsNormalised(self):
        """Test to ensure that the image embeddings are normalised when the index is normalised"""

        documents = [
            {
                "image_field_1": TestImageUrls.HIPPO_REALISTIC.value,
                "_id": "1"
            }
        ]
        for index_name in [self.unstructured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields
                    )
                )
                for item in res.dict(exclude_none=True, by_alias=True)['items']:
                    self.assertEqual(200, item['status'])

                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index_name,
                    document_ids=["1"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                embeddings = get_res['results'][0]['_tensor_facets'][0]['_embedding']
                norm = np.linalg.norm(np.array(embeddings))
                self.assertTrue(norm - 1.0 < 1e-5, f"Embedding norm is {norm}")

    def test_textIndexEmbeddingsUnnormalized(self):
        """A test to ensure that the text embeddings are unnormalised when the index is unnormalised"""
        documents = [
            {
                "text_field_1": "This is a test text",
                "_id": "1"
            }
        ]
        for index_name in [self.unstructured_text_index_unnormalized_name, self.structured_text_index_unnormalized_name]:
            tensor_fields = ["text_field_1"] if index_name == self.unstructured_text_index_unnormalized_name \
                else None
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields
                    )
                )
                for item in res.dict(exclude_none=True, by_alias=True)['items']:
                    self.assertEqual(200, item['status'])

                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index_name,
                    document_ids=["1"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)

                embeddings = get_res['results'][0]['_tensor_facets'][0]['_embedding']
                norm = np.linalg.norm(np.array(embeddings))
                self.assertTrue(norm - 1.0 > 1e-5, f"Embedding norm is {norm}")

    def test_add_private_images_proper_error_returned(self):
        """Test to ensure that private images can not be downloaded and an appropriate error is returned"""
        test_indexes = [self.structured_marqo_index_name, self.unstructured_marqo_index_name]
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]
        for index_name in test_indexes:
            tensor_fields = ["multimodal_field", "my_combination_field"] if (
                    index_name == self.unstructured_marqo_index_name) else None
            mappings = {
                "multimodal_field":
                    {
                        "type": "multimodal_combination",
                        "weights": {"image_field_1": 1.0, "text_field_1": 1.0}
                    }
            }
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields,
                        mappings=mappings
                    )
                )
                self.assertTrue(res.errors)
                items = res.items
                self.assertEqual(2, len(items))
                for item in items:
                    self.assertEqual(400, item.status)
                    self.assertIn("403", item.message)

    def test_add_private_images_success(self):
        """Test to ensure that private images can be downloaded with proper headers"""
        test_indexes = [self.structured_marqo_index_name, self.unstructured_marqo_index_name]
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]
        for index_name in test_indexes:
            tensor_fields = ["image_field_1", "multimodal_field"] if (
                    index_name == self.unstructured_marqo_index_name) else None
            mappings = {
                "multimodal_field":
                    {
                        "type": "multimodal_combination",
                        "weights": {"image_field_1": 1.0, "text_field_1": 1.0}
                    }
            }
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields,
                        media_download_headers={"marqo_media_header": "media_header_test_key"},
                        mappings=mappings
                    )
                )
                self.assertFalse(res.errors)

@unittest.skip(reason="Temporarily skipped due to no support for languagebind model")
@pytest.mark.largemodel
class TestLanguageBindModelAddDocumentCombined(MarqoTestCase):
    """A class to test the add_documents with the LanguageBind model."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_language_bind_index = cls.structured_marqo_index_request(
            name="structured_image_index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="audio_field_1", type=FieldType.AudioPointer),
                FieldRequest(name="video_field_1", type=FieldType.VideoPointer),
                FieldRequest(
                    name="multimodal_field",
                    type=FieldType.MultimodalCombination,
                    dependent_fields={
                        "image_field_1": 1.0,
                        "text_field_1": 1.0,
                        "audio_field_1": 1.0,
                        "video_field_1": 1.0,
                    }
                )
            ],
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            tensor_fields=["text_field_1", "image_field_1", "audio_field_1", "video_field_1", "multimodal_field"],
        )

        unstructured_language_bind_index = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True
        )

        unstructured_custom_language_bind_index = cls.unstructured_marqo_index_request(
            name="unstructured_image_index_custom" + str(uuid.uuid4()).replace('-', ''),
            model=Model(
                name="my-custom-language-bind-model",
                properties={
                    "dimensions": 768,
                    "type": "languagebind",
                    "supportedModalities": ["text", "audio", "video", "image"],
                    "modelLocation": {
                        "video": {
                            "hf": {
                                "repoId": "Marqo/LanguageBind_Video_V1.5_FT",
                            },
                        },
                        "audio": {
                            "hf": {
                                "repoId": "Marqo/LanguageBind_Audio_FT",
                            },
                        },
                        "image":{
                            "hf": {
                                "repoId": "Marqo/LanguageBind_Image",
                            },
                        }
                    },
                },
                custom=True
            ),
            treat_urls_and_pointers_as_images = True,
            treat_urls_and_pointers_as_media = True
        )

        cls.indexes = cls.create_indexes([structured_language_bind_index, unstructured_language_bind_index,
                                          unstructured_custom_language_bind_index])

        cls.structured_language_bind_index_name = structured_language_bind_index.name
        cls.unstructured_language_bind_index_name = unstructured_language_bind_index.name
        cls.unstructured_custom_language_bind_index_name= unstructured_custom_language_bind_index.name

        s2_inference.clear_loaded_models()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        s2_inference.clear_loaded_models()

    def test_language_bind_model_can_add_all_media_modalities(self):
        """Test to ensure that the LanguageBind model can add all media types to the index"""
        documents = [
            {
                "text_field_1": "This is a test text",
                "image_field_1": TestImageUrls.IMAGE1.value,
                "audio_field_1": TestAudioUrls.AUDIO1.value,
                "video_field_1": TestVideoUrls.VIDEO1.value,
                "_id": "1"
            }
        ]
        for index_name in [self.structured_language_bind_index_name, self.unstructured_language_bind_index_name]:
            tensor_fields = ["text_field_1", "image_field_1", "audio_field_1", "video_field_1", "multimodal_field"] \
                if index_name == self.unstructured_language_bind_index_name else None
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields
                    )
                )
                self.assertFalse(res.errors)

    def test_language_bind_model_can_add_all_private_media_modalities(self):
        documents = [
            {   # With extensions
                "text_field_1": "This is a test text",
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "audio_field_1": "https://d2k91vq0avo7lq.cloudfront.net/bark.wav",
                "video_field_1": "https://d2k91vq0avo7lq.cloudfront.net/congress.mp4",
                "_id": "1"
            },
            {
                # No extensions
                "text_field_1": "This is a test text",
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "audio_field_1": "https://d2k91vq0avo7lq.cloudfront.net/bark",
                "video_field_1": "https://d2k91vq0avo7lq.cloudfront.net/congress",
                "_id": "2"
            }
        ]
        for index_name in [self.structured_language_bind_index_name, self.unstructured_language_bind_index_name]:
            tensor_fields = ["text_field_1", "image_field_1", "audio_field_1", "video_field_1", "multimodal_field"] \
                if index_name == self.unstructured_language_bind_index_name else None
            with self.subTest(index_name):
                res = self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=documents,
                        index_name=index_name,
                        tensor_fields=tensor_fields,
                        media_download_headers={"marqo_media_header": "media_header_test_key"}
                    )
                )
                self.assertFalse(res.errors)

    def test_video_size_limit_in_batch(self):
        """Tests that adding documents with videos respects the file size limit per document"""
        with mock.patch.dict('os.environ', {'MARQO_MAX_ADD_DOCS_VIDEO_AUDIO_FILE_SIZE': '2097152',
                                            'MARQO_MAX_CPU_MODEL_MEMORY': '15',
                                            'MARQO_MAX_CUDA_MODEL_MEMORY': '15'}):  # 2MB limit
            # Test documents - one under limit (2.5MB), one over limit
            test_docs = [
                {
                    "_id": "1",
                    "video_field_1": TestVideoUrls.VIDEO2.value, # 200KB
                    "text_field_1": "This video should work"
                },
                {
                    "_id": "2", 
                    "video_field_1": TestVideoUrls.VIDEO1.value, # 2.5MB
                    "text_field_1": "This video should fail"
                }
            ]

            for index in [self.structured_language_bind_index_name, self.unstructured_language_bind_index_name]:
                with self.subTest(f"Testing video size limit for index {index}"):
                    tensor_fields = ["video_field_1", "text_field_1"] if "unstructured" in index else None
                    
                    # Add documents
                    result = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index,
                            docs=test_docs,
                            tensor_fields=tensor_fields
                        )
                    ).dict(exclude_none=True, by_alias=True)

                    # Verify results
                    self.assertTrue(result["errors"])  # Should have errors due to second document
                    self.assertEqual(2, len(result["items"]))
                    
                    # First document should succeed
                    self.assertEqual(200, result["items"][0]["status"])
                    self.assertNotIn("error", result["items"][0])
                    
                    # Second document should fail with size limit error
                    self.assertEqual(400, result["items"][1]["status"])
                    self.assertIn("exceeds the maximum allowed size", result["items"][1]["error"])

                    # Verify the first document was actually added
                    get_result = tensor_search.get_documents_by_ids(
                        config=self.config,
                        index_name=index,
                        document_ids=["1"]
                    ).dict(exclude_none=True, by_alias=True)
                    
                    self.assertEqual(1, len(get_result["results"]))
                    self.assertEqual("1", get_result["results"][0]["_id"])

    def test_supported_audio_format(self):
        """Test the supported audio format for the LanguageBind model in add_documents and search."""

        test_cases = [
            (TestAudioUrls.MP3_AUDIO1.value, "mp3"),
            (TestAudioUrls.ACC_AUDIO1.value, "aac"),
            (TestAudioUrls.OGG_AUDIO1.value, "ogg"),
            (TestAudioUrls.FLAC_AUDIO1.value, "flac")
        ]

        for test_case, audio_format in test_cases:
            for index in [self.structured_language_bind_index_name, self.unstructured_language_bind_index_name]:
                with self.subTest(f"{index} - {audio_format}"):
                    self.clear_index_by_schema_name(
                        schema_name=self.index_management.get_index(index_name=index).schema_name)
                    self.assertEqual(0, self.monitoring.get_index_stats_by_name(index_name=index).number_of_documents)
                    document = {
                        "audio_field_1": test_case,
                        "_id": "1"
                    }

                    res = self.add_documents(
                        self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index,
                            docs=[document],
                            tensor_fields=[
                                "audio_field_1"] if index == self.unstructured_language_bind_index_name else None
                        )
                    )
                    self.assertFalse(res.errors, msg=res.dict())
                    self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name=index).number_of_documents)
                    self.assertGreaterEqual(self.monitoring.get_index_stats_by_name(index_name=index).number_of_vectors,
                                            1)
                    if test_case not in [TestAudioUrls.ACC_AUDIO1.value,]:
                    # .acc is not support
                        _ = tensor_search.search(
                            config=self.config,
                            index_name=index,
                            text=test_case,
                            search_method = "TENSOR"
                        )

    def test_supported_video_format(self):
        """Test the supported video format for the LanguageBind model in add_documents and search."""

        test_cases = [
            (TestVideoUrls.AVI_VIDEO1.value, "avi"),
            (TestVideoUrls.MKV_VIDEO1.value, "mkv"),
            (TestVideoUrls.WEBM_VIDEO1.value, "webm")
        ]

        for test_case, audio_format in test_cases:
            for index in [self.structured_language_bind_index_name, self.unstructured_language_bind_index_name]:
                with self.subTest(f"{index} - {audio_format}"):
                    self.clear_index_by_schema_name(
                        schema_name=self.index_management.get_index(index_name=index).schema_name)
                    self.assertEqual(0, self.monitoring.get_index_stats_by_name(index_name=index).number_of_documents)
                    document = {
                        "video_field_1": test_case,
                        "_id": "1"
                    }

                    res = self.add_documents(
                        self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index,
                            docs=[document],
                            tensor_fields=[
                                "video_field_1"] if index == self.unstructured_language_bind_index_name else None
                        )
                    )
                    self.assertFalse(res.errors, msg=res.dict())
                    self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name=index).number_of_documents)
                    self.assertGreaterEqual(self.monitoring.get_index_stats_by_name(index_name=index).number_of_vectors,
                                            1)

                    _ = tensor_search.search(
                        config=self.config,
                        index_name=index,
                        text=test_case,
                        search_method = "TENSOR"
                    )

    def test_custom_languagebind_model(self):
        """Test the custom languagebind model in add_documents and search end-to-end."""
        docs = [
            {
                "_id": "1",
                "text_field_1": "This is a test text",
                "image_field_1": TestImageUrls.IMAGE1.value,
                "audio_field_1": TestAudioUrls.AUDIO1.value,
                "video_field_1": TestVideoUrls.VIDEO1.value
            }
        ]
        with self.subTest("custom-languagebind-model-add-documents"):
            res = self.add_documents(
                self.config,
                add_docs_params=AddDocsParams(
                    index_name=self.unstructured_custom_language_bind_index_name,
                    docs=docs,
                    tensor_fields=["text_field_1", "image_field_1", "audio_field_1", "video_field_1"]
                )
            )

            self.assertEqual(False, res.errors)
            self.assertEqual(
                1,
                self.monitoring.get_index_stats_by_name(
                    index_name=self.unstructured_custom_language_bind_index_name).number_of_documents)
            self.assertGreaterEqual(
                4,
                self.monitoring.get_index_stats_by_name(
                    index_name=self.unstructured_custom_language_bind_index_name).number_of_vectors
            )

        search_test_cases = [
            ("This is a test text", "text"),
            (TestImageUrls.IMAGE1.value, "image"),
            (TestAudioUrls.AUDIO1.value, "audio"),
            (TestVideoUrls.VIDEO1.value, "video")
        ]

        for query, modality in search_test_cases:
            with self.subTest(f"custom-languagebind-model-search-{modality}"):
                _ = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_custom_language_bind_index_name,
                    text=query,
                    search_method = "TENSOR"
                )