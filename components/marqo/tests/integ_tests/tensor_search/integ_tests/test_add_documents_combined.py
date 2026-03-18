import os
import uuid
from unittest import mock
from unittest.mock import patch

import numpy as np

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from tests.integ_tests.marqo_test import MarqoTestCase, TestImageUrls


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


        semi_structured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True,
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

            unstructured_image_index_request_unnormalized,
            unstructured_text_index_request_unnormalized,
            structured_image_index_request_unnormalized,
            structured_text_index_request_unnormalized
        ])

        cls.structured_marqo_index_name = structured_image_index_request.name
        cls.semi_structured_marqo_index_name = semi_structured_image_index_request.name
        cls.structured_image_index_unnormalized_name = structured_image_index_request_unnormalized.name
        cls.structured_text_index_unnormalized_name = structured_text_index_request_unnormalized.name

        cls.unstructured_marqo_index_name = unstructured_image_index_request.name
        cls.unstructured_image_index_unnormalized_name = unstructured_image_index_request_unnormalized.name
        cls.unstructured_text_index_unnormalized_name = unstructured_text_index_request_unnormalized.name

        cls.image_indexes = cls.indexes[:3]

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