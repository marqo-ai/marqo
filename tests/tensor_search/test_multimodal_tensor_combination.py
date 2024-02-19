import json
import os
from unittest import mock

import numpy as np

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.s2_inference.s2_inference import vectorise
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.tensor_search import vectorise_multimodal_combination_field_unstructured, vectorise_multimodal_combination_field_structured
from tests.marqo_test import MarqoTestCase


class TestMultimodalTensorCombination(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Unstructured index requests
        unstructured_random_multimodal_index = cls.unstructured_marqo_index_request(
            model=Model(name="random/small"),
            treat_urls_and_pointers_as_images=True,
        )

        unstructured_random_text_index = cls.unstructured_marqo_index_request(
            model=Model(name="random/small"),
            treat_urls_and_pointers_as_images=False,
        )

        unstructured_multimodal_index = cls.unstructured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion400m_e31"),
            treat_urls_and_pointers_as_images=True,
        )

        unstructured_unnormalized_multimodal_index = cls.unstructured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion400m_e31"),
            treat_urls_and_pointers_as_images=True,
            normalize_embeddings=False
        )

        # Structured index requests
        structured_random_multimodal_index = cls.structured_marqo_index_request(
            model=Model(name="random/small"),
            fields=[
                FieldRequest(name="Title", type=FieldType.Text),

                # For batch vectorise tests
                FieldRequest(name="text_field", type=FieldType.Text),
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="text_field_2", type=FieldType.Text),
                FieldRequest(name="text_field_3", type=FieldType.Text),
                FieldRequest(name="text_field_4", type=FieldType.Text),
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_3", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_4", type=FieldType.ImagePointer),
                FieldRequest(name="combo_text_image", type=FieldType.MultimodalCombination,
                      dependent_fields={"text_field": 0.5, "image_field": 0.8}),
                FieldRequest(name="multi_combo_text_image", type=FieldType.MultimodalCombination,
                     dependent_fields={
                         "text_field_1": 0.1, "text_field_2": 0.1, "text_field_3": 0.1, "text_field_4": 0.1,
                         "image_field_1": 0.1, "image_field_2": 0.1, "image_field_3": 0.1, "image_field_4": 0.1,
                     }),

                # For multiple multimodal field tests
                FieldRequest(name="my_multimodal_field_0", type=FieldType.MultimodalCombination,
                        dependent_fields={"text_field": 0.5, "image_field": 0.8}),
                FieldRequest(name="my_multimodal_field_1", type=FieldType.MultimodalCombination,
                        dependent_fields={"text_field": 0.5, "image_field": -1}),
                FieldRequest(name="my_multimodal_field_2", type=FieldType.MultimodalCombination,
                        dependent_fields={"Title": 0.5, "image_field": 0}),

                # For bad field tests
                FieldRequest(name="bad_test_text_field", type=FieldType.Text),
                FieldRequest(name="bad_field_float", type=FieldType.Float),
                FieldRequest(name="bad_field_int", type=FieldType.Int),
                FieldRequest(name="bad_field_bool", type=FieldType.Bool),
                FieldRequest(name="bad_field_list", type=FieldType.ArrayText),
                FieldRequest(name="bad_field_img", type=FieldType.ImagePointer),
                FieldRequest(name="bad_multimodal_field", type=FieldType.MultimodalCombination,
                        dependent_fields={"bad_test_text_field": 0.5, "bad_field_float": 0.8, "bad_field_int": 0.8,
                                          "bad_field_bool": 0.8, "bad_field_list": 0.8, "bad_field_img": 0.8})
            ],
            tensor_fields=["combo_text_image", "multi_combo_text_image", "bad_multimodal_field",
                           "my_multimodal_field_0", "my_multimodal_field_1", "my_multimodal_field_2"]
        )

        # Simulates an index where image URLs are still treated as text.
        structured_random_text_index = cls.structured_marqo_index_request(
            model=Model(name="random/small"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="text_field_2", type=FieldType.Text),
                FieldRequest(name="text_field_3", type=FieldType.Text),
                FieldRequest(name="text_field_4", type=FieldType.Text),
                FieldRequest(name="image_field_1", type=FieldType.Text),
                FieldRequest(name="image_field_2", type=FieldType.Text),
                FieldRequest(name="image_field_3", type=FieldType.Text),
                FieldRequest(name="image_field_4", type=FieldType.Text),
                FieldRequest(name="multi_combo_text_image", type=FieldType.MultimodalCombination,
                     dependent_fields={
                         "text_field_1": 0.1, "text_field_2": 0.1, "text_field_3": 0.1, "text_field_4": 0.1,
                         "image_field_1": 0.1, "image_field_2": 0.1, "image_field_3": 0.1, "image_field_4": 0.1,
                     }),
            ],
            tensor_fields=["multi_combo_text_image"]
        )

        structured_multimodal_index = cls.structured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion400m_e31"),
            fields=[
                FieldRequest(name="Title", type=FieldType.Text),
                FieldRequest(name="text_field", type=FieldType.Text),
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="combo_text_image", type=FieldType.MultimodalCombination,
                      dependent_fields={"text_field": 0.5, "image_field": 0.8}),
                FieldRequest(name="zero_weight_text_field", type=FieldType.Text),
                FieldRequest(name="zero_weight_image_field", type=FieldType.ImagePointer),
                FieldRequest(name="zero_weight_combo_text_image", type=FieldType.MultimodalCombination,
                             dependent_fields={"zero_weight_text_field": 1, "zero_weight_image_field": 0})  # Note only image_field has weight 0, just for this example.
            ],
            tensor_fields=["combo_text_image", "zero_weight_combo_text_image"]
        )

        structured_unnormalized_multimodal_index = cls.structured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion400m_e31"),
            normalize_embeddings=False,
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="text_field_2", type=FieldType.Text),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="combo_text_image", type=FieldType.MultimodalCombination,
                      dependent_fields={"text_field_1": 0.32, "text_field_2": 0, "image_field_1": -0.48, "image_field_2": 1.34}),

                # Independent text and image fields for isolating tensors
                FieldRequest(name="text_field_3", type=FieldType.Text),
                FieldRequest(name="text_field_4", type=FieldType.Text),
                FieldRequest(name="image_field_3", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_4", type=FieldType.ImagePointer),

            ],
            tensor_fields=["combo_text_image", "text_field_3", "text_field_4", "image_field_3", "image_field_4"]
        )

        cls.indexes = cls.create_indexes([unstructured_random_multimodal_index,
                                          unstructured_random_text_index,
                                          unstructured_multimodal_index,
                                          unstructured_unnormalized_multimodal_index,

                                          structured_random_multimodal_index,
                                          structured_random_text_index,
                                          structured_multimodal_index,
                                          structured_unnormalized_multimodal_index,
                                          ])

        # Assign indexes to class variables
        cls.unstructured_random_multimodal_index = cls.indexes[0]
        cls.unstructured_random_text_index = cls.indexes[1]
        cls.unstructured_multimodal_index = cls.indexes[2]
        cls.unstructured_unnormalized_multimodal_index = cls.indexes[3]

        cls.structured_random_multimodal_index = cls.indexes[4]
        cls.structured_random_text_index = cls.indexes[5]
        cls.structured_multimodal_index = cls.indexes[6]
        cls.structured_unnormalized_multimodal_index = cls.indexes[7]


    def setUp(self):
        super().setUp()
        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_add_documents_with_one_multimodal_fields(self):

        for index in [self.unstructured_random_multimodal_index, self.structured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                doc = {
                    "Title": "Horse rider",
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "1"
                }

                mappings = {
                    "combo_text_image":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field": 0.5, "image_field": 0.8
                            }
                        }
                }

                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[doc, ],
                    mappings=mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                    device="cpu",
                    tensor_fields=["combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None),
                                            )
                added_doc = tensor_search.get_document_by_id(config=self.config, index_name=index.name,
                                                             document_id="1", show_vectors=True)
                for key, value in doc.items():
                    self.assertIn(key, added_doc)
                    self.assertEqual(value, added_doc[key])

                self.assertIn("_tensor_facets", added_doc)
                self.assertEqual(1, len(added_doc["_tensor_facets"]))
                self.assertIn("_embedding", added_doc["_tensor_facets"][0])
                self.assertIn("combo_text_image", added_doc["_tensor_facets"][0])

    def test_add_documents_with_multiple_multimodal_fields(self):
        for index in [self.unstructured_random_multimodal_index, self.structured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                doc = {
                    "Title": "Horse rider",
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "1"
                }

                mappings = {
                    "my_multimodal_field_0":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field": 0.5, "image_field": 0.8
                            }
                        },
                    "my_multimodal_field_1":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field": 0.5, "image_field": -1
                            }
                        },
                    "my_multimodal_field_2":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "Title": 0.5, "image_field": 0
                            }
                        }
                }

                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[doc, ],
                    mappings=mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                    device="cpu",
                    tensor_fields=["my_multimodal_field_0", "my_multimodal_field_1", "my_multimodal_field_2"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))
                added_doc = tensor_search.get_document_by_id(config=self.config, index_name=index.name,
                                                             document_id="1", show_vectors=True)
                for key, value in doc.items():
                    self.assertIn(key, added_doc)
                    self.assertEqual(value, added_doc[key])

                self.assertIn("_tensor_facets", added_doc)
                
                def get_relevant_tensor_facets(added_doc):
                    """
                    Only get the tensor facets in the fields that start with "my_multimodal_field". This is to avoid
                    counting the tensor facets from other tests in this suite.
                    """
                    relevant_facets = []
                    for facet in added_doc["_tensor_facets"]:
                        for key in facet:
                            if key.startswith("my_multimodal_field_"):
                                relevant_facets.append(facet)
                                continue
                    return relevant_facets

                relevant_facets = get_relevant_tensor_facets(added_doc)
                self.assertEqual(3, len(relevant_facets))

                for i in range(3):
                    self.assertIn("_embedding", relevant_facets[i])

    def test_get_document_by_id_return_multimodal_params_logic(self):
        for index in [self.unstructured_random_multimodal_index, self.structured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                doc = {
                    "Title": "Horse rider",
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "1"
                }

                mappings = {
                    "combo_text_image":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field": 0.5, "image_field": 0.8
                            }
                        }
                }

                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[doc, ],
                    mappings=mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                    device="cpu",
                    tensor_fields=["combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None,
                ))

                test_cases = [
                    (True, True, "show_vectors = True, should return multimodal_params"),
                    (False, False, "show_vectors = False, should not return multimodal_params"),
                ]

                for show_vectors, is_return_multimodal_params, msg in test_cases:
                    with self.subTest(msg):
                        returned_doc = tensor_search.get_document_by_id(config=self.config,
                                                                        index_name=index.name,
                                                                        document_id="1", show_vectors=show_vectors)
                        self.assertEqual(show_vectors, "_tensor_facets" in returned_doc)
                        if isinstance(index, UnstructuredMarqoIndex):
                            self.assertEqual(is_return_multimodal_params, "multimodal_params" in returned_doc)
                        elif isinstance(index, StructuredMarqoIndex):
                            # If index is structured, multimodal_params should never be returned.
                            self.assertNotIn("multimodal_params", returned_doc)

    def test_multimodal_fields_correct_number_of_vectors(self):
        # TODO: make new structured index for this
        doc = [
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                "_id": "1"
            },
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "2"
            },
            {
                "content": "random content",
                "_id": "3"
            }

        ]

        # mappings, tensor_fields, number_of_documents, number_of_vectors
        test_cases = [
            # number_of_vectors from each document: 1, 1, 0
            ({"multimodal_fields_0": {"type": "multimodal_combination", "weights":
                {"text_field": 0.5, "image_field": 0.3}}}, ["multimodal_fields_0"], 3, 2),
            # number_of_vectors from each document: 2, 2, 0
            ({"multimodal_fields_0": {"type": "multimodal_combination", "weights":
                {"text_field": 0.5, "image_field": 0.3}}}, ["multimodal_fields_0", "Title"], 3, 4),
            # number_of_vectors from each document: 1, 1, 0
            ({"multimodal_fields_0": {"type": "multimodal_combination", "weights":
                {"text_field": 0.5, "image_field": 0.3}}}, ["multimodal_fields_0", "content"], 3, 3),
            # number_of_vectors from each document: 2, 1, 1
            ({"multimodal_fields_0": {"type": "multimodal_combination", "weights":
                {"text_field": 0.5, "image_field": 0.3}},
              "multimodal_fields_1": {"type": "multimodal_combination", "weights":
                  {"content": 0.5, "image_field": 0.3}}}, [
                 "multimodal_fields_0", "multimodal_fields_1"], 3, 4),
            # number_of_vectors from each document: 4, 3, 1
            ({"multimodal_fields_0": {"type": "multimodal_combination", "weights":
                {"text_field": 0.5, "image_field": 0.3}}},
             ["multimodal_fields_0", "Title", "text_field", "image_field", "content"], 3, 8),
        ]

        for mappings, tensor_fields, number_of_documents, number_of_vectors in test_cases:
            with self.subTest(f"{mappings}, {tensor_fields}"):
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.unstructured_random_multimodal_index_name, docs=doc,
                    mappings=mappings,
                    device="cpu",
                    tensor_fields=tensor_fields),
                )

                res = self.monitoring.get_index_stats_by_name(index_name=self.unstructured_random_multimodal_index_name)
                self.assertEqual(number_of_documents, res.number_of_documents)
                self.assertEqual(number_of_vectors, res.number_of_vectors)

                self.clear_indexes(self.indexes)


    def test_multimodal_field_bad_field_content(self):
        # TODO: Add structured index: `structured_random_multimodal_index` to this list once validation is added
        for index in [self.unstructured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                test_cases = [
                    ({"bad_test_text_field": "test", "bad_field_float": 2.4}, "received bad_field_float:2.4"),
                    ({"bad_test_text_field": "test", "bad_field_int": 1}, "received bad_field_int:1"),
                    ({"bad_test_text_field": "test", "bad_field_bool": True}, "received bad_field_bool:True"),
                    ({"bad_test_text_field": "test", "bad_field_list": ["123", "23"]}, f'received bad_field_list:{["123", "23"]}'),
                    ({"bad_test_text_field": "test", "bad_field_img": "https://a-void-image.jpg"}, "Could not find image"),
                    ({"bad_multimodal_field": "test"}, "Document and mappings object have conflicting fields")
                ]

                mappings = {
                    "bad_multimodal_field":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "bad_test_text_field": 0.5, "bad_field_float": 0.8, "bad_field_int": 0.8,
                                "bad_field_bool": 0.8, "bad_field_list": 0.8, "bad_field_img": 0.8
                            }
                        }
                }

                for document, error_msg in test_cases:
                    with self.subTest(error_msg):
                        with mock.patch("marqo.s2_inference.s2_inference.vectorise") as mock_vectorise:
                            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                                index_name=index.name, docs=[document, ],
                                mappings=mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                                device="cpu",
                                tensor_fields=["bad_multimodal_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                            ))
                            self.assertIn(error_msg, str(res))
                            self.assertEqual(0, self.monitoring.get_index_stats_by_name(
                                index.name).number_of_documents)
                            self.assertEqual(0, self.monitoring.get_index_stats_by_name(
                                index.name).number_of_vectors)
                            mock_vectorise.assert_not_called()


    def test_multimodal_tensor_combination_score(self):
        for index in [self.unstructured_multimodal_index, self.structured_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                def get_score(document):
                    self.clear_indexes(self.indexes)
                    res = tensor_search.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=index.name, docs=[document],
                            mappings={"combo_text_image": {"type": "multimodal_combination",
                                                           "weights": {"image_field": 0.5, "text_field": 0.8}}}
                            if isinstance(index, UnstructuredMarqoIndex) else None,
                            device="cpu",
                            tensor_fields=["combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )
                    self.assertEqual(1, self.monitoring.get_index_stats_by_name(index.name).number_of_documents)
                    res = tensor_search.search(config=self.config, index_name=index.name, text="",
                                               result_count=1)
                    return res["hits"][0]["_score"]

                score_1 = get_score({
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    # "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                })

                score_2 = get_score({
                    # "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                })

                score_3 = get_score({
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                })

                assert (score_3 >= min(score_1, score_2)) and (score_3 <= max(score_1, score_2))


    def test_multimodal_tensor_combination_tensor_value(self):
        for index in [self.unstructured_unnormalized_multimodal_index, self.structured_unnormalized_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[
                        {
                            "text_field_1": "A rider is riding a horse jumping over the barrier.",
                            "text_field_2": "What is the best to wear on the moon?",
                            "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                            "_id": "c1"
                        },
                        {
                            "text_field_1": "A rider is riding a horse jumping over the barrier.",
                            "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "text_field_2": "What is the best to wear on the moon?",
                            "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                            "_id": "c2"
                        },
                        {
                            "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                            "text_field_1": "A rider is riding a horse jumping over the barrier.",
                            "text_field_2": "What is the best to wear on the moon?",
                            "_id": "c3"
                        },
                        {
                            "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "text_field_1": "A rider is riding a horse jumping over the barrier.",
                            "text_field_2": "What is the best to wear on the moon?",
                            "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                            "_id": "c4"
                        }],
                    tensor_fields=["combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    device="cpu",
                    mappings={
                        "combo_text_image": {
                            "type": "multimodal_combination",
                            "weights": {"text_field_1": 0.32, "text_field_2": 0, "image_field_1": -0.48,
                                        "image_field_2": 1.34}}
                    } if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[
                        {
                            "text_field_3": "A rider is riding a horse jumping over the barrier.",
                            "_id": "1"
                        },
                        {
                            "text_field_4": "What is the best to wear on the moon?",
                            "_id": "2"
                        },
                        {
                            "image_field_3": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "_id": "3"
                        },
                        {
                            "image_field_4": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                            "_id": "4"
                        }],
                    tensor_fields=["text_field_3", "text_field_4", "image_field_3", "image_field_4"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    device="cpu",
                ))

                def get_specific_field_facet(document_id, field):
                    doc_facets = tensor_search.get_document_by_id(config=self.config,
                                                            index_name=index.name,
                                                            document_id=document_id,
                                                            show_vectors=True)['_tensor_facets']
                    # Identify the first facet that corresponds to field we're looking for
                    for facet in doc_facets:
                        if field in facet:
                            return np.array(facet['_embedding'])
                    return None

                combo_tensor_1 = get_specific_field_facet("c1", "combo_text_image")
                combo_tensor_2 = get_specific_field_facet("c2", "combo_text_image")
                combo_tensor_3 = get_specific_field_facet("c3", "combo_text_image")
                combo_tensor_4 = get_specific_field_facet("c4", "combo_text_image")

                text_tensor_1 = \
                    np.array(tensor_search.get_document_by_id(config=self.config,
                                                              index_name=index.name,
                                                              document_id="1",
                                                              show_vectors=True)['_tensor_facets'][0]["_embedding"])
                text_tensor_2 = \
                    np.array(tensor_search.get_document_by_id(config=self.config,
                                                              index_name=index.name,
                                                              document_id="2",
                                                              show_vectors=True)['_tensor_facets'][0]["_embedding"])
                image_tensor_1 = \
                    np.array(tensor_search.get_document_by_id(config=self.config,
                                                              index_name=index.name,
                                                              document_id="3",
                                                              show_vectors=True)['_tensor_facets'][0]["_embedding"])
                image_tensor_2 = \
                    np.array(tensor_search.get_document_by_id(config=self.config,
                                                              index_name=index.name,
                                                              document_id="4",
                                                              show_vectors=True)['_tensor_facets'][0]["_embedding"])

                expected_tensor = np.mean(
                    [text_tensor_1 * 0.32, text_tensor_2 * 0, image_tensor_1 * -0.48, image_tensor_2 * 1.34], axis=0)
                assert np.allclose(combo_tensor_1, expected_tensor, atol=1e-5)
                assert np.allclose(combo_tensor_2, expected_tensor, atol=1e-5)
                assert np.allclose(combo_tensor_3, expected_tensor, atol=1e-5)
                assert np.allclose(combo_tensor_4, expected_tensor, atol=1e-5)

    def test_multimodal_tensor_combination_zero_weight(self):
        for index in [self.unstructured_multimodal_index, self.structured_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                def get_score(document):
                    self.clear_indexes(self.indexes)
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=[document], device="cpu", mappings={
                            "zero_weight_combo_text_image": {
                                "type": "multimodal_combination",
                                "weights": {"zero_weight_image_field": 0, "zero_weight_text_field": 1}}} if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["zero_weight_combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None
                    ))
                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text="test", result_count=1)

                    return res["hits"][0]["_score"]

                score_1 = get_score({
                    "zero_weight_text_field": "A rider is riding a horse jumping over the barrier.",
                    "_id": "1"
                })

                score_2 = get_score({
                    "zero_weight_text_field": "A rider is riding a horse jumping over the barrier.",
                    "zero_weight_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "1"
                })

                self.assertEqual(score_1, score_2)

    def test_multimodal_tensor_combination_vectorise_call_unstructured(self):
        """check if the chunks are properly created in the add_documents"""

        def pass_through_multimodal(*args, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the args and kwargs"""
            return vectorise_multimodal_combination_field_unstructured(*args, **kwargs)

        mock_multimodal_combination = mock.MagicMock()
        mock_multimodal_combination.side_effect = pass_through_multimodal

        @mock.patch("marqo.tensor_search.tensor_search.vectorise_multimodal_combination_field_unstructured",
                    mock_multimodal_combination)
        def run():
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.unstructured_random_multimodal_index.name, docs=[
                    {
                        "text_field": "A rider is riding a horse jumping over the barrier.",
                        "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                        "_id": "123",
                    },
                    {
                        "text_field": "test-text-two.",
                        "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                        "_id": "234",
                    },
                    {  # a normal doc
                        "combo_text_image_test": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                        "_id": "534",
                    }],
                mappings={
                    "combo_text_image": {
                        "type": "multimodal_combination",
                        "weights": {"image_field": 0.5, "text_field": 0.5}}}, device="cpu",
                tensor_fields=["combo_text_image"]
            )
                                        )

            # first multimodal-doc
            real_field_0, field_content_0 = [call_args for call_args, call_kwargs
                                             in mock_multimodal_combination.call_args_list][0][0:2]
            assert real_field_0 == "combo_text_image"
            assert field_content_0 == {
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
            }

            # second multimodal=doc
            real_field_1, field_content_1 = [call_args for call_args, call_kwargs
                                             in mock_multimodal_combination.call_args_list][1][0:2]
            assert real_field_1 == "combo_text_image"
            assert field_content_1 == {
                "text_field": "test-text-two.",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            }
            # ensure we only call multimodal-combination twice
            assert len(mock_multimodal_combination.call_args_list) == 2
            return True

        assert run()


    def test_multimodal_tensor_combination_vectorise_call_structured(self):
        """
        check if the chunks are properly created in the add_documents
        Completely separate from the unstructured test, as to not make the mocking logic too complex
        """

        def pass_through_multimodal(*args, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the args and kwargs"""
            return vectorise_multimodal_combination_field_structured(*args, **kwargs)

        mock_multimodal_combination = mock.MagicMock()
        mock_multimodal_combination.side_effect = pass_through_multimodal

        @mock.patch("marqo.tensor_search.tensor_search.vectorise_multimodal_combination_field_structured",
                    mock_multimodal_combination)
        def run():
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.structured_random_multimodal_index.name, docs=[
                    {
                        "text_field": "A rider is riding a horse jumping over the barrier.",
                        "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                        "_id": "123",
                    },
                    {
                        "text_field": "test-text-two.",
                        "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                        "_id": "234",
                    },
                    {  # a normal doc
                        "combo_text_image": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                        "_id": "534",
                    }],
            )
                                        )

            # first multimodal-doc
            real_field_0, field_content_0 = [call_args for call_args, call_kwargs
                                             in mock_multimodal_combination.call_args_list][0][0:2]
            assert real_field_0 == "combo_text_image"
            assert field_content_0 == {
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
            }

            # second multimodal=doc
            real_field_1, field_content_1 = [call_args for call_args, call_kwargs
                                             in mock_multimodal_combination.call_args_list][1][0:2]
            assert real_field_1 == "combo_text_image"
            assert field_content_1 == {
                "text_field": "test-text-two.",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            }
            # ensure we only call multimodal-combination twice
            assert len(mock_multimodal_combination.call_args_list) == 2
            return True

        assert run()

    def test_batched_vectorise_call(self):
        for index in [self.unstructured_random_multimodal_index, self.structured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                def pass_through_vectorise(*arg, **kwargs):
                    """Vectorise will behave as usual, but we will be able to see the call list
                    via mock
                    """
                    return vectorise(*arg, **kwargs)

                mock_vectorise = mock.MagicMock()
                mock_vectorise.side_effect = pass_through_vectorise

                @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
                def run():
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=[
                            {
                                "text_field_1": "A rider is riding a horse jumping over the barrier_1.",
                                "text_field_2": "A rider is riding a horse jumping over the barrier_2.",
                                "text_field_3": "A rider is riding a horse jumping over the barrier_3.",
                                "text_field_4": "A rider is riding a horse jumping over the barrier_4.",
                                "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                                "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                                "image_field_3": "https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg",
                                "image_field_4": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                                "_id": "111",
                            }],
                        mappings={"multi_combo_text_image": {"type": "multimodal_combination", "weights": {
                            "text_field_1": 0.1, "text_field_2": 0.1, "text_field_3": 0.1, "text_field_4": 0.1,
                            "image_field_1": 0.1, "image_field_2": 0.1, "image_field_3": 0.1, "image_field_4": 0.1,
                        }}} if isinstance(index, UnstructuredMarqoIndex) else None,
                        device="cpu",
                        tensor_fields=["multi_combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None
                    ))
                    # Ensure the doc is added
                    assert tensor_search.get_document_by_id(config=self.config, index_name=index.name,
                                                            document_id="111")
                    # Ensure that vectorise is only called twice
                    assert len(mock_vectorise.call_args_list) == 2

                    text_content = [f"A rider is riding a horse jumping over the barrier_{i}." for i in range(1, 5)]

                    real_text_content = [call_kwargs['content'] for call_args, call_kwargs
                                         in mock_vectorise.call_args_list][0]

                    # Ensure the text vectorise is expected
                    self.assertEqual(real_text_content, text_content)
                    return True

                assert run()

    def test_batched_vectorise_call_infer_image_is_false(self):
        """
        Test to ensure that it's possible for image URLs to be treated as plain text. This is done differently depending
        on the index type.
        If structured: Each image field is declared as text in the initial index creation.
        If unstructured: treat_urls_and_pointers_as_images for the index is simply set to False.

        Check setUpClass for the index creation
        """
        for index in [self.unstructured_random_text_index, self.structured_random_text_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                def pass_through_vectorise(*arg, **kwargs):
                    """Vectorise will behave as usual, but we will be able to see the call list
                    via mock
                    """
                    return vectorise(*arg, **kwargs)

                mock_vectorise = mock.MagicMock()
                mock_vectorise.side_effect = pass_through_vectorise

                @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
                def run():
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=[
                            {
                                "text_field_1": "A rider is riding a horse jumping over the barrier_1.",
                                "text_field_2": "A rider is riding a horse jumping over the barrier_2.",
                                "text_field_3": "A rider is riding a horse jumping over the barrier_3.",
                                "text_field_4": "A rider is riding a horse jumping over the barrier_4.",
                                "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                                "image_field_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                                "image_field_3": "https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg",
                                "image_field_4": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                                "_id": "111",
                            }],
                        mappings={
                            "multi_combo_text_image": {"type": "multimodal_combination", "weights": {
                                "text_field_1": 0.1, "text_field_2": 0.1, "text_field_3": 0.1, "text_field_4": 0.1,
                                "image_field_1": 0.1, "image_field_2": 0.1, "image_field_3": 0.1, "image_field_4": 0.1,
                            }}} if isinstance(index, UnstructuredMarqoIndex) else None,
                        device="cpu",
                        tensor_fields=["multi_combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None
                    ))
                    # Ensure the doc is added
                    assert tensor_search.get_document_by_id(config=self.config, index_name=index.name,
                                                            document_id="111")
                    # Ensure that vectorise is only called twice
                    self.assertEqual(1, len(mock_vectorise.call_args_list))

                    text_content = [f"A rider is riding a horse jumping over the barrier_{i}." for i in range(1, 5)]
                    text_content = text_content + [
                        f"https://marqo-assets.s3.amazonaws.com/tests/images/image{i}.jpg"
                        for i in range(1, 5)]

                    real_text_content = [call_kwargs['content'] for call_args, call_kwargs
                                         in mock_vectorise.call_args_list][0]

                    # Ensure the text vectorise is expected
                    self.assertEqual(real_text_content, text_content)
                    return True

                assert run()

    def test_concurrent_image_downloading(self):
        # TODO: Make structured index
        def pass_through_load_image_from_path(*arg, **kwargs):
            return load_image_from_path(*arg, **kwargs)

        mock_load_image_from_path = mock.MagicMock()
        mock_load_image_from_path.side_effect = pass_through_load_image_from_path

        @mock.patch("marqo.s2_inference.clip_utils.load_image_from_path", mock_load_image_from_path)
        def run():
            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.unstructured_random_multimodal_index_name, docs=[
                    {
                        "text_0": "A rider is riding a horse jumping over the barrier_0.",
                        "text_1": "A rider is riding a horse jumping over the barrier_1.",
                        "text_2": "A rider is riding a horse jumping over the barrier_2.",
                        "text_3": "A rider is riding a horse jumping over the barrier_3.",
                        "text_4": "A rider is riding a horse jumping over the barrier_4.",
                        "image_0": "https://marqo-assets.s3.amazonaws.com/tests/images/image0.jpg",
                        "image_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                        "image_2": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                        "image_3": "https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg",
                        "image_4": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                        "_id": "111",
                    }],
                mappings={
                    "combo_text_image": {"type": "multimodal_combination", "weights": {
                        "text_0": 0.1, "text_1": 0.1, "text_2": 0.1, "text_3": 0.1, "text_4": 0.1,
                        "image_0": 0.1, "image_1": 0.1, "image_2": 0.1, "image_3": 0.1, "image_4": 0.1,
                    }}}, device="cpu", tensor_fields=["combo_text_image"]))
            assert tensor_search.get_document_by_id(config=self.config, index_name=self.unstructured_random_multimodal_index_name,
                                                    document_id="111")
            # Ensure that vectorise is only called twice
            self.assertEqual(5, len(mock_load_image_from_path.call_args_list))
            return True

        assert run()

    def test_lexical_search_on_multimodal_combination(self):
        # TODO: Make structured index
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.unstructured_multimodal_index_name, docs=[
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection",
                    "_id": "article_591",
                    "Genre": "Science",
                    "my_image": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                    "some_text": "hello there",
                    "lexical_field": "search me please"
                }
            ],
            mappings={
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "my_image": 0.5,
                        "some_text": 0.5,
                        "lexical_field": 0.1,
                        "additional_field": 0.2,
                    }
                }}, device="cpu", tensor_fields=["my_combination_field"]
        ))

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.unstructured_multimodal_index_name, docs=[
                {
                    "Title": "text",
                    "Description": "text_2",
                    "_id": "article_592",
                    "Genre": "text",
                    "my_image_1": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                    "some_text_1": "hello there",
                    "lexical_field_1": "no no no",
                    "additional_field_1": "test_search here"}
            ],
            mappings={
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "my_image_1": 0.5,
                        "some_text_1": 0.5,
                        "lexical_field_1": 0.1,
                        "additional_field_1": 0.2,
                    }
                }}, device="cpu", tensor_fields=["my_combination_field"])
                                    )
        res = tensor_search.search(config=self.config, index_name=self.unstructured_multimodal_index_name,
                                   text="search me please", search_method="LEXICAL")
        assert res["hits"][0]["_id"] == "article_591"

        res = tensor_search.search(config=self.config, index_name=self.unstructured_multimodal_index_name,
                                   text="test_search here", search_method="LEXICAL")
        assert res["hits"][0]["_id"] == "article_592"

    def test_search_with_filtering_and_infer_image_false(self):
        # TODO: Make structured index
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.unstructured_random_multimodal_index_name, docs=[
                    {
                        "Title": "Extravehicular Mobility Unit (EMU)",
                        "_id": "0",
                        "my_image": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                        "some_text": "hello there",
                        "filter_field": "test_this_0"
                    },
                    {
                        "Title": "Extravehicular Mobility Unit (EMU)",
                        "_id": "1",
                        "my_image": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                        "some_text": "hello there",
                        "filter_field": "test_this_1",
                    },
                    {
                        "Title": "Extravehicular Mobility Unit (EMU)",
                        "_id": "2",
                        "my_image": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                        "some_text": "hello there",
                        "filter_field": "test_this_2",
                    }
                ],
                mappings={
                    "my_combination_field": {
                        "type": "multimodal_combination",
                        "weights": {
                            "my_image": 0.5,
                            "some_text": 0.5,
                            "filter_field": 0,
                        }
                    }}, device="cpu", tensor_fields=["my_combination_field"]
            ))
        res_exist_0 = tensor_search.search(index_name=self.unstructured_random_multimodal_index_name, config=self.config,
                                           text="", filter="filter_field:test_this_0")

        assert res_exist_0["hits"][0]["_id"] == "0"

        res_exist_2 = tensor_search.search(index_name=self.unstructured_random_multimodal_index_name, config=self.config,
                                           text="", filter="filter_field:test_this_2")

        assert res_exist_2["hits"][0]["_id"] == "2"

        res_nonexist_1 = tensor_search.search(index_name=self.unstructured_random_multimodal_index_name, config=self.config,
                                              text="", filter="filter_field:test_this_5")

        assert res_nonexist_1["hits"] == []

    def test_multimodal_combination_chunks(self):
        for index in [self.unstructured_random_multimodal_index, self.structured_random_multimodal_index]:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                test_doc = {
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
                    "text_field": "marqo is good",
                    "_id": "123",
                }

                res = tensor_search.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        docs=[test_doc], index_name=index.name, device="cpu",
                        mappings={"combo_text_image": {"type": "multimodal_combination", "weights": {
                            "text_field": 0.5, "image_field": 0.5
                        }}} if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["combo_text_image"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                doc_w_facets = tensor_search.get_document_by_id(
                    self.config, index_name=index.name, document_id='123', show_vectors=True)
                # check tensor facets:
                self.assertEqual(1, len(doc_w_facets[TensorField.tensor_facets]))
                self.assertIn('combo_text_image', doc_w_facets[TensorField.tensor_facets][0])

                self.assertEqual(doc_w_facets[TensorField.tensor_facets][0]['combo_text_image'],
                                 json.dumps({"text_field": "marqo is good",
                                             "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg"}
                                            )
                                 )

                self.assertNotIn("combo_text_image", doc_w_facets)
