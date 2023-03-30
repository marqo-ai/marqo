import time
import unittest.mock
import pprint

import torch

import marqo.tensor_search.backend
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents, vectorise_multimodal_combination_field
from marqo.errors import DocumentNotFoundError
import numpy as np
from marqo.tensor_search.validation import validate_dict
from marqo.s2_inference.s2_inference import vectorise
import requests
from marqo.s2_inference.clip_utils import load_image_from_path
import json
from unittest.mock import patch
from marqo.errors import MarqoWebError
from pprint import pprint
from marqo._httprequests import HttpRequests

class TestMultimodalTensorCombination(MarqoTestCase):

    def setUp(self):
        self.index_name = "test-index"

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })
        pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

    def test_search_result_not_affected_if_fields_not_exist(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        tensor_search.add_documents(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)


        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                             text = "what is the rider doing?", score_modifiers=None, result_count=10)

        normal_score = normal_res["hits"][0]["_score"]

        modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text = "what is the rider doing?",
                                                score_modifiers={
                                                    "multiply_score_by":
                                                        [{"field_name": "multiply_1",
                                                          "weight": 1,},
                                                         {"field_name": "multiply_2",}
                                                        ],
                                                    "add_to_score": [
                                                        {"field_name": "add_1",
                                                         },
                                                        {"field_name": "add_2",
                                                         }
                                                    ]
                                                })

        modifier_score = modifier_res["hits"][0]["_score"]
        self.assertEqual(normal_score, modifier_score)


    def get_expected_score(self, doc, ori_score, score_modifiers):
        add = 0.0
        for config in score_modifiers.get("multiply_score_by"):
            field_name = config["field_name"]
            weight = config.get("weight", 1)
            if field_name in doc:
                if isinstance(doc[field_name], (int, float)):
                    ori_score = ori_score * weight * doc[field_name]

        for config in score_modifiers.get("add_to_score"):
            field_name = config["field_name"]
            weight = config.get("weight", 1)
            if field_name in doc:
                if isinstance(doc[field_name], (int, float)):
                    add = add + weight * doc[field_name]
        return max(0.0, (ori_score + add))


    def test_search_score_modified_as_expected(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # 4 fields
             "multiply_1": 1,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # let's multiply by 0
             "multiply_1": 0,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": 1,
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # float
             "multiply_1": 1,
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        tensor_search.add_documents(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)

        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = [
            {
                # miss one weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1,},
                     {"field_name": "multiply_2",}],
                "add_to_score": [
                    {"field_name": "add_1", "weight" : -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # 0 weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1.0, },
                     {"field_name": "multiply_2",
                      "weight": 0}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # remove one field
                "multiply_score_by":
                    [
                     {"field_name": "multiply_2",
                      "weight": 1.2}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {  # one part to be none
                "multiply_score_by": [],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
        ]

        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                    text = "what is the rider doing?",
                                                    score_modifiers=score_modifiers, result_count=10)

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_search_score_modified_as_expected_with_skipped_fieds(self):
        # multiply_1 is string here, which should be skipped in modification
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # 4 fields
             "multiply_1": "1",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # let's multiply by 0
             "multiply_1": "0",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": "1",
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # float
             "multiply_1": "1",
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        tensor_search.add_documents(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)

        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = [
            {
                # miss one weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1,},
                     {"field_name": "multiply_2",}],
                "add_to_score": [
                    {"field_name": "add_1", "weight" : -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # 0 weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1.0, },
                     {"field_name": "multiply_2",
                      "weight": 0}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # remove one field
                "multiply_score_by":
                    [
                     {"field_name": "multiply_2",
                      "weight": 1.2}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {  # one part to be none
                "multiply_score_by": [],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
        ]

        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                    text = "what is the rider doing?",
                                                    score_modifiers=score_modifiers, result_count=10)

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_expected_error_raised(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             # 4 fields
             "multiply_1": 1,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
        ]

        invalid_score_modifiers =             {
                # typo in multiply_score_by
                "multiply_scores_by":
                    [{"field_name": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_to_score": [
                    {"field_name": "rate",
                     }],
            },

        tensor_search.add_documents(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        try:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text = "what is the rider doing?",
                                                score_modifiers=invalid_score_modifiers, result_count=10)
            raise AssertionError
        except InvalidArgError:
            pass











