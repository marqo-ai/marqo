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
from marqo.errors import MarqoWebError


class TestMultimodalTensorCombination(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.endpoint = self.authorized_url

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
        {
            "Title": "Horse rider",
            "text_field": "A rider is riding a horse jumping over the barrier.",
            "_id": "1"
        }], auto_refresh=True)

    def test_search(self):
        query ={
            "A rider is riding a horse jumping over the barrier" : 1,
        }
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context =
        {"tensor":[{"vector" : [1,] * 512, "weight": 0}, {"vector": [2,] * 512, "weight" : 0}],})
        print(res["hits"][0])



