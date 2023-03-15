import unittest.mock
import pprint

import torch

import marqo.tensor_search.backend
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase


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
                    IndexSettingsField.normalize_embeddings: True
                }
            })
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            }], auto_refresh=True)

    def test_search(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        {"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], })

    def test_search_with_incorrect_tensor_dimension(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        try:
            res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
            {"tensor": [{"vector": [1, ] * 3, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], })
            raise AssertionError
        except InvalidArgError as e:
            assert "This causes the error when we do `numpy.mean()` over" in e.message

    def test_search_with_incorrect_query_format(self):
        query = "A rider is riding a horse jumping over the barrier"
        try:
            res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
            {"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], })
            raise AssertionError
        except InvalidArgError as e:
            assert "This is not supported as the context only works when the query is a dictionary." in e.message

    def test_search_score(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }

        res_1 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query)
        res_2 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        {"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], })

        assert res_1["hits"][0]["_score"] == res_2["hits"][0]["_score"]
