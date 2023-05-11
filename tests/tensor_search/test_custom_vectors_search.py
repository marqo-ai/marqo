from tests.utils.transition import add_docs_caller
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from unittest.mock import patch
import numpy as np


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
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            }], auto_refresh=True)

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except:
            pass

    def test_search(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        {"tensor": [{"vector": [1, ] * 512, "weight": 2}, {"vector": [2, ] * 512, "weight": -1}], })

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
        res_3 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        {"tensor": [{"vector": [1, ] * 512, "weight": -1}, {"vector": [1, ] * 512, "weight": 1}], })

        assert res_1["hits"][0]["_score"] == res_2["hits"][0]["_score"]
        assert res_1["hits"][0]["_score"] == res_3["hits"][0]["_score"]

    def test_search_vectors(self):
        with patch("numpy.mean", wraps = np.mean) as mock_mean:
            query = {
                "A rider is riding a horse jumping over the barrier": 1,
            }
            res_1 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query)

            weight_1, weight_2, weight_3 = 2.5, 3.4, -1.334
            vector_2 = [-1,] * 512
            vector_3 = [1.3,] * 512
            query = {
                "A rider is riding a horse jumping over the barrier": weight_1,
            }

            res_2 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
            {"tensor": [{"vector": vector_2, "weight": weight_2}, {"vector": vector_3, "weight": weight_3}], })

            args_list = [args[0] for args in mock_mean.call_args_list]
            vectorised_string = args_list[0][0][0]
            weighted_vectors = args_list[1][0]

            assert np.allclose(vectorised_string * weight_1, weighted_vectors[0], atol=1e-9)
            assert np.allclose(np.array(vector_2) * weight_2, weighted_vectors[1], atol=1e-9)
            assert np.allclose(np.array(vector_3) * weight_3, weighted_vectors[2], atol=1e-9)