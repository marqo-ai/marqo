import numpy as np

from marqo.errors import IndexNotFoundError
from marqo.s2_inference.errors import InvalidModelPropertiesError, UnknownModelError, ModelLoadError
from marqo.tensor_search import tensor_search

from marqo.s2_inference.s2_inference import (
    available_models,
    vectorise,
    _validate_model_properties,
    _update_available_models
)

from tests.marqo_test import MarqoTestCase


class TestBoostFieldScores(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass
        finally:
            tensor_search.create_vector_index(
                index_name=self.index_name_1, config=self.config)

            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing Polo's travels",
                    "_id": "article_590"
                }
                ,
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection, "
                                   "mobility, life support, and communications for astronauts",
                    "_id": "article_591"
                }
            ], auto_refresh=True)

    def tearDown(self) -> None:
        pass

    def test_score_is_boosted(self):
        q = "What is the best outfit to wear on the moon?"

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
        )
        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={'Title': (5, 1)}
        )

        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']

        self.assertGreater(score_boosted, score)

    def test_boost_empty_dict(self):
        q = "What is the best outfit to wear on the moon?"

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q
        )
        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={}
        )

        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']

        self.assertEqual(score_boosted, score)

    def test_different_attributes_searched_and_boosted(self):
        q = "What is the best outfit to wear on the moon?"

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
            searchable_attributes=['Description'],
        )
        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
            searchable_attributes=['Description'], boost={'Title': (5, 1)}
        )

        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']

        self.assertEqual(score_boosted, score)
