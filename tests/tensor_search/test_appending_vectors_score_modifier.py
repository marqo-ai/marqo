import copy
import unittest.mock
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from marqo.tensor_search.tensor_search import _create_normal_tensor_search_query
from pydantic.error_wrappers import ValidationError


class TestScoreModifiersSearch(MarqoTestCase):

    def setUp(self):
        self.index_name = "test-index"

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

    def test_index_with_mappings(self):
        documents = [{
            "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg",
            "my_text_fied": "Marqo is good",
            "Reputation": 3.8,
            "Rate": 5.9,
            "Price": 3.0,
            "_id": "1",
        },
        {
            "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
            "my_text_fied": "Marqo is very good",
            "Reputation": 3.12,
            "Rate": 7.0,
            "Price": 2.1,
            "Warranty": 5.0,
            "_id": "2",
        },]

        mappings = {
            "Reputation": {"type": "score_modifier_field",
                           "scale_factor": 0.5,  #
                           "appending_vector_position": 0, },

            "Rate": {"type": "score_modifier_field",
                     "scale_factor": 0.5,  #
                     "appending_vector_position": 1, },

            "Price": {"type": "score_modifier_field",
                      "scale_factor": 0.5,  #
                      "appending_vector_position": 2, },

            "Warranty": {"type": "score_modifier_field",
                         "scale_factor": 0.5,  #
                         "appending_vector_position": 3, }
        }

        tensor_search.create_vector_index(
            index_name=self.index_name, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True,
                    IndexSettingsField.mappings: mappings,
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name, documents=documents)


