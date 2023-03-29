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
        pass

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.index_name = "my-test-index-1"
        try:
            tensor_search.delete_index(config=cls.config, index_name=cls.index_name)
        except IndexNotFoundError as e:
            pass

        cls.documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "reputation" : 1,
             #"rate" : 20,
             "_id" : "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "reputation": 2,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "reputation": 3,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "reputation": 4,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
             "reputation": 5,
             "_id": "5"
             },
            # {"my_test_field": "A rider is riding a horse jumping over the barrier.",
            #  "my_image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            #  "_id": "6"
            #  },
        ]
        tensor_search.create_vector_index(
            index_name=cls.index_name, config=cls().config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "random/small",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })

        tensor_search.add_documents(config=cls().config, index_name=cls.index_name, docs = cls.documents,
                                    non_tensor_fields=["reputation"], auto_refresh=True)


    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except:
            pass

    def test_search_results(self):


        #pprint(tensor_search.get_index_info(config=self.config, index_name=self.index_name)[1])
        res = tensor_search._vector_text_search(config=self.config, index_name=self.index_name,
                                                query="what is the rider doing?",
                                                custom_score_fields = [
                                                    {"field_name" : "reputation" ,
                                                     "weight" : 1,
                                                     "combine_style" : "multiply"
                                                     },
                                                    {"field_name" : "rate" ,
                                                     "weight" : 1,
                                                     "combine_style" : "additive"
                                                    }
                                                ])
        # pprint(res["hits"])
        # ids = [i['_id'] for i in res["hits"]]
        # assert ids == ["1", "5", "4", "3", "2"]

        pprint(tensor_search.get_index_info(config=self.config, index_name=self.index_name)[1])



"""
    double additive = 0;
    if (doc['__chunks.reputation'].size() > 0 &&
        (doc['__chunks.reputation'].value instanceof java.lang.Number)) {
        copy_score = copy_score * doc['__chunks.reputation'].value * 1;
    }
    
    if (doc['__chunks.rate'].size() > 0 &&
    (doc['__chunks.rate'].value instanceof java.lang.Number)) {
    additive = additive + doc['__chunks.rate'].value * 1;
    }
    
    return _score + additive;
"""