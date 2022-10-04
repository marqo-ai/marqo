import unittest
from marqo.client import Client
from marqo.errors import IndexNotFoundError
from marqo.tensor_search import tensor_search

from marqo.s2_inference.s2_inference import vectorise
import pprint

from tests.marqo_test import MarqoTestCase


'''
Temporary test file, will refactor it for unit testing purposes later on
'''

class TestGenericHuggingFaceModels():

    def __init__(self) -> None:
        #giving an error??
        # mq = Client(**self.client_settings)
        # self.config = mq.config
        # self.client = mq

        self.index_name_1 = "my-test-hf-index-1"

    def _get_settings(self) -> dict:
        """Helper function"""

        settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "image_preprocessing": {
                    "patch_method": "simple"
                },
                "model": {"name": "sentence-transformers/all-mpnet-base-v2",
                        "dimensions": 768,
                        "tokens":128,
                        "type":"sbert"},
                "normalize_embeddings":True,
            },
        }

        return settings

    def _delete_index(self, index_name: str):
        try:
            tensor_search.delete_index(config=self.config, index_name=index_name)
            print("Index successfully deleted.")
        except Exception as e:
            print(e)
            print("Index does not exist.")

    def _create_index(self, index_name: str, settings):
        try:
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, **settings)
            print("index created")
        except Exception as e:
            print(e)
            print("index already exists")

    def _run_test(self):
        settings = self._get_settings()
        self._delete_index(self.index_name_1)
        self._create_index(self.index_name_1, settings)
        model = settings["index_defaults"]["model"] #dict
        result = vectorise(model, "some string")
        #print(result)

data = [{'name': 'document 1'}]
_test_client = TestGenericHuggingFaceModels()
_test_client._run_test()
