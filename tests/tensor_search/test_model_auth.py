import pprint
import time
from marqo.tensor_search import enums, backend
from marqo.tensor_search import tensor_search
import unittest
import copy
from marqo.errors import InvalidArgError, IndexNotFoundError
from tests.marqo_test import MarqoTestCase
import random


class TestModelAuth(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    @staticmethod
    def _get_base_index_settings():
        return {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": 'my_model',
                "normalize_embeddings": True,
                # notice model properties aren't here. Each test has to add it
            }
        }

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_model_auth_s3(self):
        """check against hardcoded signed URL. """
        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": 'your-bucket-name',
                    "Key": 'path/to/your/object.ext',

                },
                "auth_required": True
            },
            "type": "clip",
        }
        s3_settings = self._get_base_index_settings()
        s3_settings['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)
        # TODO: mock presigned URL
        raise NotImplementedError

    def test_model_auth_hf(self):
        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "repo_id": "Marqo/test-private",
                    "filename": "dummy_model.pt",
                },
                "auth_required": True
            },
            "type": "clip",
        }
        s3_settings = self._get_base_index_settings()
        s3_settings['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)
        # TODO: mock call to HF
        raise NotImplementedError
