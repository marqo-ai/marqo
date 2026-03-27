"""Please have a running Marqo instance to test against!

Pass its settings to local_marqo_settings.
"""
import uuid
from enum import Enum
from typing import List, Dict, Optional
import json
import time

import unittest
from marqo.utils import construct_authorized_url
from marqo import Client
from marqo.errors import MarqoWebError
import requests


class MarqoTestCase(unittest.TestCase):

    indexes_to_delete = []
    _MARQO_URL = "http://localhost:8882"

    @classmethod
    def setUpClass(cls) -> None:
        local_marqo_settings = {
            "url": cls._MARQO_URL
        }
        cls.client_settings = local_marqo_settings
        cls.authorized_url = cls.client_settings["url"]
        # A list with index names to be cleared in each setUp call and to be deleted in tearDownClass call
        cls.indexes_to_delete: List[str] = []
        cls.client = Client(**cls.client_settings)

    @classmethod
    def tearDownClass(cls) -> None:
        # A function that will be automatically called after each test call
        # This removes all the loaded models to save memory space.
        cls.removeAllModels()
        if cls.indexes_to_delete:
            cls.delete_indexes(cls.indexes_to_delete)

    def setUp(self) -> None:
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    @classmethod
    def random_index_name(cls, prefix: Optional[str] = 'a') -> str:
        return prefix + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def create_indexes(cls, index_settings_with_name: List[Dict]):
        """A function to call the internal Marqo API to create a batch of indexes.
         Use camelCase for the keys.
        """

        r = requests.post(f"{cls._MARQO_URL}/batch/indexes/create", data=json.dumps(index_settings_with_name))

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise MarqoWebError(e)

    @classmethod
    def delete_indexes(cls, index_names: List[str]):
        r = requests.post(f"{cls._MARQO_URL}/batch/indexes/delete", data=json.dumps(index_names))

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise MarqoWebError(e)

    @classmethod
    def clear_indexes(cls, index_names: List[str]):
        for index_name in index_names:
            r = requests.delete(f"{cls._MARQO_URL}/indexes/{index_name}/documents/delete-all")
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise MarqoWebError(e)

            r_queries = requests.delete(
                f"{cls._MARQO_URL}/indexes/{index_name}/suggestions/queries/delete-all",
                headers={"Content-Type": "application/json"}
            )
            try:
                r_queries.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise MarqoWebError(e)

    @classmethod
    def removeAllModels(cls) -> None:
        # A function that can be called to remove loaded models in Marqo.
        # Use it whenever you think there is a risk of OOM problem.
        # E.g., add it into the `tearDown` function to remove models between test cases.
        loaded_models :list[dict] = requests.get(f"{cls._MARQO_URL}/models").json()["models"]
        for model in loaded_models:
            model_name = model["modelName"]
            try:
                _ = requests.delete(f"{cls._MARQO_URL}/models?model_name={model_name}")
            except requests.exceptions.HTTPError as e:
                pass


class TestImageUrls(str, Enum):
    __test__ = False  # Prevent pytest from collecting this class as a test
    IMAGE0 = 'https://marqo-assets.s3.amazonaws.com/tests/images/image0.jpg'
    IMAGE1 = 'https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg'
    IMAGE2 = 'https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg'
    IMAGE3 = 'https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg'
    IMAGE4 = 'https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg'
    HIPPO_REALISTIC = 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic_small.png'
    HIPPO_REALISTIC_LARGE = 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png'
    HIPPO_STATUE = 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue_small.png'


EXAMPLE_FASHION_DOCUMENTS = [
  {
    "_id": "1",
    "title": "Slim Fit Denim Jacket",
    "brand": "SnugNest",
    "description": "A timeless piece with a modern slim-fit design, perfect for casual layering.",
    "color": "yellow",
    "size": "S",
    "style": "casual",
    "price": 83.42
  },
  {
    "_id": "2",
    "title": "Classic Cotton Shirt",
    "brand": "SnugNest",
    "description": "Comfortable and breathable cotton shirt suitable for everyday wear.",
    "color": "red",
    "size": "M",
    "style": "partywear",
    "price": 49.03
  },
  {
    "_id": "3",
    "title": "High-Waisted Skirt",
    "brand": "PulseWear",
    "description": "Elegant skirt with a high waistline and flattering silhouette.",
    "color": "coral",
    "size": "L",
    "style": "streetwear",
    "price": 1.2
  },
  {
    "_id": "4",
    "title": "Knitted Winter Sweater",
    "brand": "SprintX",
    "description": "Chunky knit sweater designed for warmth and comfort in cold seasons.",
    "color": "red",
    "size": "Free",
    "style": "loungewear",
    "price": 92.99
  },
  {
    "_id": "5",
    "title": "Casual Linen Trousers",
    "brand": "PulseWear",
    "description": "Relaxed-fit trousers crafted from lightweight linen for maximum comfort.",
    "color": "charcoal",
    "size": "M",
    "style": "partywear",
    "price": 88.14
  },
  {
    "_id": "6",
    "title": "Embroidered Kurta",
    "brand": "RetroHue",
    "description": "Traditional kurta with intricate embroidery for festive occasions.",
    "color": "green",
    "size": "S",
    "style": "streetwear",
    "price": 81.33
  },
  {
    "_id": "7",
    "title": "Floral Summer Dress",
    "brand": "SnugNest",
    "description": "Breezy and lightweight dress ideal for sunny summer days.",
    "color": "green",
    "size": "XS",
    "style": "streetwear",
    "price": 28.71
  },
  {
    "_id": "8",
    "title": "Athletic Running Shorts",
    "brand": "PulseWear",
    "description": "Performance shorts made from moisture-wicking fabric for workouts.",
    "color": "green",
    "size": "Free",
    "style": "biker",
    "price": 73.88
  },
  {
    "_id": "9",
    "title": "Hooded Windbreaker",
    "brand": "CozyCore",
    "description": "Windproof and waterproof jacket with adjustable hood.",
    "color": "charcoal",
    "size": "S",
    "style": "streetwear",
    "price": 55.54
  },
  {
    "_id": "10",
    "title": "Fleece Zip-Up Hoodie",
    "brand": "SnugNest",
    "description": "Super soft fleece hoodie for a relaxed and cozy look.",
    "color": "gray",
    "size": "M",
    "style": "loungewear",
    "price": 49.3
  }
]
