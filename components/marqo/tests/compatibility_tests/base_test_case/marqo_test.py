"""This test class requires you to have a running Marqo instance to test against!

Pass its settings to local_marqo_settings.
"""
from typing import List, Dict
import json

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
        # These indexes will:
        # 1) be cleared in each setUp call
        # 2) be deleted in tearDownClass call
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

    @classmethod
    def removeAllModels(cls) -> None:
        # A function that can be called to remove loaded models in Marqo.
        # Use it whenever you think there is a risk of OOM problem.
        # E.g., add it into the `tearDown` function to remove models between test cases.
        loaded_models :list[dict] = requests.get(f"{cls._MARQO_URL}/models").json()["models"]
        for model in loaded_models:
            if "model_name" in model:
                try:
                    _ = requests.delete(f"{cls._MARQO_URL}/models?model_name={model['model_name']}&device={model['model_device']}")
                except requests.exceptions.HTTPError as e:
                    pass
            # We remove the device concept in 2.25.0
            if "modelName" in model:
                try:
                    _ = requests.delete(f"{cls._MARQO_URL}/models?model_name={model['modelName']}")
                except requests.exceptions.HTTPError as e:
                    pass
