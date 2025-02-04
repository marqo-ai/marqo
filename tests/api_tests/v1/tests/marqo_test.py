"""Please have a running Marqo instance to test against!

Pass its settings to local_marqo_settings.
"""
from typing import List, Dict
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
        client = Client(**cls.client_settings)
        index_names_list: List[str] = [item["indexName"] for item in client.get_indexes()["results"]]
        for index_name in index_names_list:
            loaded_models = client.index(index_name).get_loaded_models().get("models", [])
            for model in loaded_models:
                try:
                    client.index(index_name).eject_model(model_name=model["model_name"], model_device=model["model_device"])
                except MarqoWebError:
                    pass

