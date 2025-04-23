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
from tests.compatibility_tests.compatibility_test_logger import get_logger


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

        if not hasattr(cls, 'logger'):
            cls.logger = get_logger(f"tests.compatibility_tests.{cls.__module__}.{cls.__name__}")

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
        cls.logger.debug(f"Starting index creation method.")

        # Attempt to delete all existing indexes first
        index_names = [index["indexName"] for index in index_settings_with_name]
        try:
            cls.logger.debug(f"First attempting to run batch delete on {index_names}.")
            r = requests.post(f"{cls._MARQO_URL}/batch/indexes/delete", data=json.dumps(index_names))
            cls.logger.debug(r.text)
        except requests.exceptions.HTTPError as e:
            cls.logger.debug(f"Initial error deleting indexes: {e}")
            pass  # Ignore errors if indexes don't exists

        # Now create the indexes
        cls.logger.debug(f"Creating indexes {index_settings_with_name}")
        r = requests.post(f"{cls._MARQO_URL}/batch/indexes/create", data=json.dumps(index_settings_with_name))
        cls.logger.debug(r.text)

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise MarqoWebError(e)

        cls.logger.debug(f"Succeeded creating indexes {index_settings_with_name}")


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

