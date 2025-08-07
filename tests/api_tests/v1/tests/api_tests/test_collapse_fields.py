import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError

class TestCollapseFields(MarqoTestCase):
    unstructured_text_index_name = "unstructured_index_text" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)
        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "collapseFields": [
                    {"name": "parent_id", "minGroups": 100}
                ]
            },
        ])

        cls.indexes_to_delete = [cls.unstructured_text_index_name]

    def test_collapse_fields_is_in_index_settings(self):
        index_name = self.unstructured_text_index_name
        index_settings = self.client.index(index_name).get_settings()
        self.assertTrue("collapseFields" in index_settings)
        self.assertEqual(len(index_settings["collapseFields"]), 1)
        self.assertEqual(index_settings["collapseFields"][0], {"name": "parent_id", "minGroups": 100})

