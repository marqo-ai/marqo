import json
import os

import pytest

from marqo.core.index_management.vespa_application_package import IndexSettingStore
from marqo.core.models.marqo_index import Field, FieldType
from tests.marqo_test import MarqoTestCase


@pytest.mark.unnittest
class TestIndexSettingStore(MarqoTestCase):

    def setUp(self) -> None:
        self.index1 = self.structured_marqo_index(
            name ="index1",
            schema_name="schema1",
            fields=[
                Field(name='title', type=FieldType.Text)
            ],
            tensor_fields=[],
            marqo_version='2.12.0'
        )

    def test_initialise_with_empty_object(self):
        store = IndexSettingStore('{}', '{}')
        self.assertEqual(len(store._index_settings), 0)
        self.assertEqual(len(store._index_settings_history), 0)

    def test_initialise_with_index_settings(self):
        index = self.index1

        index_json_string = json.dumps({index.name: json.loads(index.json())})
        index_history_json_string = json.dumps({index.name: [json.loads(index.json())]})

        store = IndexSettingStore(index_json_string, index_history_json_string)
        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index)
        self.assertEqual(len(store._index_settings_history), 1)
        self.assertEqual(store._index_settings_history[index.name][0], index)

    def test_to_json_strings(self):
        index = self.index1

        index_json_string = json.dumps({index.name: json.loads(index.json())})
        index_history_json_string = json.dumps({index.name: [json.loads(index.json())]})

        store = IndexSettingStore(index_json_string, index_history_json_string)
        output_index, output_index_history = store.to_json()

        self.assertEqual(output_index, index_json_string)
        self.assertEqual(output_index_history, index_history_json_string)
