import json
from typing import Optional

import pytest

from marqo.core.exceptions import OperationConflictError
from marqo.core.index_management.vespa_application_package import IndexSettingStore
from marqo.core.models.marqo_index import Field, FieldType, MarqoIndex, Model
from tests.marqo_test import MarqoTestCase


@pytest.mark.unittest
class TestIndexSettingStore(MarqoTestCase):

    def _get_index(self, index_name: str = 'index1', version: Optional[int] = None) -> MarqoIndex:
        return self.structured_marqo_index(
            name=index_name,
            schema_name="schema1",
            model=Model(name='hf/e5-small'),
            fields=[
                Field(name='title', type=FieldType.Text)
            ],
            tensor_fields=[],
            marqo_version='2.12.0',
            version=version,
        )

    def test_initialise_with_empty_object(self):
        store = IndexSettingStore('{}', '{}')
        self.assertEqual(len(store._index_settings), 0)
        self.assertEqual(len(store._index_settings_history), 0)

    def test_initialise_with_index_settings(self):
        index = self._get_index(version=1)

        index_json_string = json.dumps({index.name: json.loads(index.json())})
        index_history_json_string = json.dumps({index.name: [json.loads(index.json())]})

        store = IndexSettingStore(index_json_string, index_history_json_string)
        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index)
        self.assertEqual(len(store._index_settings_history), 1)
        self.assertEqual(store._index_settings_history[index.name][0], index)

    def test_to_json(self):
        index = self._get_index(version=1)

        index_json_string = json.dumps({index.name: json.loads(index.json())})
        index_history_json_string = json.dumps({index.name: [json.loads(index.json())]})

        store = IndexSettingStore(index_json_string, index_history_json_string)
        output_index, output_index_history = store.to_json()

        self.assertEqual(output_index, index_json_string)
        self.assertEqual(output_index_history, index_history_json_string)

    def test_create_new_index_without_version_should_succeed(self):
        index = self._get_index()
        store = IndexSettingStore('{}', '{}')
        store.save_index_setting(index)

        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index.copy(update={'version': 1}))

    def test_create_new_index_with_version_1_should_succeed(self):
        index = self._get_index(version=1)
        store = IndexSettingStore('{}', '{}')
        store.save_index_setting(index)

        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index)

    def test_create_new_index_with_version_2_should_raise_operation_conflict_error(self):
        index = self._get_index(version=2)
        store = IndexSettingStore('{}', '{}')
        with self.assertRaises(OperationConflictError) as e:
            store.save_index_setting(index)

        self.assertIn("The index does not exist or has been deleted", str(e.exception))

    def test_update_with_correct_version_should_succeed(self):
        index = self._get_index(version=1)
        store = IndexSettingStore('{}', '{}')
        store.save_index_setting(index)

        updated_index = index.copy(deep=True, update={'version': 2, 'marqo_version': '2.13.0'})
        store.save_index_setting(updated_index)

        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], updated_index)
        # should persist history
        self.assertEqual(len(store._index_settings_history), 1)
        self.assertEqual(len(store._index_settings_history[index.name]), 1)
        self.assertEqual(store._index_settings_history[index.name][0], index)

    def test_update_with_wrong_version_should_raise_operation_conflict_error(self):
        index = self._get_index(version=1)
        index_json_string = json.dumps({index.name: json.loads(index.json())})
        store = IndexSettingStore(index_json_string, '{}')

        # current version in the store is 1, but the target version is still 1
        updated_index = index

        with self.assertRaises(OperationConflictError) as e:
            store.save_index_setting(updated_index)

        self.assertIn("Current version 1, new version 1", str(e.exception))

    def test_delete_index_should_succeed(self):
        index = self._get_index(version=1)
        index_json_string = json.dumps({index.name: json.loads(index.json())})
        store = IndexSettingStore(index_json_string, '{}')

        store.delete_index_setting(index.name)

        self.assertNotIn(index.name, store._index_settings)
        # should persist history
        self.assertEqual(len(store._index_settings_history), 1)
        self.assertEqual(len(store._index_settings_history[index.name]), 1)
        self.assertEqual(store._index_settings_history[index.name][0], index)

    def test_delete_nonexistent_index(self):
        index = self._get_index(version=1)
        index_json_string = json.dumps({index.name: json.loads(index.json())})
        store = IndexSettingStore(index_json_string, '{}')

        store.delete_index_setting('random-index')

        # assert nothing happened to the store
        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index)
        self.assertEqual(len(store._index_settings_history), 0)

    def test_update_deleted_index_should_raise_operation_conflict_error(self):
        index = self._get_index(version=1)
        index_json_string = json.dumps({index.name: json.loads(index.json())})
        store = IndexSettingStore(index_json_string, '{}')

        store.delete_index_setting(index.name)

        with self.assertRaises(OperationConflictError) as e:
            updated_index = index.copy(deep=True, update={'version': 2, 'marqo_version': '2.13.0'})
            store.save_index_setting(updated_index)

        self.assertIn("The index does not exist or has been deleted", str(e.exception))

    def test_create_new_index_with_same_name_as_deleted_index_should_succeed(self):
        index = self._get_index(version=1)
        index_json_string = json.dumps({index.name: json.loads(index.json())})
        store = IndexSettingStore(index_json_string, '{}')

        store.delete_index_setting(index.name)
        self.assertIn(index.name, store._index_settings_history)

        store.save_index_setting(index)
        self.assertEqual(len(store._index_settings), 1)
        self.assertEqual(store._index_settings[index.name], index)
        # the history of the previous index should be deleted
        self.assertNotIn(index.name, store._index_settings_history)

    def test_history_version_limit(self):
        index = self._get_index()
        store = IndexSettingStore('{}', '{}')

        for v in range(1, 7):
            # version 1-6 is saved to store
            updated_index = index.copy(deep=True, update={'version': v})
            store.save_index_setting(updated_index)

        # latest version is 6
        self.assertEqual(store._index_settings[index.name].version, 6)
        # history should have a limit of 3 versions
        self.assertEqual(len(store._index_settings_history[index.name]), 3)
        # the history should be sorted by version number in descending order
        self.assertEqual([h.version for h in store._index_settings_history[index.name]], [5, 4, 3])