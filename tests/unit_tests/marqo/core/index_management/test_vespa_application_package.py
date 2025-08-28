import json
import unittest
from unittest.mock import Mock

from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, VespaApplicationStore
from tests.unit_tests.marqo_test import MarqoTestCase


class TestVespaApplicationPackage(MarqoTestCase):
    """Test cases for typeahead schema integration in VespaApplicationPackage."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = Mock(spec=VespaApplicationStore)
        
        # Mock services.xml content
        self.mock_store.read_text_file.side_effect = self._mock_read_text_file
        self.mock_store.file_exists.side_effect = self._mock_file_exists
        
        self.vespa_app = VespaApplicationPackage(self.mock_store)

    def _mock_file_exists(self, *paths):
        """Mock file_exists to return True for services.xml, False for config files."""
        return paths in ('marqo_config.json', 'services.xml')

    def _mock_read_text_file(self, *paths):
        """Mock read_text_file to return appropriate content."""
        if paths == ('services.xml',):
            return '''<?xml version="1.0" encoding="utf-8"?>
<services xmlns:deploy="vespa" xmlns:preprocess="properties">
    <container>
        <document-api/>
        <document-processing/>
        <search/>
    </container>
    <content>
        <documents>
        </documents>
    </content>
</services>'''
        elif paths == ('marqo_config.json',):
            return '{"version": "2.23.0"}'
        else:
            return None

    def test_batch_add_index_setting_and_schema_includes_typeahead_schema(self):
        """Test that batch_add_index_setting_and_schema handles both main and typeahead schemas."""
        # Setup
        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            typeahead_schema_name="marqo__test_index_typeahead",
            version="1"
        )

        main_schema_content = """schema marqo__test_index {
            document marqo__test_index {
                field content type string {
                    indexing: index | summary
                }
            }
        }"""
        
        typeahead_schema_content = """schema marqo__test_index_typeahead {
            document marqo__test_index_typeahead {
                field query type string {
                    indexing: summary | attribute
                }
            }
        }"""
        
        indexes = [(main_schema_content, typeahead_schema_content, marqo_index)]
        
        # Execute
        self.vespa_app.batch_add_index_setting_and_schema(indexes)

        self.assertEqual(5, self.mock_store.save_file.call_count)
        arg_list = self.mock_store.save_file.call_args_list
        # check the new schemas are persisted
        self.assertListEqual([main_schema_content, 'schemas', 'marqo__test_index.sd'], list(arg_list[0][0]))
        self.assertListEqual([typeahead_schema_content, 'schemas', 'marqo__test_index_typeahead.sd'], list(arg_list[1][0]))

        # check the index settings are persisted
        index_setting_str = json.dumps({"test_index": json.loads(marqo_index.json())})
        self.maxDiff = None
        self.assertListEqual([index_setting_str, 'marqo_index_settings.json'], list(arg_list[2][0]))
        self.assertListEqual(['{}', 'marqo_index_settings_history.json'], list(arg_list[3][0]))

        # check the new schemas are added to service_xml file
        services_xml_store_call = arg_list[4][0]
        self.assertEqual('services.xml', services_xml_store_call[1])
        self.assertIn('<document type="marqo__test_index" mode="index" />', services_xml_store_call[0])
        self.assertIn('<document type="marqo__test_index_typeahead" mode="index" />', services_xml_store_call[0])

        # Verify deployment was called
        self.mock_store.deploy_application.assert_called_once()

    def test_batch_add_multiple_indexes_creates_multiple_typeahead_schemas(self):
        """Test that batch_add_index_setting_and_schema handles multiple indexes with typeahead schemas."""
        # Setup
        index1 = self.semi_structured_marqo_index(
            name="products",
            schema_name="marqo__products",
            typeahead_schema_name="marqo__products_typeahead",
        )

        index2 = self.semi_structured_marqo_index(
            name="users",
            schema_name="marqo__users",
            typeahead_schema_name="marqo__users_typeahead",
        )

        indexes = [
            ("products_main_schema", "products_typeahead_schema", index1),
            ("users_main_schema", "users_typeahead_schema", index2)
        ]
        
        # Execute
        self.vespa_app.batch_add_index_setting_and_schema(indexes)
        
        # Verify all four schemas were saved (2 main + 2 typeahead)
        schema_saves = {}
        for call in self.mock_store.save_file.call_args_list:
            args, kwargs = call
            if len(args) >= 3 and args[1] == 'schemas':
                schema_saves[args[2]] = args[0]
        
        expected_schemas = {
            'marqo__products.sd': 'products_main_schema',
            'marqo__products_typeahead.sd': 'products_typeahead_schema',
            'marqo__users.sd': 'users_main_schema',
            'marqo__users_typeahead.sd': 'users_typeahead_schema'
        }
        
        for schema_file, expected_content in expected_schemas.items():
            self.assertIn(schema_file, schema_saves, f"Schema file {schema_file} should be saved")
            self.assertEqual(schema_saves[schema_file], expected_content)
        
        # Verify deployment was called once for the batch
        self.mock_store.deploy_application.assert_called_once()

    def test_batch_add_with_existing_index_raises_error(self):
        """Test that adding an index that already exists raises IndexExistsError."""
        # Setup - mock has_index to return True
        self.vespa_app.has_index = Mock(return_value=True)
        
        marqo_index = self.semi_structured_marqo_index(
            name="existing_index",
            schema_name="marqo__existing_index",
            typeahead_schema_name="marqo__existing_index_typeahead"
        )

        indexes = [("main_schema", "typeahead_schema", marqo_index)]
        
        # Execute & Verify
        with self.assertRaises(Exception) as context:
            self.vespa_app.batch_add_index_setting_and_schema(indexes)
        
        # Should raise IndexExistsError
        self.assertIn("already exists", str(context.exception))
        
        # Verify no deployment happened
        self.mock_store.deploy_application.assert_not_called()

    def test_batch_delete_index_setting_and_schema_removes_typeahead_schema(self):
        """Test that batch_delete_index_setting_and_schema removes both main and typeahead schemas."""
        # Setup
        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            typeahead_schema_name="marqo__test_index_typeahead"
        )
        
        # Mock the index store to return the index
        self.vespa_app._index_setting_store.get_index = Mock(return_value=marqo_index)
        
        # Execute
        self.vespa_app.batch_delete_index_setting_and_schema(["test_index"])
        
        # Verify that both main and typeahead schema files are removed
        expected_remove_calls = [
            unittest.mock.call('schemas', 'marqo__test_index.sd'),
            unittest.mock.call('schemas', 'marqo__test_index_typeahead.sd')
        ]
        
        actual_remove_calls = [
            call for call in self.mock_store.remove_file.call_args_list
            if len(call[0]) == 2 and call[0][0] == 'schemas'
        ]
        
        self.assertEqual(len(actual_remove_calls), 2)
        self.assertIn(expected_remove_calls[0], actual_remove_calls)
        self.assertIn(expected_remove_calls[1], actual_remove_calls)
        
        # Verify deployment was called
        self.mock_store.deploy_application.assert_called_once()

    def test_batch_delete_index_setting_and_schema_handles_no_typeahead_schema(self):
        """Test that batch_delete_index_setting_and_schema works when index has no typeahead schema."""
        # Setup - index without typeahead schema
        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            typeahead_schema_name=None  # No typeahead schema
        )
        
        # Mock the index store to return the index
        self.vespa_app._index_setting_store.get_index = Mock(return_value=marqo_index)
        
        # Execute
        self.vespa_app.batch_delete_index_setting_and_schema(["test_index"])
        
        # Verify that only the main schema file is removed
        schema_remove_calls = [
            call for call in self.mock_store.remove_file.call_args_list
            if len(call[0]) == 2 and call[0][0] == 'schemas'
        ]
        
        self.assertEqual(len(schema_remove_calls), 1)
        self.assertEqual(schema_remove_calls[0], unittest.mock.call('schemas', 'marqo__test_index.sd'))
        
        # Verify deployment was called
        self.mock_store.deploy_application.assert_called_once()

    def test_batch_delete_multiple_indexes_removes_all_typeahead_schemas(self):
        """Test that batch_delete_index_setting_and_schema removes typeahead schemas for multiple indexes."""
        # Setup
        index1 = self.semi_structured_marqo_index(
            name="products",
            schema_name="marqo__products",
            typeahead_schema_name="marqo__products_typeahead"
        )
        
        index2 = self.semi_structured_marqo_index(
            name="users",
            schema_name="marqo__users",
            typeahead_schema_name="marqo__users_typeahead"
        )
        
        index3 = self.semi_structured_marqo_index(
            name="categories",
            schema_name="marqo__categories",
            typeahead_schema_name=None  # No typeahead schema
        )
        
        # Mock the index store to return appropriate indexes
        def mock_get_index(name):
            if name == "products":
                return index1
            elif name == "users":
                return index2
            elif name == "categories":
                return index3
            return None
        
        self.vespa_app._index_setting_store.get_index = Mock(side_effect=mock_get_index)
        
        # Execute
        self.vespa_app.batch_delete_index_setting_and_schema(["products", "users", "categories"])
        
        # Verify that all main schemas and typeahead schemas (where they exist) are removed
        schema_remove_calls = [
            call for call in self.mock_store.remove_file.call_args_list
            if len(call[0]) == 2 and call[0][0] == 'schemas'
        ]
        
        expected_removes = [
            unittest.mock.call('schemas', 'marqo__products.sd'),
            unittest.mock.call('schemas', 'marqo__products_typeahead.sd'),
            unittest.mock.call('schemas', 'marqo__users.sd'),
            unittest.mock.call('schemas', 'marqo__users_typeahead.sd'),
            unittest.mock.call('schemas', 'marqo__categories.sd')
            # Note: marqo__categories_typeahead.sd should NOT be removed since it's None
        ]
        
        self.assertEqual(len(schema_remove_calls), 5)
        for expected_call in expected_removes:
            self.assertIn(expected_call, schema_remove_calls)
        
        # Ensure typeahead schema for categories was NOT removed
        categories_typeahead_call = unittest.mock.call('schemas', 'marqo__categories_typeahead.sd')
        self.assertNotIn(categories_typeahead_call, schema_remove_calls)
        
        # Verify deployment was called once for the batch
        self.mock_store.deploy_application.assert_called_once()

if __name__ == '__main__':
    unittest.main()