"""Unit tests for IndexManagement.apply_latest_schema_template() method."""
import unittest
from unittest.mock import Mock, MagicMock, patch

from marqo.core.exceptions import IndexNotFoundError, InternalError, UnsupportedFeatureError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.index_management.vespa_application_package import (
    VespaApplicationPackage,
    ApplicationPackageDeploymentSessionStore,
    VespaApplicationFileStore
)
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, StructuredMarqoIndex
from tests.unit_tests.marqo_test import MarqoTestCase


class TestIndexManagementSchemaUpdate(MarqoTestCase):
    """Test cases for IndexManagement.apply_latest_schema_template() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock()
        self.mock_zookeeper_client = Mock()

        # Create IndexManagement instance with mocks
        self.index_mgmt = IndexManagement(
            vespa_client=self.mock_vespa_client,
            zookeeper_client=self.mock_zookeeper_client,
            enable_index_operations=True
        )

        # Mock the vespa deployment lock to be a no-op context manager
        self.mock_lock = MagicMock()
        self.mock_lock.__enter__ = Mock(return_value=None)
        self.mock_lock.__exit__ = Mock(return_value=None)
        self.index_mgmt._vespa_deployment_lock = Mock(return_value=self.mock_lock)

    def test_apply_latest_schema_template_no_changes(self):
        """Test update when schema is already up-to-date."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { document test_schema {} }"

        # Mock get_index to return test index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)
        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation to return same schema
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = current_schema

            # Execute
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

        # Verify
        self.assertFalse(result['updated'])
        self.assertFalse(result['schemaChanged'])
        self.assertEqual(result['reason'], "Schema is already up to date")
        self.assertEqual(result['configChangeActions'], {})
        mock_vespa_app.update_index_setting_and_schema.assert_not_called()

    def test_apply_latest_schema_template_with_changes_no_actions(self):
        """Test update when schema changed but no Vespa actions required."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { document test_schema { field old_field type string {} } }"
        new_schema = "schema test_schema { document test_schema { field new_field type string {} } }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        # Mock prepare response with no actions
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

        # Verify
        self.assertTrue(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertEqual(result['reason'], "Schema updated successfully")
        self.assertEqual(result['configChangeActions'], {})

        # Verify prepare was called
        mock_vespa_app.update_index_setting_and_schema.assert_called_once()
        call_args = mock_vespa_app.update_index_setting_and_schema.call_args
        self.assertTrue(call_args[1]['prepare_only'])

        # Verify activate was called
        mock_vespa_app.activate_prepared_deployment.assert_called_once_with(prepare_response)

    def test_apply_latest_schema_template_with_actions_not_forced(self):
        """Test update blocks when actions required and force=False."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        # Mock prepare response with restart action
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [
                    {
                        'name': 'restart',
                        'services': ['searchnode'],
                        'messages': ['Field new_field added']
                    }
                ]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

        # Verify
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertIn("Vespa requires manual actions", result['reason'])
        self.assertIn('restart', result['configChangeActions'])

        # Verify activate was NOT called
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_with_actions_forced(self):
        """Test update proceeds when actions required but force=True."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        # Mock prepare response with restart action
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [
                    {
                        'name': 'restart',
                        'services': ['searchnode']
                    }
                ]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=True)

        # Verify
        self.assertTrue(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertEqual(result['reason'], "Update forced despite required actions")
        self.assertIn('restart', result['configChangeActions'])

        # Verify activate WAS called
        mock_vespa_app.activate_prepared_deployment.assert_called_once_with(prepare_response)

    def test_apply_latest_schema_template_index_not_found(self):
        """Test error when index doesn't exist."""
        # Mock get_index to raise IndexNotFoundError
        self.index_mgmt.get_index = Mock(side_effect=IndexNotFoundError("Index not found"))

        # Execute and verify exception
        with self.assertRaises(IndexNotFoundError):
            self.index_mgmt.apply_latest_schema_template("nonexistent_index")

    def test_apply_latest_schema_template_wrong_index_type(self):
        """Test error when index is not SemiStructuredMarqoIndex."""
        # Setup structured index
        test_index = self.structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Execute and verify exception
        with self.assertRaises(InternalError) as context:
            self.index_mgmt.apply_latest_schema_template("test_index")

        self.assertIn("only semi-structured indexes support schema updates", str(context.exception))

    def test_configChangeActions_detection_refeed(self):
        """Test detection of refeed actions."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        self.index_mgmt.get_index = Mock(return_value=test_index)

        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        # Mock prepare response with refeed action
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'refeed': [
                    {
                        'name': 'refeed',
                        'documentType': 'test_schema',
                        'clusterName': 'content'
                    }
                ]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)
        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute with force=False
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

        # Verify - should block on refeed action
        self.assertFalse(result['updated'])
        self.assertIn('refeed', result['configChangeActions'])
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_configChangeActions_detection_reindex(self):
        """Test detection of reindex actions."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        self.index_mgmt.get_index = Mock(return_value=test_index)

        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        # Mock prepare response with reindex action
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'reindex': [
                    {
                        'name': 'reindex',
                        'documentType': 'test_schema'
                    }
                ]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)
        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute with force=False
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

        # Verify - should block on reindex action
        self.assertFalse(result['updated'])
        self.assertIn('reindex', result['configChangeActions'])
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_version_too_old(self):
        """Test error when index was created with Marqo < 2.23.0."""
        # Setup index with old version
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema",
            marqo_version="2.22.0"  # Below 2.23.0
        )

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Execute and verify exception
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.index_mgmt.apply_latest_schema_template("test_index")

        # Verify error message contains version information
        self.assertIn("2.23.0", str(context.exception))
        self.assertIn("2.22.0", str(context.exception))

    def test_apply_latest_schema_template_dry_run_no_changes(self):
        """Test dry_run when schema is already up to date."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { document test_schema {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation to return same schema
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = current_schema

            # Execute with dry_run=True
            result = self.index_mgmt.apply_latest_schema_template("test_index", dry_run=True)

        # Verify
        self.assertFalse(result['updated'])
        self.assertFalse(result['schemaChanged'])
        self.assertEqual(result['reason'], "Schema is already up to date")
        self.assertIn('oldSchema', result)
        self.assertIn('newSchema', result)
        self.assertIn('schemaDiff', result)
        self.assertEqual(result['schemaDiff'], 'No changes')

    def test_apply_latest_schema_template_dry_run_with_changes(self):
        """Test dry_run with schema changes - should not deploy."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema { document test_schema { field old_field type string {} } }"
        new_schema = "schema test_schema { document test_schema { field new_field type string {} } }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)

        # Mock prepare response with no actions
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute with dry_run=True
            result = self.index_mgmt.apply_latest_schema_template("test_index", dry_run=True)

        # Verify
        self.assertFalse(result['updated'])  # Should not be updated in dry run
        self.assertTrue(result['schemaChanged'])
        self.assertEqual(result['reason'], "Dry run - no changes deployed")
        self.assertIn('oldSchema', result)
        self.assertIn('newSchema', result)
        self.assertIn('schemaDiff', result)
        self.assertNotEqual(result['schemaDiff'], 'No changes')

        # Verify prepare was called but activate was NOT
        mock_vespa_app.update_index_setting_and_schema.assert_called_once()
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_dry_run_with_actions(self):
        """Test dry_run with actions required - should still not deploy."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema {}"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)

        # Mock prepare response with restart action
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [{'name': 'restart', 'services': ['searchnode']}]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute with dry_run=True
            result = self.index_mgmt.apply_latest_schema_template("test_index", dry_run=True)

        # Verify
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertEqual(result['reason'], "Dry run - no changes deployed")
        self.assertIn('restart', result['configChangeActions'])

        # Verify activate was NOT called
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_dry_run_ignores_force(self):
        """Test that dry_run takes precedence over force parameter."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )

        current_schema = "schema test_schema {}"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)

        # Mock prepare response with actions
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [{'name': 'restart'}]
            }
        }
        mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute with both dry_run=True and force=True
            result = self.index_mgmt.apply_latest_schema_template("test_index", force=True, dry_run=True)

        # Verify - dry_run should take precedence, no deployment
        self.assertFalse(result['updated'])
        self.assertEqual(result['reason'], "Dry run - no changes deployed")
        mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_with_old_vespa_store_raises_error(self):
        """Test that prepare_only with VespaApplicationFileStore raises InternalError.

        This covers the error path in update_index_setting_and_schema() when prepare_only=True
        is used with VespaApplicationFileStore (old Vespa < 8.382.22 that doesn't support
        deployment session API).
        """
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema"
        )
        current_schema = "schema test_schema { document test_schema {} }"
        new_schema = "# Modified\nschema test_schema { document test_schema {} }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Create VespaApplicationPackage with VespaApplicationFileStore (old Vespa)
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema.return_value = current_schema

        # Create actual VespaApplicationFileStore to trigger the isinstance check
        mock_file_store = Mock(spec=VespaApplicationFileStore)
        mock_vespa_app._store = mock_file_store

        # When update_index_setting_and_schema is called with prepare_only=True,
        # it should raise InternalError because VespaApplicationFileStore doesn't support it
        def raise_internal_error(*args, **kwargs):
            if kwargs.get('prepare_only'):
                raise InternalError("prepare_only mode requires ApplicationPackageDeploymentSessionStore")
            return None

        mock_vespa_app.update_index_setting_and_schema = Mock(side_effect=raise_internal_error)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

        # Mock schema generation
        with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
            mock_schema_class.generate_vespa_schema.return_value = new_schema

            # Execute - this should raise InternalError
            with self.assertRaises(InternalError) as ctx:
                self.index_mgmt.apply_latest_schema_template("test_index")

        # Verify error message
        self.assertIn("prepare_only mode requires ApplicationPackageDeploymentSessionStore", str(ctx.exception))

    def test_activate_prepared_deployment_with_old_vespa_store_raises_error(self):
        """Test that activate_prepared_deployment with VespaApplicationFileStore raises InternalError.

        This directly tests the activate_prepared_deployment() method's error path when called
        with VespaApplicationFileStore.
        """
        # Create a mock VespaApplicationFileStore with proper XML content
        mock_file_store = Mock(spec=VespaApplicationFileStore)
        mock_file_store.file_exists.return_value = True

        # Return valid XML for services.xml and JSON for config files
        def mock_read_text_file(filename):
            if filename == 'services.xml':
                return '''<?xml version="1.0" encoding="utf-8" ?>
                <services version="1.0">
                    <container id="default" version="1.0"></container>
                    <content id="content_default" version="1.0">
                        <documents>
                            <document type="test" mode="index"/>
                        </documents>
                    </content>
                </services>'''
            elif filename == 'marqo_config.json':
                return '{"version": "1.0.0"}'
            elif filename in ['marqo_index_settings.json', 'marqo_index_settings_history.json']:
                return '{}'
            return None

        mock_file_store.read_text_file.side_effect = mock_read_text_file

        # Create VespaApplicationPackage with the file store
        vespa_app = VespaApplicationPackage(store=mock_file_store)

        # Prepare response
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }

        # Execute - should raise InternalError because VespaApplicationFileStore doesn't support
        # the two-phase deployment (prepare/activate separately)
        with self.assertRaises(InternalError) as ctx:
            vespa_app.activate_prepared_deployment(prepare_response)

        # Verify error message
        self.assertIn("Deployment activation requires ApplicationPackageDeploymentSessionStore", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
