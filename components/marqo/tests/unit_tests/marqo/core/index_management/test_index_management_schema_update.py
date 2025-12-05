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

    def _create_test_index(self, schema_template_version=None, marqo_version=None):
        """Helper to create a test index with common defaults."""
        kwargs = {
            'schema_template_version': schema_template_version,
            'name': 'test_index',
            'schema_name': 'test_schema'
        }
        if marqo_version:
            kwargs['marqo_version'] = marqo_version
        return self.semi_structured_marqo_index(**kwargs)

    def _setup_vespa_app_mock(self, current_schema, prepare_response=None):
        """Helper to set up vespa application mock."""
        mock_vespa_app = Mock(spec=VespaApplicationPackage)
        mock_vespa_app.get_schema = Mock(return_value=current_schema)
        mock_vespa_app._store = Mock(spec=ApplicationPackageDeploymentSessionStore)

        if prepare_response is not None:
            mock_vespa_app.update_index_setting_and_schema = Mock(return_value=prepare_response)

        self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)
        return mock_vespa_app

    def _create_prepare_response(self, actions=None):
        """Helper to create a prepare response with given actions."""
        response = {
            'activate': 'http://activate_url',
            'configChangeActions': actions or {}
        }
        return response

    def test_apply_latest_schema_template_no_changes(self):
        """Test update when schema is already up-to-date."""
        # Setup
        test_index = self._create_test_index()
        current_schema = "schema test_schema { document test_schema {} }"

        # Mock get_index to return test index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        mock_vespa_app = self._setup_vespa_app_mock(current_schema)

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
        test_index = self._create_test_index()
        current_schema = "schema test_schema { document test_schema { field old_field type string {} } }"
        new_schema = "schema test_schema { document test_schema { field new_field type string {} } }"

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Mock vespa application
        prepare_response = self._create_prepare_response()
        mock_vespa_app = self._setup_vespa_app_mock(current_schema, prepare_response)

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

    def test_apply_latest_schema_template_force_parameter(self):
        """Test force parameter behavior with actions required."""
        test_index = self._create_test_index()
        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Test cases: (force, should_update, expected_reason)
        test_cases = [
            (False, False, "Vespa requires manual actions"),
            (True, True, "Update forced despite required actions")
        ]

        for force, should_update, expected_reason_fragment in test_cases:
            with self.subTest(force=force):
                # Mock get_index
                self.index_mgmt.get_index = Mock(return_value=test_index)

                # Mock vespa application with restart action
                prepare_response = self._create_prepare_response({
                    'restart': [{'name': 'restart', 'services': ['searchnode']}]
                })
                mock_vespa_app = self._setup_vespa_app_mock(current_schema, prepare_response)

                # Mock schema generation
                with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
                    mock_schema_class.generate_vespa_schema.return_value = new_schema

                    # Execute
                    result = self.index_mgmt.apply_latest_schema_template("test_index", force=force)

                # Verify
                self.assertEqual(result['updated'], should_update)
                self.assertTrue(result['schemaChanged'])
                self.assertIn(expected_reason_fragment, result['reason'])
                self.assertIn('restart', result['configChangeActions'])

                # Verify activate was called only when force=True
                if should_update:
                    mock_vespa_app.activate_prepared_deployment.assert_called_once_with(prepare_response)
                else:
                    mock_vespa_app.activate_prepared_deployment.assert_not_called()

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

    def test_configChangeActions_detection(self):
        """Test detection of different configChangeActions."""
        test_index = self._create_test_index()
        current_schema = "schema test_schema { }"
        new_schema = "schema test_schema { field new_field type string {} }"

        # Test cases: (action_type, action_data)
        test_cases = [
            ('refeed', {'refeed': [{'name': 'refeed', 'documentType': 'test_schema', 'clusterName': 'content'}]}),
            ('reindex', {'reindex': [{'name': 'reindex', 'documentType': 'test_schema'}]}),
            ('restart', {'restart': [{'name': 'restart', 'services': ['searchnode']}]})
        ]

        for action_type, action_data in test_cases:
            with self.subTest(action=action_type):
                self.index_mgmt.get_index = Mock(return_value=test_index)

                # Mock vespa application with the specific action
                prepare_response = self._create_prepare_response(action_data)
                mock_vespa_app = self._setup_vespa_app_mock(current_schema, prepare_response)

                with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
                    mock_schema_class.generate_vespa_schema.return_value = new_schema

                    # Execute with force=False
                    result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

                # Verify - should block on the action
                self.assertFalse(result['updated'])
                self.assertIn(action_type, result['configChangeActions'])
                mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_version_too_old(self):
        """Test error when index was created with Marqo < 2.23.0."""
        # Setup index with old version
        test_index = self._create_test_index(marqo_version="2.22.0")

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Execute and verify exception
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.index_mgmt.apply_latest_schema_template("test_index")

        # Verify error message contains version information
        self.assertIn("2.23.0", str(context.exception))
        self.assertIn("2.22.0", str(context.exception))

    def test_apply_latest_schema_template_index_from_future_version(self):
        """Test error when index was created with Marqo version > current version."""
        # Setup index with future version
        test_index = self._create_test_index(marqo_version="2.99.0")

        # Mock get_index
        self.index_mgmt.get_index = Mock(return_value=test_index)

        # Execute and verify exception
        with self.assertRaises(InternalError) as context:
            self.index_mgmt.apply_latest_schema_template("test_index")

        # Verify error message contains version information
        error_msg = str(context.exception)
        self.assertIn("The index was created with a newer version of Marqo than is currently running.", error_msg)

    def test_apply_latest_schema_template_dry_run(self):
        """Test dry_run mode in different scenarios."""
        # Test cases: (scenario_name, schemas_match, has_actions)
        test_cases = [
            ("no_changes", True, False),
            ("with_changes", False, False),
            ("with_actions", False, True),
            ("ignores_force", False, True)  # Test dry_run takes precedence over force
        ]

        for scenario, schemas_match, has_actions in test_cases:
            with self.subTest(scenario=scenario):
                test_index = self._create_test_index()
                current_schema = "schema test_schema { document test_schema {} }"
                new_schema = current_schema if schemas_match else "schema test_schema { field new_field type string {} }"

                # Mock get_index
                self.index_mgmt.get_index = Mock(return_value=test_index)

                # Mock vespa application
                actions = {'restart': [{'name': 'restart', 'services': ['searchnode']}]} if has_actions else {}
                prepare_response = self._create_prepare_response(actions)
                mock_vespa_app = self._setup_vespa_app_mock(current_schema, prepare_response)

                # Mock schema generation
                with patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema') as mock_schema_class:
                    mock_schema_class.generate_vespa_schema.return_value = new_schema

                    # Execute with dry_run=True (and force=True for "ignores_force" scenario)
                    force = (scenario == "ignores_force")
                    result = self.index_mgmt.apply_latest_schema_template("test_index", dry_run=True, force=force)

                # Verify - should never update in dry run
                self.assertFalse(result['updated'])

                if schemas_match:
                    self.assertFalse(result['schemaChanged'])
                    self.assertEqual(result['reason'], "Schema is already up to date")
                    self.assertEqual(result['schemaDiff'], 'No changes')
                else:
                    self.assertTrue(result['schemaChanged'])
                    self.assertEqual(result['reason'], "Dry run - no changes deployed")
                    self.assertNotEqual(result['schemaDiff'], 'No changes')

                # Verify result contains schema information
                self.assertIn('oldSchema', result)
                self.assertIn('newSchema', result)
                self.assertIn('schemaDiff', result)

                # Verify activate was NEVER called in dry run
                mock_vespa_app.activate_prepared_deployment.assert_not_called()

    def test_apply_latest_schema_template_with_old_vespa_store_raises_error(self):
        """Test that prepare_only with VespaApplicationFileStore raises InternalError.

        This covers the error path in update_index_setting_and_schema() when prepare_only=True
        is used with VespaApplicationFileStore (old Vespa < 8.382.22 that doesn't support
        deployment session API).
        """
        # Setup
        test_index = self._create_test_index()
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

    @patch('marqo.version.get_version')
    @patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_apply_latest_schema_template_schema_template_version_handling(self, mock_generate_schema, mock_get_version):
        """Test schema_template_version-based shortcut and normal processing paths."""
        mock_get_version.return_value = "2.24.6"

        # Test cases: (scenario, schema_template_version, should_shortcut)
        test_cases = [
            ("current", "2.24.6", "2.24.6", True),
            ("outdated", "2.24.0", "2.24.5", False),
            ("none", "2.24.4", None, False)
        ]

        for scenario, marqo_version, schema_template_version, should_shortcut in test_cases:
            with self.subTest(scenario=scenario, schema_template_version=schema_template_version):
                # Create an index with the test schema_template_version
                existing_index = self._create_test_index(schema_template_version=schema_template_version,
                                                         marqo_version=marqo_version)
                self.index_mgmt.get_index = Mock(return_value=existing_index)

                if should_shortcut:
                    # Test shortcut path
                    result = self.index_mgmt.apply_latest_schema_template("test_index", force=False)

                    # Verify early shortcut response
                    self.assertFalse(result["updated"])
                    self.assertFalse(result["schemaChanged"])
                    self.assertIn("already at current Marqo version 2.24.6", result["reason"])
                else:
                    # Test normal path
                    mock_generate_schema.return_value = "new_schema_content"
                    mock_generate_schema.reset_mock()

                    mock_vespa_app = Mock()
                    mock_vespa_app.get_schema.return_value = "old_schema_content"
                    self.index_mgmt._get_vespa_application = Mock(return_value=mock_vespa_app)

                    self.index_mgmt.apply_latest_schema_template("test_index", force=False)

                    # Verify schema generation was called (not short-circuited)
                    mock_generate_schema.assert_called_once()
                    # Verify get_schema was called to get current schema
                    mock_vespa_app.get_schema.assert_called_once()


if __name__ == '__main__':
    unittest.main()
