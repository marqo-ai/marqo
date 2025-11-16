import os
from unittest.mock import patch

from marqo.core.exceptions import IndexNotFoundError, InternalError, UnsupportedFeatureError
from marqo.core.models.marqo_index import Model
from tests.integ_tests.marqo_test import MarqoTestCase


class TestIndexManagementSchemaUpdate(MarqoTestCase):
    """Integration tests for the update_index_main_schema feature."""

    def setUp(self):
        super().setUp()
        # Bootstrap Vespa
        self.index_management.bootstrap_vespa()

        # Create a semi-structured index (Marqo >= 2.13.0) for testing
        self.test_index_name = "test_schema_update_index"

        # Delete the index if it already exists from previous test
        try:
            self.index_management.delete_index_by_name(self.test_index_name)
        except IndexNotFoundError:
            pass

        self.request = self.unstructured_marqo_index_request(
            name=self.test_index_name,
            model=Model(name='hf/e5-small')
        )
        created_index = self.index_management.create_index(self.request)

        # Track index for cleanup in tearDownClass
        if created_index not in self.indexes:
            self.indexes.append(created_index)

        # Get the created index
        self.test_index = self.index_management.get_index(self.test_index_name)

        # Get the original schema for reference by downloading the app
        app = self.vespa_client.download_application()
        schema_path = os.path.join(app, 'schemas', f'{self.test_index.schema_name}.sd')
        with open(schema_path, 'r') as f:
            self.original_schema = f.read()

    # ============================================================================
    # Basic Flow Tests
    # ============================================================================

    def test_update_schema_no_changes(self):
        """When schema hasn't changed, should return schema_changed=False and not deploy."""
        result = self.index_management.update_index_main_schema(self.test_index_name)

        self.assertFalse(result['updated'])
        self.assertFalse(result['schema_changed'])
        self.assertEqual('Schema is already up to date', result['reason'])
        self.assertEqual('No changes', result['schema_diff'])
        self.assertEqual(self.original_schema, result['old_schema'])
        self.assertEqual(self.original_schema, result['new_schema'])

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_successful(self, mock_generate_schema):
        """When schema changes with no actions required, should deploy successfully."""
        # Generate a modified schema (add a comment to create a harmless change)
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Updated schema\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.update_index_main_schema(self.test_index_name)

        # Verify the result
        self.assertTrue(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Schema updated successfully', result['reason'])
        self.assertEqual(self.original_schema, result['old_schema'])
        self.assertEqual(modified_schema, result['new_schema'])
        self.assertIn('# Updated schema', result['schema_diff'])

        # Verify schema was actually deployed to Vespa
        app = self.vespa_client.download_application()
        schema_path = os.path.join(app, 'schemas', f'{self.test_index.schema_name}.sd')
        with open(schema_path, 'r') as f:
            deployed_schema = f.read()
        self.assertEqual(modified_schema, deployed_schema)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_dry_run_prevents_deployment(self, mock_generate_schema):
        """When dry_run=True, should show changes but never deploy."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Dry run test\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            dry_run=True
        )

        # Verify no deployment occurred
        self.assertFalse(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertEqual('Dry run - no changes deployed', result['reason'])
        self.assertEqual(modified_schema, result['new_schema'])
        self.assertIn('# Dry run test', result['schema_diff'])

        # Verify schema was NOT deployed to Vespa (original schema still present)
        app = self.vespa_client.download_application()
        schema_path = os.path.join(app, 'schemas', f'{self.test_index.schema_name}.sd')
        with open(schema_path, 'r') as f:
            deployed_schema = f.read()
        self.assertEqual(self.original_schema, deployed_schema)
        self.assertNotIn('# Dry run test', deployed_schema)

    # ============================================================================
    # configChangeActions Tests
    # ============================================================================

    @patch('marqo.vespa.vespa_client.VespaClient.prepare')
    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_restart_actions_blocked(self, mock_generate_schema, mock_prepare):
        """When restart actions are required and force=False, should block deployment."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Change requiring restart\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        # Mock Vespa prepare response with restart actions
        mock_prepare.return_value = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [{
                    'clusterName': 'marqo_content',
                    'clusterType': 'search',
                    'serviceType': 'searchnode',
                    'messages': ['Change requires service restart'],
                    'services': [{'serviceName': 'searchnode', 'serviceType': 'searchnode'}]
                }]
            }
        }

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('restart', result['config_change_actions'])
        self.assertEqual(1, len(result['config_change_actions']['restart']))

        # Verify schema was NOT deployed
        app = self.vespa_client.download_application()
        schema_path = os.path.join(app, 'schemas', f'{self.test_index.schema_name}.sd')
        with open(schema_path, 'r') as f:
            deployed_schema = f.read()
        self.assertEqual(self.original_schema, deployed_schema)

    @patch('marqo.vespa.vespa_client.VespaClient.activate')
    @patch('marqo.vespa.vespa_client.VespaClient.prepare')
    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_restart_actions_forced(self, mock_generate_schema, mock_prepare, mock_activate):
        """When restart actions are required and force=True, should deploy anyway."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Forced change with restart\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        # Mock Vespa prepare response with restart actions
        mock_prepare.return_value = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [{
                    'clusterName': 'marqo_content',
                    'clusterType': 'search',
                    'serviceType': 'searchnode',
                    'messages': ['Change requires service restart'],
                    'services': [{'serviceName': 'searchnode', 'serviceType': 'searchnode'}]
                }]
            }
        }

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            force=True
        )

        # Verify deployment proceeded despite actions
        self.assertTrue(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Update forced despite required actions', result['reason'])
        self.assertIn('warning', result)
        self.assertIn('restart', result['config_change_actions'])

        # Verify activate was called
        mock_activate.assert_called_once()

    @patch('marqo.vespa.vespa_client.VespaClient.prepare')
    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_refeed_actions_blocked(self, mock_generate_schema, mock_prepare):
        """When refeed actions are required and force=False, should block deployment."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Change requiring refeed\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        # Mock Vespa prepare response with refeed actions
        mock_prepare.return_value = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'refeed': [{
                    'name': 'refeed',
                    'documentType': self.test_index.schema_name,
                    'clusterName': 'marqo_content',
                    'messages': ['Field type change requires re-feeding']
                }]
            }
        }

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('refeed', result['config_change_actions'])

    @patch('marqo.vespa.vespa_client.VespaClient.activate')
    @patch('marqo.vespa.vespa_client.VespaClient.prepare')
    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_refeed_actions_forced(self, mock_generate_schema, mock_prepare, mock_activate):
        """When refeed actions are required and force=True, should deploy anyway."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Forced refeed change\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        # Mock Vespa prepare response with refeed actions
        mock_prepare.return_value = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'refeed': [{
                    'name': 'refeed',
                    'documentType': self.test_index.schema_name,
                    'clusterName': 'marqo_content',
                    'messages': ['Field type change requires re-feeding']
                }]
            }
        }

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            force=True
        )

        # Verify deployment proceeded despite actions
        self.assertTrue(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Update forced despite required actions', result['reason'])
        self.assertIn('refeed', result['config_change_actions'])

        # Verify activate was called
        mock_activate.assert_called_once()

    @patch('marqo.vespa.vespa_client.VespaClient.prepare')
    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_reindex_actions_blocked(self, mock_generate_schema, mock_prepare):
        """When reindex actions are required and force=False, should block deployment."""
        modified_schema = self.original_schema.replace(
            'schema marqo__',
            '# Change requiring reindex\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        # Mock Vespa prepare response with reindex actions
        mock_prepare.return_value = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'reindex': [{
                    'name': 'reindex',
                    'documentType': self.test_index.schema_name,
                    'messages': ['Indexing script change requires reindexing']
                }]
            }
        }

        result = self.index_management.update_index_main_schema(
            self.test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schema_changed'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('reindex', result['config_change_actions'])

    # ============================================================================
    # Error Case Tests
    # ============================================================================

    def test_update_schema_index_not_found(self):
        """Should raise IndexNotFoundError for non-existent index."""
        with self.assertRaisesStrict(IndexNotFoundError) as ctx:
            self.index_management.update_index_main_schema('nonexistent_index')

        self.assertIn('nonexistent_index', str(ctx.exception))

    def test_update_schema_wrong_index_type_structured(self):
        """Should raise InternalError for structured indexes (not supported)."""
        from marqo.core.models.marqo_index_request import FieldRequest
        from marqo.core.models.marqo_index import FieldType

        # Create a structured index
        structured_request = self.structured_marqo_index_request(
            name='structured_index',
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        self.index_management.create_index(structured_request)

        try:
            with self.assertRaisesStrict(InternalError) as ctx:
                self.index_management.update_index_main_schema('structured_index')

            self.assertIn('only semi-structured indexes support schema updates', str(ctx.exception))
        finally:
            self.index_management.delete_index_by_name('structured_index')

    def test_update_schema_wrong_index_type_legacy_unstructured(self):
        """Should raise InternalError for legacy unstructured indexes (Marqo < 2.13.0)."""
        # Create a legacy unstructured index (marqo_version < 2.13.0)
        legacy_request = self.unstructured_marqo_index_request(
            name='legacy_index',
            marqo_version='2.12.0',
            model=Model(name='hf/e5-small')
        )
        self.index_management.create_index(legacy_request)

        try:
            with self.assertRaisesStrict(InternalError) as ctx:
                self.index_management.update_index_main_schema('legacy_index')

            self.assertIn('only semi-structured indexes support schema updates', str(ctx.exception))
        finally:
            self.index_management.delete_index_by_name('legacy_index')

    def test_update_schema_version_too_old(self):
        """Should raise UnsupportedFeatureError for indexes created with Marqo < 2.23.0."""
        # Create an index with Marqo 2.22.0 (before schema update feature)
        old_version_request = self.unstructured_marqo_index_request(
            name='old_version_index',
            marqo_version='2.22.0',
            model=Model(name='hf/e5-small')
        )
        self.index_management.create_index(old_version_request)

        try:
            with self.assertRaisesStrict(UnsupportedFeatureError) as ctx:
                self.index_management.update_index_main_schema('old_version_index')

            self.assertIn('Schema update is only supported for indexes created with Marqo 2.23.0 or later', str(ctx.exception))
            self.assertIn('created with Marqo 2.22.0', str(ctx.exception))
        finally:
            self.index_management.delete_index_by_name('old_version_index')
