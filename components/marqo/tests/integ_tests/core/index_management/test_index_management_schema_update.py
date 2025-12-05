import os
import textwrap
from datetime import datetime, timedelta
from unittest.mock import patch

from marqo import version
from marqo.core.exceptions import IndexNotFoundError, InternalError, UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import FieldType
from marqo.core.models.marqo_index_request import FieldRequest
from tests.integ_tests.marqo_test import MarqoTestCase


class TestIndexManagementSchemaUpdate(MarqoTestCase):
    """Integration tests for the apply_latest_schema_template feature."""

    @classmethod
    def _add_validation_overrides(self, app_root_path: str):
        tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{tomorrow}'>indexing-change</allow>
                 <allow until='{tomorrow}'>field-type-change</allow>
                 <allow until='{tomorrow}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        with open(os.path.join(app_root_path, 'validation-overrides.xml'), 'w') as f:
            f.write(content)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Deploy initial app package with validation overrides
        app_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'initial_vespa_app')
        cls._add_validation_overrides(app_root_path)
        cls.vespa_client.deploy_application(app_root_path)
        cls.vespa_client.wait_for_application_convergence()

        # Bootstrap Vespa
        cls.index_management.bootstrap_vespa()

        cls.original_marqo_version = '2.23.0'
        cls.original_schema_template_version = '2.24.0'

        # pre-create all indexes for tests
        index_requests = [
            cls.unstructured_marqo_index_request(name='test_schema_update_index', marqo_version=cls.original_marqo_version, schema_template_version=cls.original_schema_template_version),
            cls.unstructured_marqo_index_request(name='test_config_change_restart', marqo_version=cls.original_marqo_version, schema_template_version=cls.original_schema_template_version),
            cls.unstructured_marqo_index_request(name='test_config_change_reindex', marqo_version=cls.original_marqo_version, schema_template_version=cls.original_schema_template_version),
            cls.unstructured_marqo_index_request(name='test_config_change_refeed', marqo_version=cls.original_marqo_version, schema_template_version=cls.original_schema_template_version),
            cls.unstructured_marqo_index_request(name='test_schema_template_version_current'),
            cls.unstructured_marqo_index_request(name='old_version_index', marqo_version='2.22.0'),
            cls.unstructured_marqo_index_request(name='legacy_unstructured_index', marqo_version='2.12.0'),
            cls.structured_marqo_index_request(
                name='structured_index',
                fields=[FieldRequest(name='title', type=FieldType.Text)],
                tensor_fields=['title']
            )
        ]
        cls.create_indexes(index_requests)

        # feed in a doc to add a 'title' lexical field to 'test_config_change_restart', keep the schema version unchanged
        with patch('marqo.version.get_version', return_value=cls.original_schema_template_version):
            cls.add_documents(cls.config, AddDocsParams(
                index_name='test_config_change_restart',
                docs=[{'_id': '1', 'title': 'hello'}],
                tensor_fields=[]
            ))

    def _get_validation_overrides(self) -> str:
        app = self.vespa_client.download_application()
        with open(os.path.join(app, 'validation-overrides.xml'), 'r') as f:
            return f.read()

    def _get_schema_from_vespa(self, schema_name: str) -> str:
        # Get the original schema for reference by downloading the app
        app = self.vespa_client.download_application()
        schema_path = os.path.join(app, 'schemas', f'{schema_name}.sd')
        with open(schema_path, 'r') as f:
            return f.read()

    # ============================================================================
    # Basic Flow Tests
    # ============================================================================

    def test_update_schema_no_changes(self):
        """When schema hasn't changed, should return schemaChanged=False and not deploy."""
        test_index_name = 'test_schema_update_index'
        result = self.index_management.apply_latest_schema_template(test_index_name)

        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)
        original_version = saved_index.version

        self.assertFalse(result['updated'])
        self.assertFalse(result['schemaChanged'])
        self.assertEqual('Schema is already up to date', result['reason'])
        self.assertEqual('No changes', result['schemaDiff'])
        self.assertEqual(original_schema, result['oldSchema'])
        self.assertEqual(original_schema, result['newSchema'])

        # Verify version and schema_template_version not updated (remains None when no changes)
        updated_index = self.index_management.get_index(test_index_name)
        self.assertEqual(self.original_schema_template_version, updated_index.schema_template_version)
        self.assertEqual(original_version, updated_index.version)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_successful(self, mock_generate_schema):
        """When schema changes with no actions required, should deploy successfully."""

        test_index_name = 'test_schema_update_index'
        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)
        original_version = saved_index.version

        # Generate a modified schema (add a comment to create a harmless change)
        modified_schema = original_schema.replace(
            'schema marqo__',
            '# Updated schema\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.apply_latest_schema_template(test_index_name)

        # Verify the result
        self.assertTrue(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertIn('Schema updated successfully', result['reason'])
        self.assertEqual(original_schema, result['oldSchema'])
        self.assertEqual(modified_schema, result['newSchema'])
        self.assertIn('# Updated schema', result['schemaDiff'])

        # Verify schema was actually deployed to Vespa
        self.assertEqual(modified_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version updated to current version; and index version becomes +1
        updated_index = self.index_management.get_index(test_index_name)
        self.assertEqual(version.get_version(), updated_index.schema_template_version)
        self.assertEqual(original_version + 1, updated_index.version)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_dry_run_prevents_deployment(self, mock_generate_schema):
        """When dry_run=True, should show changes but never deploy."""
        test_index_name = 'test_schema_update_index'
        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)

        modified_schema = original_schema.replace(
            'schema marqo__',
            '# Dry run test\nschema marqo__'
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.apply_latest_schema_template(
            test_index_name,
            dry_run=True
        )

        # Verify no deployment occurred
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertEqual('Dry run - no changes deployed', result['reason'])
        self.assertEqual(modified_schema, result['newSchema'])
        self.assertIn('# Dry run test', result['schemaDiff'])

        # Verify schema was NOT deployed to Vespa (original schema still present)
        self.assertEqual(original_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version not updated in dry run
        updated_index = self.index_management.get_index(test_index_name)
        self.assertEqual(self.original_schema_template_version, updated_index.schema_template_version)

    # ============================================================================
    # Error Case Tests
    # ============================================================================

    def test_update_schema_index_not_found(self):
        """Should raise IndexNotFoundError for non-existent index."""
        with self.assertRaisesStrict(IndexNotFoundError) as ctx:
            self.index_management.apply_latest_schema_template('nonexistent_index')

        self.assertIn('nonexistent_index', str(ctx.exception))

    def test_update_schema_wrong_index_type_structured(self):
        """Should raise InternalError for structured indexes (not supported)."""
        with self.assertRaisesStrict(InternalError) as ctx:
            self.index_management.apply_latest_schema_template('structured_index')

        self.assertIn('only semi-structured indexes support schema updates', str(ctx.exception))

    def test_update_schema_wrong_index_type_legacy_unstructured(self):
        """Should raise InternalError for legacy unstructured indexes (Marqo < 2.13.0)."""
        with self.assertRaisesStrict(InternalError) as ctx:
            self.index_management.apply_latest_schema_template('legacy_unstructured_index')

        self.assertIn('only semi-structured indexes support schema updates', str(ctx.exception))

    def test_update_schema_template_version_too_old(self):
        """Should raise UnsupportedFeatureError for indexes created with Marqo < 2.23.0."""
        with self.assertRaisesStrict(UnsupportedFeatureError) as ctx:
            self.index_management.apply_latest_schema_template('old_version_index')

        self.assertIn('Schema update is only supported for indexes created with Marqo 2.23.0 or later',
                      str(ctx.exception))
        self.assertIn('created with Marqo 2.22.0', str(ctx.exception))

    def test_shortcut_when_schema_template_version_current(self):
        """When schema_template_version matches current version, shortcut is triggered."""
        current_version = version.get_version()
        index_name = 'test_schema_template_version_current'

        result = self.index_management.apply_latest_schema_template(index_name)

        # Verify shortcut response
        self.assertFalse(result['updated'])
        self.assertFalse(result['schemaChanged'])
        self.assertIn(f'already at current Marqo version {current_version}', result['reason'])

        # Verify schema_template_version unchanged
        index = self.index_management.get_index(index_name)
        self.assertEqual(current_version, index.schema_template_version)

    # ============================================================================
    # configChangeActions Tests
    # ============================================================================

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_restart_actions(self, mock_generate_schema):
        """When restart actions are required and force=False, should block deployment.
        In this case, we will replace
        ```
        field marqo__lexical_title type string {
            indexing: index | summary
            index: enable-bm25
        }
        ```
        with
        ```
        field marqo__lexical_title type string {
            indexing: index | summary | attribute
            index: enable-bm25
        }
        ```
        Adding the attribute aspect will require a restart.
        See https://docs.vespa.ai/en/reference/schema-reference.html#changes-that-require-restart-but-not-re-feed for details
        """

        test_index_name = 'test_config_change_restart'
        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)

        modified_schema = original_schema.replace(
            'indexing: index | summary',
            'indexing: index | summary | attribute',
            1
        )
        mock_generate_schema.return_value = modified_schema

        # Verify it reject updates if not forced
        result = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('restart', result['configChangeActions'])
        self.assertEqual(1, len(result['configChangeActions']['restart']))
        self.assertIn("Field 'marqo__lexical_title' changed: add attribute aspect", result['configChangeActions']['restart'][0]['messages'][0])

        # Verify schema was NOT deployed
        self.assertEqual(original_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version not updated when blocked
        blocked_index = self.index_management.get_index(test_index_name)
        self.assertEqual(self.original_schema_template_version, blocked_index.schema_template_version)

        # Verify it updates the schema when forced set to true
        result_forced = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=True
        )
        self.assertTrue(result_forced['updated'])
        self.assertTrue(result_forced['schemaChanged'])
        self.assertIn('Update forced despite required actions', result_forced['reason'])
        self.assertEqual(modified_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version updated when forced
        forced_index = self.index_management.get_index(test_index_name)
        self.assertEqual(version.get_version(), forced_index.schema_template_version)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_refeed_actions(self, mock_generate_schema):
        """When re-feed actions are required and force=False, should block deployment.
        In this case, we will replace
        ```
        field marqo__id type string
        summary marqo__id type string
        ```
        with
        ```
        field marqo__id type int
        summary marqo__id type int
        ```
        Changing field type will require a re-feed.
        See https://docs.vespa.ai/en/reference/schema-reference.html#changes-that-require-re-feed for details
        """

        test_index_name = 'test_config_change_refeed'
        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)

        modified_schema = original_schema.replace(
            'marqo__id type string',
            'marqo__id type int',
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('refeed', result['configChangeActions'])
        self.assertIn("Field 'marqo__id' changed: data type: 'string' -> 'int'",
                      result['configChangeActions']['refeed'][0]['messages'][0])

        self.assertEqual(original_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version not updated when blocked
        blocked_index = self.index_management.get_index(test_index_name)
        self.assertEqual(self.original_schema_template_version, blocked_index.schema_template_version)

        # Force update
        result_forced = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=True
        )

        # Verify deployment proceeded despite actions
        self.assertTrue(result_forced['updated'])
        self.assertTrue(result_forced['schemaChanged'])
        self.assertIn('Update forced despite required actions', result_forced['reason'])

        self.assertEqual(modified_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version updated when forced
        forced_index = self.index_management.get_index(test_index_name)
        self.assertEqual(version.get_version(), forced_index.schema_template_version)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_schema_with_reindex_actions(self, mock_generate_schema):
        """When reindex actions are required and force=False, should block deployment.
        In this case, we will replace
        ```
        field marqo__id type string {
            indexing: attribute | summary
            attribute: fast-search
            rank: filter
        }
        ```
        with
        ```
        field marqo__id type string {
            indexing: attribute | summary | index
            attribute: fast-search
            rank: filter
        }
        ```
        Adding the index aspect to a string field will require a reindex.
        See https://docs.vespa.ai/en/reference/schema-reference.html#changes-that-require-reindexing for details
        """
        test_index_name = 'test_config_change_reindex'
        saved_index = self.index_management.get_index(test_index_name)
        original_schema = self._get_schema_from_vespa(saved_index.schema_name)

        modified_schema = original_schema.replace(
            'indexing: attribute | summary',
            'indexing: attribute | summary | index',
            1
        )
        mock_generate_schema.return_value = modified_schema

        result = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=False
        )

        # Verify deployment was blocked
        self.assertFalse(result['updated'])
        self.assertTrue(result['schemaChanged'])
        self.assertIn('Vespa requires manual actions before proceeding', result['reason'])
        self.assertIn('reindex', result['configChangeActions'])
        self.assertIn("Field 'marqo__id' changed: add index aspect",
                      result['configChangeActions']['reindex'][0]['messages'][0])

        self.assertEqual(original_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version not updated when blocked
        blocked_index = self.index_management.get_index(test_index_name)
        self.assertEqual(self.original_schema_template_version, blocked_index.schema_template_version)

        # Force update
        result_forced = self.index_management.apply_latest_schema_template(
            test_index_name,
            force=True
        )

        # Verify deployment is done
        self.assertTrue(result_forced['updated'])
        self.assertTrue(result_forced['schemaChanged'])
        self.assertIn('Update forced despite required actions', result_forced['reason'])
        self.assertIn('reindex', result_forced['configChangeActions'])
        self.assertEqual(modified_schema, self._get_schema_from_vespa(saved_index.schema_name))

        # Verify schema_template_version updated when forced
        forced_index = self.index_management.get_index(test_index_name)
        self.assertEqual(version.get_version(), forced_index.schema_template_version)
