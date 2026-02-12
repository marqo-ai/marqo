"""Unit tests for schema update functionality in VespaApplicationPackage and IndexManagement."""
import json
import unittest
from unittest.mock import Mock, MagicMock, patch, call

from marqo.core.exceptions import IndexNotFoundError, InternalError
from marqo.core.index_management.vespa_application_package import (
    VespaApplicationPackage,
    VespaApplicationStore,
    ApplicationPackageDeploymentSessionStore
)
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, StructuredMarqoIndex
from tests.unit_tests.marqo_test import MarqoTestCase


class TestVespaApplicationPackageSchemaUpdate(MarqoTestCase):
    """Test cases for schema update methods in VespaApplicationPackage."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = Mock(spec=VespaApplicationStore)

        # Mock basic file operations
        self.mock_store.read_text_file.side_effect = self._mock_read_text_file
        self.mock_store.file_exists.side_effect = self._mock_file_exists

        self.vespa_app = VespaApplicationPackage(self.mock_store)

    def _mock_file_exists(self, *paths):
        """Mock file_exists."""
        if paths == ('marqo_config.json',) or paths == ('services.xml',):
            return True
        if paths == ('schemas', 'test_schema.sd'):
            return True
        return False

    def _mock_read_text_file(self, *paths):
        """Mock read_text_file."""
        if paths == ('services.xml',):
            return '''<?xml version="1.0" encoding="utf-8"?>
<services xmlns:deploy="vespa" xmlns:preprocess="properties">
    <container><document-api/></container>
    <content><documents></documents></content>
</services>'''
        elif paths == ('marqo_config.json',):
            return '{"version": "2.23.0"}'
        elif paths == ('marqo_index_settings.json',):
            return '{}'
        elif paths == ('schemas', 'test_schema.sd'):
            return '''schema test_schema {
    document test_schema {
        field test_field type string {}
    }
}'''
        return None

    def test_get_schema_returns_existing_schema(self):
        """Test that get_schema returns schema content for existing schema."""
        result = self.vespa_app.get_schema('test_schema')

        self.assertIsNotNone(result)
        self.assertIn('schema test_schema', result)
        self.assertIn('field test_field', result)
        self.mock_store.read_text_file.assert_called_with('schemas', 'test_schema.sd')

    def test_get_schema_returns_none_when_not_exists(self):
        """Test that get_schema returns None for non-existent schema."""
        result = self.vespa_app.get_schema('nonexistent_schema')

        self.assertIsNone(result)
        self.mock_store.read_text_file.assert_called_with('schemas', 'nonexistent_schema.sd')


class TestApplicationPackageDeploymentSessionStore(MarqoTestCase):
    """Test cases for prepare_deployment and activate_deployment methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock()
        self.mock_vespa_client.create_deployment_session.return_value = (
            'http://content_url',
            'http://prepare_url'
        )
        self.mock_vespa_client.list_contents.return_value = []

        self.store = ApplicationPackageDeploymentSessionStore(
            vespa_client=self.mock_vespa_client,
            deploy_timeout=60,
            wait_for_convergence_timeout=120
        )

    def test_prepare_deployment_returns_response(self):
        """Test that prepare_deployment returns Vespa prepare response."""
        # Setup mock prepare response
        expected_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [],
                'refeed': [],
                'reindex': []
            }
        }
        self.mock_vespa_client.prepare.return_value = expected_response

        # Execute
        result = self.store.prepare_deployment()

        # Verify
        self.assertEqual(result, expected_response)
        self.mock_vespa_client.prepare.assert_called_once_with(
            'http://prepare_url',
            timeout=60
        )

    def test_activate_deployment_calls_activate_and_waits(self):
        """Test that activate_deployment activates and waits for convergence."""
        prepare_response = {
            'activate': 'http://activate_url'
        }

        # Execute
        self.store.activate_deployment(prepare_response)

        # Verify
        self.mock_vespa_client.activate.assert_called_once_with(
            'http://activate_url',
            timeout=60
        )
        self.mock_vespa_client.wait_for_application_convergence.assert_called_once_with(
            timeout=120
        )

    def test_deploy_application_prepares_and_activates(self):
        """Test that deploy_application calls prepare then activate."""
        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }
        self.mock_vespa_client.prepare.return_value = prepare_response

        # Execute
        self.store.deploy_application()

        # Verify prepare was called
        self.mock_vespa_client.prepare.assert_called_once()

        # Verify activate was called with the activate URL
        self.mock_vespa_client.activate.assert_called_once_with(
            'http://activate_url',
            timeout=60
        )

        # Verify convergence wait was called
        self.mock_vespa_client.wait_for_application_convergence.assert_called_once()


class TestVespaApplicationPackagePrepareOnly(MarqoTestCase):
    """Test cases for update_index_setting_and_schema with prepare_only mode."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock deployment session store
        self.mock_vespa_client = Mock()
        self.mock_vespa_client.create_deployment_session.return_value = (
            'http://content_url',
            'http://prepare_url'
        )
        self.mock_vespa_client.list_contents.return_value = []
        self.mock_vespa_client.get_text_content.return_value = '''<?xml version="1.0" encoding="utf-8"?>
<services><container><document-api/></container></services>'''

        self.mock_store = ApplicationPackageDeploymentSessionStore(
            vespa_client=self.mock_vespa_client,
            deploy_timeout=60,
            wait_for_convergence_timeout=120
        )

        # Mock store methods
        self.mock_store.file_exists = Mock(return_value=True)
        self.mock_store.read_text_file = Mock(side_effect=self._mock_read_text_file)
        self.mock_store.save_file = Mock()

        self.vespa_app = VespaApplicationPackage(self.mock_store)

        # Mock has_index to return True for test_index
        self.vespa_app.has_index = Mock(return_value=True)

        # Mock the index setting store methods to avoid version conflicts
        self.vespa_app._index_setting_store.save_index_setting = Mock()
        self.vespa_app._persist_index_settings = Mock()

    def _mock_read_text_file(self, *paths):
        """Mock read_text_file."""
        if paths == ('services.xml',):
            return '''<?xml version="1.0" encoding="utf-8"?>
<services xmlns:deploy="vespa" xmlns:preprocess="properties">
    <container>
        <document-api/>
    </container>
    <content>
        <documents></documents>
    </content>
</services>'''
        elif paths == ('marqo_config.json',):
            return '{"version": "2.23.0"}'
        elif paths == ('marqo_index_settings.json',):
            # Return empty dict to avoid index serialization issues during setup
            return '{}'
        return '{}'

    def test_update_index_setting_and_schema_prepare_only_returns_response(self):
        """Test that prepare_only=True returns prepare response without activating."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema",
            version=1
        )
        test_schema = "schema test_schema { document test_schema {} }"

        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {
                'restart': [{'name': 'restart', 'services': ['searchnode']}]
            }
        }
        self.mock_vespa_client.prepare.return_value = prepare_response

        # Execute
        result = self.vespa_app.update_index_setting_and_schema(
            test_index,
            test_schema,
            prepare_only=True
        )

        # Verify
        self.assertEqual(result, prepare_response)
        self.mock_vespa_client.prepare.assert_called_once()
        self.mock_vespa_client.activate.assert_not_called()

    def test_update_index_setting_and_schema_activates_when_not_prepare_only(self):
        """Test that prepare_only=False activates deployment."""
        # Setup
        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema",
            version=1
        )
        test_schema = "schema test_schema { document test_schema {} }"

        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }
        self.mock_vespa_client.prepare.return_value = prepare_response

        # Execute
        result = self.vespa_app.update_index_setting_and_schema(
            test_index,
            test_schema,
            prepare_only=False
        )

        # Verify
        self.assertIsNone(result)
        self.mock_vespa_client.prepare.assert_called_once()
        self.mock_vespa_client.activate.assert_called_once()
        self.mock_vespa_client.wait_for_application_convergence.assert_called_once()

    @patch('marqo.core.index_management.vespa_application_package.time')
    def test_update_index_setting_and_schema_sets_updated_at(self, mock_time):
        """Test that update_index_setting_and_schema sets updated_at to current time."""
        mock_time.time.return_value = 1700000000.0

        test_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema",
            version=1,
            updated_at=1000  # old value
        )
        test_schema = "schema test_schema { document test_schema {} }"

        prepare_response = {
            'activate': 'http://activate_url',
            'configChangeActions': {}
        }
        self.mock_vespa_client.prepare.return_value = prepare_response

        self.vespa_app.update_index_setting_and_schema(test_index, test_schema)

        saved_index = self.vespa_app._index_setting_store.save_index_setting.call_args[0][0]
        self.assertEqual(saved_index.updated_at, 1700000000)


if __name__ == '__main__':
    unittest.main()
