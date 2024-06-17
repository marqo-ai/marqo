from unittest.mock import patch

from marqo.core.exceptions import IndexNotFoundError
from tests.marqo_test import MarqoTestCase

from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams


class TestCreateIndexResilience(MarqoTestCase):
    """A test to check the resilience of the create index operation.

    Index creation in Marqo is not atomic. We need to ensure that the create index operation is resilient to failures.
    Users should at least be able to bring Marqo to a consistent state by repeating the call to create index and get a
    200 response code.

    The index creation operation consists of the following steps:
    1. Download the application package.
    2. Check the configuration of the index from the marqo__config index.
    3. Check the existence of the index in the schema.
    3.5 Optional: Add the marqo__config index if it does not exist.
    4. Generate the schema.
    5. Add new schema .sd file
    6. Add the new schema to the service
    7. Deploy the new application package
    8. Add the index settings to marqo__settings index
    9. Add the Marqo version to the marqo__config index
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.index_management.bootstrap_vespa()

    def setUp(self) -> None:
        self.index_name = "test_index_creation_resilience"
        try:
            # Clear up the index before test
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass
        self.index_request = self.unstructured_marqo_index_request(name=self.index_name)

    def tearDown(self) -> None:
        try:
            # Clear up the index after test
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass

    def check_resilience(self):
        # Ensure the index can be created again by repeating the create index operation
        self.index_management.create_index(self.index_request)

        # Add documents to the index
        r = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_name,
                tensor_fields=[],
                docs=[{"test": "test", "_id": "1"}]
            )
        )
        self.assertEqual(r["errors"], False)

        # Test search
        r = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="test",
            search_method="LEXICAL"
        )
        self.assertEqual(1, len(r["hits"]))

        # Ensure the index can be deleted successfully
        tensor_search.delete_index(self.config, self.index_name)

    def test_downloadApplicationPackageFail(self):
        """Test that the create index operation is resilient to failures in downloading the application package."""
        error = Exception("Failed to download the application package")
        with patch("marqo.vespa.vespa_client.VespaClient.download_application", side_effect=error) \
                as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_checkIndexConfigurationFail(self):
        """Test that the create index operation is resilient to failures in checking the configuration of the index."""
        error = Exception("Failed to check the configuration of the index")
        with patch("marqo.core.index_management.index_management.IndexManagement._marqo_config_exists",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_checkIndexExistsFail(self):
        """Test that the create index operation is resilient to failures in checking the existence of the index in the
        schema.
        """
        error = Exception("Failed to check the existence of the index in the schema")
        with patch("marqo.core.index_management.index_management.IndexManagement.index_exists",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_addMarqoConfigIndexFail(self):
        """Test that the create index operation is resilient to failures in adding the marqo__config index."""
        error = Exception("Failed to add the marqo__config index")
        with patch("marqo.core.index_management.index_management.IndexManagement._marqo_config_exists",
                   return_value=False):
            with patch("marqo.core.index_management.index_management.IndexManagement._add_marqo_config",
                       side_effect=error) as mock_create_index_by_name:
                with self.assertRaises(Exception) as e:
                    self.index_management.create_index(self.index_request)
                self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_generateSchemaFail(self):
        """Test that the create index operation is resilient to failures in generating the schema."""
        error = Exception("Failed to generate the schema")
        with patch("marqo.core.unstructured_vespa_index.unstructured_vespa_schema.UnstructuredVespaSchema."
                   "generate_schema", side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_addNewSchemaSdFileFail(self):
        """Test that the create index operation is resilient to failures in add new schema .sd file."""
        error = Exception("Failed to add the new schema .sd file")
        with patch("marqo.core.index_management.index_management.IndexManagement._add_schema",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_addNewSchemaToServiceFail(self):
        """Test that the create index operation is resilient to failures in adding the new schema to the service."""
        error = Exception("Failed to add the new schema to the service")
        with patch("marqo.core.index_management.index_management.IndexManagement._add_schema_to_services",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_deployApplicationPackageFail(self):
        """Test that the create index operation is resilient to failures in deploying the new application package."""
        error = Exception("Failed to deploy the new application package")
        with patch("marqo.vespa.vespa_client.VespaClient.deploy_application",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_addTheIndexSettingsToMarqoSettingsIndexFail(self):
        """Test that the create index operation is resilient to failures in adding the index settings to the
        marqo__settings index.
        """
        error = Exception("Failed to add the index settings to the marqo__settings index")
        with patch("marqo.core.index_management.index_management.IndexManagement._save_index_settings",
                   side_effect=error) as mock_create_index_by_name:
            with self.assertRaises(Exception) as e:
                self.index_management.create_index(self.index_request)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_saveMarqoVersionFail(self):
        """Test that the create index operation is resilient to failures in saving the Marqo version."""
        error = Exception("Failed to save the Marqo version")
        with patch("marqo.core.index_management.index_management.IndexManagement._marqo_config_exists",
                   return_value=False):
            with patch("marqo.core.index_management.index_management.IndexManagement._save_marqo_version",
                       side_effect=error) as mock_create_index_by_name:
                with self.assertRaises(Exception) as e:
                    self.index_management.create_index(self.index_request)
                self.assertEqual(e.exception, error)

        with patch("marqo.core.index_management.index_management.IndexManagement._marqo_config_exists",
                   return_value=False):
            self.check_resilience()