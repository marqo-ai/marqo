from unittest.mock import patch

from marqo.core.exceptions import IndexNotFoundError
from tests.marqo_test import MarqoTestCase

from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams


class TestDeleteIndexResilience(MarqoTestCase):
    """ This is a test class for testing the resilience of the delete index operation.

    Index deletion is not atomic in Marqo. We need to ensure that the delete index operation is resilient to failures.
    Users should at least be able to bring Marqo to a consistent state by repeating the call to delete index and get
    a 200 response code.

    I divide the delete index operation into the following steps:
    1. Fetch the index object by name from the marqo__settings schema. Raise 400 index_not_found error if the index
    does not exist.
    2. Download the application package.
    3. Remove the index in the schema. Raise os error (Internal Error 500) if the schema file path does not exist
    4. Deploy the new application package
    5. Delete the document in the Vespa marqo__settings schema
    6. Delete the index cache

    We will simulate failures in each of these steps and ensure that the delete index operation is resilient to these.
    """

    def setUp(self) -> None:
        self.index_name = "test_index_deletion"
        test_index_request = self.unstructured_marqo_index_request(
            name=self.index_name
        )
        self.index_management.create_index(test_index_request)

    def tearDown(self) -> None:
        try:
            # Clear up the index after test
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass

    def check_resilience(self):
        try:
            # Repeat the delete index operation, it is OK to raise IndexNotFoundError
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass
        self.create_indexes([self.unstructured_marqo_index_request(name=self.index_name)])

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

    def test_fetchIndexObjectFail(self):
        """Test that the delete index operation is resilient to failures in fetching the index object by name from the
        marqo__settings schema.
        """

        error = Exception("Failed to fetch index object")
        with patch("marqo.core.index_management.index_management.IndexManagement.get_index", side_effect=error) \
                as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_downloadApplicationPackageFail(self):
        """Test that the delete index operation is resilient to failures in downloading the application package."""
        error = Exception("Failed to download the application package")
        with patch("marqo.vespa.vespa_client.VespaClient.download_application", side_effect=error) \
                as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_removeIndexInSchemaFail(self):
        """Test that the delete index operation is resilient to failures in removing the index in the schema."""
        error = Exception("Failed to remove the index in the schema")
        with patch("marqo.core.index_management.index_management.IndexManagement._remove_schema_from_services",
                   side_effect=error) as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_deployApplicationPackage(self):
        """Test that the delete index operation is resilient to failures in deploying the new application package."""
        error = Exception("Failed to remove the index in the schema")
        with patch("marqo.vespa.vespa_client.VespaClient.deploy_application",
                   side_effect=error) as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_deleteDocumentInVespaSchemaFail(self):
        """Test that the delete index operation is resilient to failures in deleting the document in the Vespa
        marqo__settings schema.
        """
        error = Exception("Failed to delete the document in the Vespa marqo__settings schema")
        with patch("marqo.core.index_management.index_management.IndexManagement._delete_index_settings_by_name",
                   side_effect=error) \
                as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()

    def test_deleteCacheFail(self):
        """Test that the delete index operation is resilient to failures in deleting index_name cache."""
        error = Exception("Failed to get cache")
        with patch("marqo.tensor_search.tensor_search.get_cache", side_effect=error) \
                as mock_delete_index_by_name:
            with self.assertRaises(Exception) as e:
                tensor_search.delete_index(self.config, self.index_name)
            self.assertEqual(e.exception, error)

        self.check_resilience()
