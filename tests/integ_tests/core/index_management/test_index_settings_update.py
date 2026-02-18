import os
import time
import unittest

from marqo import version
from marqo.core.exceptions import IndexNotFoundError, InvalidModelPropertiesError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import Model
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from tests.integ_tests.marqo_test import MarqoTestCase


class TestIndexSettingsUpdate(MarqoTestCase):
    """Integration tests for the update_index_settings feature.

    These tests require Vespa to be running locally.
    """

    def setUp(self):
        super().setUp()
        self.index_management = IndexManagement(
            self.vespa_client,
            zookeeper_client=self.zookeeper_client,
            enable_index_operations=True,
            deployment_timeout_seconds=30,
            convergence_timeout_seconds=120
        )
        self._test_dir = str(os.path.dirname(os.path.abspath(__file__)))
        # Bootstrap vespa to ensure it's ready
        self.index_management.bootstrap_vespa()

    def _create_semi_structured_index(self, index_name: str, model: Model) -> None:
        """Helper to create a semi-structured index."""
        request = UnstructuredMarqoIndexRequest(
            name=index_name,
            model=model,
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=100,
        )
        self.index_management.create_index(request)

    def test_update_model_properties_e2e(self):
        """Test updating model properties on an existing index end-to-end."""
        index_name = f"test_update_props_{int(time.time())}"

        try:
            # Create index with a custom model
            self._create_semi_structured_index(
                index_name,
                model=Model(
                    name='my-custom-model',
                    properties={
                        "dimensions": 384,
                        "type": "open_clip",
                        "name": "ViT-B-16",
                        "url": "https://old-url.com/model.pt",
                    },
                    custom=True
                )
            )

            # Verify index exists
            index = self.index_management.get_index(index_name)
            self.assertEqual(index.model.properties["url"], "https://old-url.com/model.pt")

            # Update model properties
            new_properties = {
                "dimensions": 384,
                "type": "open_clip",
                "name": "ViT-B-16",
                "url": "https://new-url.com/model.pt",
            }
            self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": new_properties}
            )

            # Verify update
            updated_index = self.index_management.get_index(index_name)
            self.assertEqual(updated_index.model.properties["url"], "https://new-url.com/model.pt")
            self.assertTrue(updated_index.model.custom)
            # Version should have incremented
            self.assertGreater(updated_index.version, index.version or 0)
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_update_non_existent_index_raises_error(self):
        """Test that updating a non-existent index raises IndexNotFoundError."""
        with self.assertRaises(IndexNotFoundError):
            self.index_management.update_index_settings_by_settings_dict(
                "non_existent_index",
                {"modelProperties": {"dimensions": 384, "type": "hf"}}
            )

    def test_update_dimension_change_raises_error(self):
        """Test that changing dimensions raises InvalidModelPropertiesError."""
        index_name = f"test_dim_change_{int(time.time())}"

        try:
            self._create_semi_structured_index(
                index_name,
                model=Model(
                    name='my-custom-model',
                    properties={
                        "dimensions": 384,
                        "type": "open_clip",
                        "url": "https://example.com/model.pt",
                    },
                    custom=True
                )
            )

            with self.assertRaises(InvalidModelPropertiesError):
                self.index_management.update_index_settings_by_settings_dict(
                    index_name,
                    {"modelProperties": {
                        "dimensions": 768,  # Changed!
                        "type": "open_clip",
                        "url": "https://example.com/model.pt",
                    }}
                )
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass
