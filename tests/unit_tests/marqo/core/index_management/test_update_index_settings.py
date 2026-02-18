import unittest
from unittest.mock import Mock, MagicMock, patch

import marqo.version
from marqo.core.exceptions import IndexNotFoundError, InternalError, InvalidModelPropertiesError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import Model
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase


class TestUpdateIndexSettings(MarqoTestCase):
    def setUp(self):
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_zookeeper_client = Mock()
        self.index_management = IndexManagement(
            vespa_client=self.mock_vespa_client,
            zookeeper_client=self.mock_zookeeper_client,
            enable_index_operations=True
        )

    def _setup_mocks(self, existing_index):
        """Helper to set up common mocks."""
        mock_vespa_app = Mock()
        mock_deployment_lock = MagicMock()
        self.index_management.get_index = Mock(return_value=existing_index)
        self.index_management._get_vespa_application = Mock(return_value=mock_vespa_app)
        self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)
        return mock_vespa_app, mock_deployment_lock

    def test_update_model_properties_semi_structured_non_custom(self):
        """Test successful update for semi-structured index with non-custom model."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(name='hf/e5-small'),
            version=1
        )
        # Non-custom model: properties come from registry
        existing_index.model.properties = {
            "dimensions": 384,
            "type": "hf",
            "name": "hf/e5-small",
        }
        existing_index.model.custom = False

        mock_vespa_app, _ = self._setup_mocks(existing_index)

        new_properties = {
            "dimensions": 384,
            "type": "hf",
            "name": "hf/e5-small",
            "url": "https://new-url.com/model.pt",
        }

        self.index_management.update_index_settings_by_settings_dict(
            "test_index", {"modelProperties": new_properties}
        )

        mock_vespa_app.update_index_setting.assert_called_once()
        updated_index = mock_vespa_app.update_index_setting.call_args[0][0]
        self.assertEqual(updated_index.model.properties, new_properties)
        self.assertTrue(updated_index.model.custom)

    def test_update_model_properties_semi_structured_custom(self):
        """Test successful update for semi-structured index with custom model."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(
                name='my-custom-model',
                properties={
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                },
                custom=True
            ),
            version=1
        )

        mock_vespa_app, _ = self._setup_mocks(existing_index)

        new_properties = {
            "dimensions": 768,
            "type": "open_clip",
            "name": "ViT-B-16",
            "url": "https://new-url.com/model.pt",
        }

        self.index_management.update_index_settings_by_settings_dict(
            "test_index", {"modelProperties": new_properties}
        )

        mock_vespa_app.update_index_setting.assert_called_once()
        updated_index = mock_vespa_app.update_index_setting.call_args[0][0]
        self.assertEqual(updated_index.model.properties, new_properties)
        self.assertTrue(updated_index.model.custom)

    def test_update_model_properties_structured_index(self):
        """Test successful update for structured index."""
        existing_index = self.structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            model=Model(
                name='my-custom-model',
                properties={
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                },
                custom=True
            ),
        )

        mock_vespa_app, _ = self._setup_mocks(existing_index)

        new_properties = {
            "dimensions": 768,
            "type": "open_clip",
            "name": "ViT-B-16",
            "url": "https://new-url.com/model.pt",
        }

        self.index_management.update_index_settings_by_settings_dict(
            "test_index", {"modelProperties": new_properties}
        )

        mock_vespa_app.update_index_setting.assert_called_once()
        updated_index = mock_vespa_app.update_index_setting.call_args[0][0]
        self.assertEqual(updated_index.model.properties, new_properties)
        self.assertTrue(updated_index.model.custom)

    def test_non_existent_index_raises_index_not_found(self):
        """Test that updating a non-existent index raises IndexNotFoundError."""
        mock_deployment_lock = MagicMock()
        self.index_management.get_index = Mock(
            side_effect=IndexNotFoundError("Index test_index not found")
        )
        self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)

        with self.assertRaises(IndexNotFoundError):
            self.index_management.update_index_settings_by_settings_dict(
                "test_index", {"modelProperties": {"dimensions": 384, "type": "hf"}}
            )

    def test_dimension_change_raises_invalid_model_properties(self):
        """Test that changing dimensions raises InvalidModelPropertiesError."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(
                name='my-custom-model',
                properties={
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                },
                custom=True
            ),
            version=1
        )

        self._setup_mocks(existing_index)

        with self.assertRaises(InvalidModelPropertiesError) as ctx:
            self.index_management.update_index_settings_by_settings_dict(
                "test_index",
                {"modelProperties": {
                    "dimensions": 384,  # changed!
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://new-url.com/model.pt",
                }}
            )
        self.assertIn("dimensions", str(ctx.exception))

    def test_type_change_raises_invalid_model_properties(self):
        """Test that changing type raises InvalidModelPropertiesError."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(
                name='my-custom-model',
                properties={
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                },
                custom=True
            ),
            version=1
        )

        self._setup_mocks(existing_index)

        with self.assertRaises(InvalidModelPropertiesError) as ctx:
            self.index_management.update_index_settings_by_settings_dict(
                "test_index",
                {"modelProperties": {
                    "dimensions": 768,
                    "type": "hf",  # changed!
                    "name": "ViT-B-16",
                    "url": "https://new-url.com/model.pt",
                }}
            )
        self.assertIn("type", str(ctx.exception))

    def test_key_removal_raises_invalid_model_properties(self):
        """Test that removing keys raises InvalidModelPropertiesError."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(
                name='my-custom-model',
                properties={
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                },
                custom=True
            ),
            version=1
        )

        self._setup_mocks(existing_index)

        with self.assertRaises(InvalidModelPropertiesError) as ctx:
            self.index_management.update_index_settings_by_settings_dict(
                "test_index",
                {"modelProperties": {
                    "dimensions": 768,
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    # "url" key removed!
                }}
            )
        self.assertIn("remove", str(ctx.exception).lower())

    def test_disallowed_settings_key_raises_internal_error(self):
        """Test that disallowed settings keys raise InternalError."""
        with self.assertRaises(InternalError) as ctx:
            self.index_management.update_index_settings_by_settings_dict(
                "test_index",
                {"modelProperties": {"dimensions": 384}, "someOtherSetting": "value"}
            )
        self.assertIn("someOtherSetting", str(ctx.exception))

    def test_validate_updated_model_properties_success(self):
        """Test that valid updates pass validation."""
        current = {"dimensions": 768, "type": "open_clip", "url": "https://old.com"}
        updated = {"dimensions": 768, "type": "open_clip", "url": "https://new.com", "extra_key": "value"}

        # Should not raise
        IndexManagement.validate_updated_model_properties(current, updated)

    def test_updated_index_sets_custom_true(self):
        """Test that _updated_index_with_model_properties sets custom=True."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(name='hf/e5-small'),
            version=1
        )
        existing_index.model.custom = False

        new_properties = {"dimensions": 384, "type": "hf"}
        updated = self.index_management._updated_index_with_model_properties(
            existing_index, new_properties
        )

        self.assertTrue(updated.model.custom)
        self.assertEqual(updated.model.properties, new_properties)
        # Original should be unchanged
        self.assertFalse(existing_index.model.custom)
