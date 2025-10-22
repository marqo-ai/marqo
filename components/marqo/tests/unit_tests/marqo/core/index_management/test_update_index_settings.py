import unittest
from unittest.mock import Mock, patch, MagicMock

import marqo.version
import marqo.version
from marqo.core.exceptions import InternalError, IndexNotFoundError, InvalidModelPropertiesError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import Model, ImagePreProcessing
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import (
    SemiStructuredMarqoIndex, StructuredMarqoIndex, Field, FieldType, ImagePreProcessing, DistanceMetric,
    HnswConfig, TextPreProcessing
)


class TestUpdateIndexSettings(MarqoTestCase):
    """
    A class to test the update_index_settings functionality in IndexManagement.
    """

    def setUp(self):
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_zookeeper_client = Mock()
        self.index_management = IndexManagement(
            vespa_client=self.mock_vespa_client,
            zookeeper_client=self.mock_zookeeper_client,
            enable_index_operations=True
        )

    def _create_semistructured_index_with_model_properties(self, model: Model) -> SemiStructuredMarqoIndex:
        return SemiStructuredMarqoIndex(
            name="test_index",
            schema_name="test_schema",
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=100, split_overlap=0, split_method="word"),
            image_preprocessing=ImagePreProcessing(),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            vector_numeric_type="float",
            model=model,
            lexical_fields=[Field(name="test_field", type=FieldType.Text)],
            tensor_fields=[],
            marqo_version="2.24.0",
            created_at=1,
            updated_at=2,
            treat_urls_and_pointers_as_images=True,
            filter_string_max_length=1000
        )

    def _create_structured_index_with_model_properties(self, model: Model) -> StructuredMarqoIndex:
        return StructuredMarqoIndex(
            name="test_index",
            schema_name="test_schema",
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=100, split_overlap=0, split_method="word"),
            image_preprocessing=ImagePreProcessing(),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            vector_numeric_type="float",
            model=model,
            fields=[Field(name="test_field", type=FieldType.Text)],
            lexical_fields=[Field(name="test_field", type=FieldType.Text)],
            tensor_fields=[],
            marqo_version="2.24.0",
            created_at=1,
            updated_at=2,
            treat_urls_and_pointers_as_images=True,
            filter_string_max_length=1000
        )

    def test_update_semistructured_marqo_index_custom_model_properties(self):
        """Test updating model properties for a semi-structured index with a custom model."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch(
                    "marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }
            mock_get_index.return_value = original_index
            self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={
                    "modelProperties": updated_model_properties
                }
            )
            mock_app.update_index_setting.assert_called_once()
            updated_index_settings = mock_app.update_index_setting.call_args[0][0]

            original_index_settings_dict = original_index.dict()
            updated_index_settings_dict = updated_index_settings.dict()
            # Model properties should be updated
            self.assertEqual(updated_model_properties, updated_index_settings_dict["model"]["properties"])
            # Everything else should remain the same
            original_index_settings_dict["model"].pop("properties")
            updated_index_settings_dict["model"].pop("properties")
            self.assertEqual(original_index_settings_dict, updated_index_settings_dict)

    def test_update_semistructured_marqo_index_non_custom_model_properties(self):
        """Test updating model properties when the original model is not custom."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch(
                    "marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=False)
            )

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }
            mock_get_index.return_value = original_index
            self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={
                    "modelProperties": updated_model_properties
                }
            )
            mock_app.update_index_setting.assert_called_once()
            updated_index_settings = mock_app.update_index_setting.call_args[0][0]

            original_index_settings_dict = original_index.dict()
            updated_index_settings_dict = updated_index_settings.dict()

            # Model properties should be updated and custom should be set to True
            self.assertEqual(updated_model_properties, updated_index_settings_dict["model"]["properties"])
            self.assertEqual(True, updated_index_settings_dict["model"]["custom"])
            # Everything else should remain the same
            original_index_settings_dict["model"].pop("custom")
            updated_index_settings_dict["model"].pop("properties")
            updated_index_settings_dict["model"].pop("custom")
            self.assertEqual(original_index_settings_dict, updated_index_settings_dict)

    def test_update_structured_marqo_index_custom_model_properties(self):
        """Test updating model properties on StructuredMarqoIndex with custom model."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch(
                    "marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_structured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }
            mock_get_index.return_value = original_index
            self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={
                    "modelProperties": updated_model_properties
                }
            )
            mock_app.update_index_setting.assert_called_once()
            updated_index_settings = mock_app.update_index_setting.call_args[0][0]

            original_index_settings_dict = original_index.dict()
            updated_index_settings_dict = updated_index_settings.dict()
            # Model properties should be updated
            self.assertEqual(updated_model_properties, updated_index_settings_dict["model"]["properties"])
            # Everything else should remain the same
            original_index_settings_dict["model"].pop("properties")
            updated_index_settings_dict["model"].pop("properties")
            self.assertEqual(original_index_settings_dict, updated_index_settings_dict)

    def test_update_structured_marqo_index_non_custom_model_properties(self):
        """Test updating model properties on a non-custom StructuredMarqoIndex."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch(
                    "marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_structured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=False)
            )

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }
            mock_get_index.return_value = original_index
            self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={
                    "modelProperties": updated_model_properties
                }
            )
            mock_app.update_index_setting.assert_called_once()
            updated_index_settings = mock_app.update_index_setting.call_args[0][0]

            original_index_settings_dict = original_index.dict()
            updated_index_settings_dict = updated_index_settings.dict()

            # Model properties should be updated and custom should be set to True
            self.assertEqual(updated_model_properties, updated_index_settings_dict["model"]["properties"])
            self.assertEqual(True, updated_index_settings_dict["model"]["custom"])
            # Everything else should remain the same
            original_index_settings_dict["model"].pop("custom")
            updated_index_settings_dict["model"].pop("properties")
            updated_index_settings_dict["model"].pop("custom")
            self.assertEqual(original_index_settings_dict, updated_index_settings_dict)

    def test_update_index_settings_non_existent_index(self):
        """Test updating settings for an index that does not exist."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            mock_get_index.side_effect = IndexNotFoundError("Index non_existent_index not found")

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }

            with self.assertRaises(IndexNotFoundError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="non_existent_index",
                    settings_dict={
                        "modelProperties": updated_model_properties
                    }
                )

            self.assertIn("non_existent_index", str(context.exception))

    def test_update_index_settings_change_dimensions(self):
        """Test that changing model dimensions raises InvalidModelPropertiesError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to change dimensions from 512 to 768
            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 768
            }

            with self.assertRaises(InvalidModelPropertiesError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "modelProperties": updated_model_properties
                    }
                )

            self.assertIn("dimensions", str(context.exception))

    def test_update_index_settings_change_model_type(self):
        """Test that changing model type raises InvalidModelPropertiesError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to change type from open_clip to hf
            updated_model_properties = {
                "type": "hf",
                "name": "ViTB-B-16",
                "dimensions": 512
            }

            with self.assertRaises(InvalidModelPropertiesError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "modelProperties": updated_model_properties
                    }
                )

            self.assertIn("type", str(context.exception))

    def test_update_index_settings_remove_required_keys(self):
        """Test that removing required keys from model properties raises InvalidModelPropertiesError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512,
                "url": "https://example.com/model"
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to remove the "url" key
            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }

            with self.assertRaises(InvalidModelPropertiesError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "modelProperties": updated_model_properties
                    }
                )

            self.assertIn("must contain all keys", str(context.exception))

    def test_update_index_settings_with_disallowed_settings(self):
        """Test that updating with disallowed settings raises InternalError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to update with a disallowed setting
            with self.assertRaises(InternalError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "normalizeEmbeddings": False  # This is not allowed to be updated
                    }
                )

            self.assertIn("Only the following settings can be updated", str(context.exception))

    def test_update_index_settings_with_multiple_disallowed_settings(self):
        """Test that updating with multiple disallowed settings raises InternalError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to update with multiple disallowed settings
            with self.assertRaises(InternalError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "normalizeEmbeddings": False,
                        "distanceMetric": "cosine"
                    }
                )

            self.assertIn("Only the following settings can be updated", str(context.exception))

    def test_update_index_settings_mixed_allowed_and_disallowed_settings(self):
        """Test that updating with mixed allowed and disallowed settings raises InternalError."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            original_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-16",
                "dimensions": 512
            }
            original_index = self._create_semistructured_index_with_model_properties(
                model=Model(name="default-model", properties=original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            updated_model_properties = {
                "type": "open_clip",
                "name": "ViTB-B-26",
                "dimensions": 512
            }

            # Try to update with both allowed and disallowed settings
            with self.assertRaises(InternalError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="test_index",
                    settings_dict={
                        "modelProperties": updated_model_properties,
                        "normalizeEmbeddings": False
                    }
                )

            self.assertIn("Only the following settings can be updated", str(context.exception))
