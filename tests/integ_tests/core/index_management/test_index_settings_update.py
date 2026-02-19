import os
import time
import unittest

from marqo import version
from marqo.core.exceptions import IndexNotFoundError, InvalidModelPropertiesError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model
from marqo.inference.native_inference.load_model import (
    get_available_models, _create_model_cache_key
)
from marqo.tensor_search import tensor_search
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

    def test_update_model_properties_e2e(self):
        """Test updating model properties on an existing index end-to-end."""
        index_name = f"test_update_props_{int(time.time())}"

        try:
            # Create index with a custom model
            request = self.unstructured_marqo_index_request(
                name=index_name,
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
            self.index_management.create_index(request)

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
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": new_properties}
            )

            self.assertTrue(result["updated"])

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
        """Test that changing dimensions returns an error result."""
        index_name = f"test_dim_change_{int(time.time())}"

        try:
            request = self.unstructured_marqo_index_request(
                name=index_name,
                model=Model(
                    name='my-custom-model',
                    properties={
                        "dimensions": 384,
                        "type": "open_clip",
                        "name": "ViT-B-16",
                        "url": "https://example.com/model.pt",
                    },
                    custom=True
                )
            )
            self.index_management.create_index(request)

            result = self.index_management.update_index_settings_by_settings_dict(
                index_name,
                {"modelProperties": {
                    "dimensions": 768,  # Changed!
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://example.com/model.pt",
                }}
            )

            self.assertTrue(result["error"])
            self.assertFalse(result["updated"])
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_dry_run_does_not_modify_index(self):
        """Test that dry_run returns diff without changing index."""
        index_name = f"test_dry_run_{int(time.time())}"

        try:
            request = self.unstructured_marqo_index_request(
                name=index_name,
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
            self.index_management.create_index(request)

            new_properties = {
                "dimensions": 384,
                "type": "open_clip",
                "name": "ViT-B-16",
                "url": "https://new-url.com/model.pt",
            }
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": new_properties}, dry_run=True
            )

            self.assertFalse(result["updated"])
            self.assertFalse(result["error"])
            self.assertIn("old-url", result["settingsDiff"])
            self.assertIn("new-url", result["settingsDiff"])

            # Verify index was NOT modified
            index = self.index_management.get_index(index_name)
            self.assertEqual(index.model.properties["url"], "https://old-url.com/model.pt")
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_force_applies_update(self):
        """Test that force=True deploys despite validation errors."""
        index_name = f"test_force_{int(time.time())}"

        try:
            request = self.unstructured_marqo_index_request(
                name=index_name,
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
            self.index_management.create_index(request)

            # Change dimensions (normally invalid) with force=True
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name,
                {"modelProperties": {
                    "dimensions": 768,  # Changed!
                    "type": "open_clip",
                    "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                }},
                force=True
            )

            self.assertTrue(result["updated"])
            self.assertTrue(result["error"])  # Still flagged as error

            # Verify index was modified despite validation error
            updated_index = self.index_management.get_index(index_name)
            self.assertEqual(updated_index.model.properties["dimensions"], 768)
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_no_changes_returns_early(self):
        """Test that identical settings return early without deploying."""
        index_name = f"test_no_changes_{int(time.time())}"

        try:
            props = {
                "dimensions": 384,
                "type": "open_clip",
                "name": "ViT-B-16",
                "url": "https://example.com/model.pt",
            }
            request = self.unstructured_marqo_index_request(
                name=index_name,
                model=Model(
                    name='my-custom-model',
                    properties=props,
                    custom=True
                )
            )
            self.index_management.create_index(request)

            index_before = self.index_management.get_index(index_name)

            result = self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": props}
            )

            self.assertFalse(result["updated"])
            self.assertFalse(result["error"])
            self.assertEqual(result["reason"], "Settings are already up to date")

            # Version should NOT have changed
            index_after = self.index_management.get_index(index_name)
            self.assertEqual(index_before.version, index_after.version)
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_change_name_reloads_model(self):
        """Test that changing 'name' in model properties causes a new model cache entry."""
        index_name = f"test_cache_reload_{int(time.time())}"
        model_name = "my-custom-model"
        original_properties = {"name": "random/small", "dimensions": 32, "type": "random"}

        try:
            request = self.unstructured_marqo_index_request(
                name=index_name,
                model=Model(name=model_name, properties=original_properties, custom=True)
            )
            self.index_management.create_index(request)

            # Add a document and search to trigger model loading
            self.add_documents(
                config=self.config,
                add_docs_params=AddDocsParams(
                    index_name=index_name,
                    docs=[{"_id": "1", "text_field": "hello world"}],
                    tensor_fields=["text_field"],
                    device="cpu"
                )
            )
            tensor_search.search(
                config=self.config, index_name=index_name, text="hello", device="cpu"
            )

            # Verify original cache key exists
            original_key = _create_model_cache_key(model_name, "cpu", original_properties)
            self.assertIn(original_key, get_available_models(),
                          "Original model cache key should exist after first search")

            # Update: change properties['name'] from random/small to random/large
            new_properties = {**original_properties, "name": "random/large"}
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": new_properties}, force=True
            )
            self.assertTrue(result["updated"])

            # Search again to trigger model loading with new properties
            tensor_search.search(
                config=self.config, index_name=index_name, text="hello", device="cpu"
            )

            # Verify new cache key is different and exists
            new_key = _create_model_cache_key(model_name, "cpu", new_properties)
            self.assertNotEqual(original_key, new_key,
                                "Cache key should change when properties['name'] changes")
            self.assertIn(new_key, get_available_models(),
                          "New model cache key should exist after search with updated properties")
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass

    def test_add_triton_model_name_does_not_reload_model(self):
        """Test that adding tritonModelName does NOT create a new model cache entry."""
        index_name = f"test_cache_no_reload_{int(time.time())}"
        model_name = "my-custom-model-2"
        original_properties = {"name": "random/small", "dimensions": 32, "type": "random"}

        try:
            request = self.unstructured_marqo_index_request(
                name=index_name,
                model=Model(name=model_name, properties=original_properties, custom=True)
            )
            self.index_management.create_index(request)

            # Add a document and search to trigger model loading
            self.add_documents(
                config=self.config,
                add_docs_params=AddDocsParams(
                    index_name=index_name,
                    docs=[{"_id": "1", "text_field": "hello world"}],
                    tensor_fields=["text_field"],
                    device="cpu"
                )
            )
            tensor_search.search(
                config=self.config, index_name=index_name, text="hello", device="cpu"
            )

            # Verify original cache key exists
            original_key = _create_model_cache_key(model_name, "cpu", original_properties)
            self.assertIn(original_key, get_available_models(),
                          "Original model cache key should exist after first search")
            keys_before = set(k for k in get_available_models().keys()
                              if k.startswith(model_name))

            # Update: add tritonModelName (not part of cache key formula)
            new_properties = {**original_properties, "tritonModelName": "some-triton-model"}
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name, {"modelProperties": new_properties}
            )
            self.assertTrue(result["updated"])

            # Search again with updated properties
            tensor_search.search(
                config=self.config, index_name=index_name, text="hello", device="cpu"
            )

            # Cache key should be identical since tritonModelName is not in the key
            new_key = _create_model_cache_key(model_name, "cpu", new_properties)
            self.assertEqual(original_key, new_key,
                             "Cache key should NOT change when only tritonModelName is added")
            self.assertIn(original_key, get_available_models(),
                          "Same cache key should still exist")
            keys_after = set(k for k in get_available_models().keys()
                             if k.startswith(model_name))
            self.assertEqual(keys_before, keys_after,
                             "No new cache keys should be added for this model")
        finally:
            try:
                self.index_management.delete_index_by_name(index_name)
            except Exception:
                pass
