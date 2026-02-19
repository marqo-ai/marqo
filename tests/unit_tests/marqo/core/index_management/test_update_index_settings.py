import unittest
from unittest.mock import Mock, MagicMock, patch

import pydantic.v1

import marqo.version
from marqo.api.models.update_index_settings import UpdateIndexSettingsBodyParams
from marqo.core.exceptions import IndexNotFoundError, InternalError, InvalidModelPropertiesError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, VespaApplicationStore
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

    @patch("marqo.tensor_search.index_meta_cache.get_index")
    def test_update_model_properties_by_index_type_and_custom_flag(self, mock_cache_get_index):
        """Test successful update for different index types and custom flags."""
        cases = [
            ("semi_structured_non_custom", self.semi_structured_marqo_index(
                name="test_index", model=Model(name='hf/e5-small'), version=1
            ), False, {"dimensions": 384, "type": "hf", "name": "hf/e5-small"}),
            ("semi_structured_custom", self.semi_structured_marqo_index(
                name="test_index",
                model=Model(name='my-custom-model', properties={
                    "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                }, custom=True),
                version=1
            ), True, {"dimensions": 768, "type": "open_clip", "name": "ViT-B-16"}),
            ("structured_custom", self.structured_marqo_index(
                name="test_index", schema_name="marqo__test_index",
                model=Model(name='my-custom-model', properties={
                    "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                    "url": "https://old-url.com/model.pt",
                }, custom=True),
            ), True, {"dimensions": 768, "type": "open_clip", "name": "ViT-B-16"}),
            ("structured_non_custom", self.structured_marqo_index(
                name="test_index", schema_name="marqo__test_index",
                model=Model(name='hf/e5-small'),
            ), False, {"dimensions": 384, "type": "hf", "name": "hf/e5-small"}),
        ]

        for label, existing_index, is_custom, base_props in cases:
            with self.subTest(label):
                if not is_custom:
                    existing_index.model.properties = base_props.copy()
                    existing_index.model.custom = False

                mock_vespa_app, _ = self._setup_mocks(existing_index)

                new_properties = {**base_props, "url": "https://new-url.com/model.pt"}

                result = self.index_management.update_index_settings_by_settings_dict(
                    "test_index", {"modelProperties": new_properties}
                )

                self.assertTrue(result["updated"])
                self.assertFalse(result["error"])
                mock_vespa_app.update_index_setting.assert_called_once()
                updated_index = mock_vespa_app.update_index_setting.call_args[0][0]
                self.assertEqual(updated_index.model.properties, new_properties)
                self.assertTrue(updated_index.model.custom)

    def test_update_index_settings_validation_errors(self):
        """Test that validation errors are returned in result dict."""
        cases = [
            ("dimension_change", {
                "dimensions": 384, "type": "open_clip", "name": "ViT-B-16",
                "url": "https://new-url.com/model.pt",
            }, "dimensions"),
            ("type_change", {
                "dimensions": 768, "type": "hf", "name": "ViT-B-16",
                "url": "https://new-url.com/model.pt",
            }, "type"),
            ("key_removal", {
                "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                # "url" key removed
            }, "remove"),
        ]

        for label, new_props, expected_in_reason in cases:
            with self.subTest(label):
                existing_index = self.semi_structured_marqo_index(
                    name="test_index",
                    model=Model(name='my-custom-model', properties={
                        "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                        "url": "https://old-url.com/model.pt",
                    }, custom=True),
                    version=1
                )
                mock_vespa_app, _ = self._setup_mocks(existing_index)

                result = self.index_management.update_index_settings_by_settings_dict(
                    "test_index", {"modelProperties": new_props}
                )

                self.assertTrue(result["error"])
                self.assertFalse(result["updated"])
                self.assertIn(expected_in_reason, result["reason"].lower())
                mock_vespa_app.update_index_setting.assert_not_called()

    def test_update_index_settings_disallowed_settings(self):
        """Test that disallowed settings keys raise InternalError."""
        cases = [
            ("single_disallowed", {"someOtherSetting": "value"}, "someOtherSetting"),
            ("multiple_disallowed", {"foo": 1, "bar": 2}, "foo"),
            ("mixed_allowed_disallowed", {"modelProperties": {"dimensions": 384}, "extra": "x"}, "extra"),
        ]

        for label, settings_dict, expected_key in cases:
            with self.subTest(label):
                with self.assertRaises(InternalError) as ctx:
                    self.index_management.update_index_settings_by_settings_dict(
                        "test_index", settings_dict
                    )
                self.assertIn(expected_key, str(ctx.exception))

    def test_update_index_settings_non_existent_index(self):
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

    def test_update_index_settings_no_changes(self):
        """Test no-op when properties haven't changed."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(name='my-custom-model', properties={
                "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                "url": "https://example.com/model.pt",
            }, custom=True),
            version=1
        )
        mock_vespa_app, _ = self._setup_mocks(existing_index)

        result = self.index_management.update_index_settings_by_settings_dict(
            "test_index",
            {"modelProperties": {
                "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                "url": "https://example.com/model.pt",
            }}
        )

        self.assertFalse(result["updated"])
        self.assertFalse(result["error"])
        self.assertIn("No changes", result["reason"])
        mock_vespa_app.update_index_setting.assert_not_called()

    @patch("marqo.tensor_search.index_meta_cache.get_index")
    def test_dry_run_and_force_combinations(self, mock_cache_get_index):
        """Test all combinations of dry_run, force, and valid/invalid updates."""
        valid_props = {
            "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
            "url": "https://new-url.com/model.pt",
        }
        invalid_props = {
            "dimensions": 384,  # changed dimension
            "type": "open_clip", "name": "ViT-B-16",
            "url": "https://new-url.com/model.pt",
        }

        cases = [
            # (label, dry_run, force, props, expected_updated, expected_error, should_deploy)
            ("dry_run_valid", True, False, valid_props, False, False, False),
            ("dry_run_invalid", True, False, invalid_props, False, True, False),
            ("dry_run_force_valid", True, True, valid_props, False, False, False),
            ("dry_run_force_invalid", True, True, invalid_props, False, True, False),
            ("no_dry_run_no_force_valid", False, False, valid_props, True, False, True),
            ("no_dry_run_no_force_invalid", False, False, invalid_props, False, True, False),
            ("force_valid", False, True, valid_props, True, False, True),
            ("force_invalid", False, True, invalid_props, True, True, True),
        ]

        for label, dry_run, force, props, exp_updated, exp_error, should_deploy in cases:
            with self.subTest(label):
                existing_index = self.semi_structured_marqo_index(
                    name="test_index",
                    model=Model(name='my-custom-model', properties={
                        "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                        "url": "https://old-url.com/model.pt",
                    }, custom=True),
                    version=1
                )
                mock_vespa_app, _ = self._setup_mocks(existing_index)

                result = self.index_management.update_index_settings_by_settings_dict(
                    "test_index", {"modelProperties": props},
                    force=force, dry_run=dry_run
                )

                self.assertEqual(result["updated"], exp_updated, f"updated mismatch for {label}")
                self.assertEqual(result["error"], exp_error, f"error mismatch for {label}")
                if should_deploy:
                    mock_vespa_app.update_index_setting.assert_called_once()
                else:
                    mock_vespa_app.update_index_setting.assert_not_called()

    @patch("marqo.tensor_search.index_meta_cache.get_index")
    def test_result_contains_diff_and_settings(self, mock_cache_get_index):
        """Verify result dict structure and diff content."""
        existing_index = self.semi_structured_marqo_index(
            name="test_index",
            model=Model(name='my-custom-model', properties={
                "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
                "url": "https://old-url.com/model.pt",
            }, custom=True),
            version=1
        )
        mock_vespa_app, _ = self._setup_mocks(existing_index)

        new_properties = {
            "dimensions": 768, "type": "open_clip", "name": "ViT-B-16",
            "url": "https://new-url.com/model.pt",
        }
        result = self.index_management.update_index_settings_by_settings_dict(
            "test_index", {"modelProperties": new_properties}
        )

        # Check all expected keys exist
        for key in ("updated", "error", "oldSettings", "newSettings", "settingsDiff", "reason"):
            self.assertIn(key, result)

        # Check settings content
        self.assertEqual(result["oldSettings"]["modelProperties"]["url"], "https://old-url.com/model.pt")
        self.assertEqual(result["newSettings"]["modelProperties"]["url"], "https://new-url.com/model.pt")

        # Check diff contains the URLs
        self.assertIn("old-url", result["settingsDiff"])
        self.assertIn("new-url", result["settingsDiff"])


class TestUpdateIndexSettingsBodyParams(unittest.TestCase):
    """Tests for the UpdateIndexSettingsBodyParams request model."""

    def test_valid_request_with_alias(self):
        """Test parsing a valid request using the modelProperties alias."""
        body = UpdateIndexSettingsBodyParams(**{"modelProperties": {"dimensions": 384, "type": "hf"}})
        self.assertEqual(body.model_properties, {"dimensions": 384, "type": "hf"})

    def test_valid_request_with_field_name(self):
        """Test parsing a valid request using the model_properties field name."""
        body = UpdateIndexSettingsBodyParams(**{"model_properties": {"dimensions": 384, "type": "hf"}})
        self.assertEqual(body.model_properties, {"dimensions": 384, "type": "hf"})

    def test_dict_by_alias(self):
        """Test that .dict(by_alias=True) returns modelProperties key."""
        body = UpdateIndexSettingsBodyParams(**{"modelProperties": {"dimensions": 384}})
        result = body.dict(by_alias=True)
        self.assertIn("modelProperties", result)
        self.assertEqual(result["modelProperties"], {"dimensions": 384})

    def test_missing_model_properties_raises_validation_error(self):
        """Test that missing modelProperties raises a validation error."""
        with self.assertRaises(pydantic.v1.ValidationError):
            UpdateIndexSettingsBodyParams(**{})

    def test_extra_field_raises_validation_error(self):
        """Test that extra fields are rejected (StrictBaseModel)."""
        with self.assertRaises(pydantic.v1.ValidationError):
            UpdateIndexSettingsBodyParams(**{
                "modelProperties": {"dimensions": 384},
                "extraField": "not_allowed"
            })


class TestVespaApplicationPackageUpdateIndexSetting(MarqoTestCase):
    """Tests for VespaApplicationPackage.update_index_setting()."""

    def setUp(self):
        self.mock_store = Mock(spec=VespaApplicationStore)
        self.mock_store.read_text_file.side_effect = self._mock_read_text_file
        self.mock_store.file_exists.side_effect = self._mock_file_exists
        self.vespa_app = VespaApplicationPackage(self.mock_store)

    def _mock_file_exists(self, *paths):
        return paths in (('marqo_config.json',), ('services.xml',))

    def _mock_read_text_file(self, *paths):
        if paths == ('services.xml',):
            return '''<?xml version="1.0" encoding="utf-8"?>
<services xmlns:deploy="vespa" xmlns:preprocess="properties">
    <container>
        <document-api/>
        <document-processing/>
        <search/>
    </container>
    <content>
        <documents>
        </documents>
    </content>
</services>'''
        elif paths == ('marqo_config.json',):
            return '{"version": "2.24.0"}'
        else:
            return None

    def test_update_index_setting_success(self):
        """Test successful settings-only update persists and deploys."""
        index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            version=1
        )
        # Add the index to the store so has_index returns True
        self.vespa_app._index_setting_store.save_index_setting(index)
        self.mock_store.save_file.reset_mock()

        updated_index = index.copy(update={
            'model': index.model.copy(update={'properties': {"dimensions": 384, "type": "hf"}, 'custom': True})
        })

        self.vespa_app.update_index_setting(updated_index)

        # Should persist index settings (2 calls: settings + history)
        # and deploy (1 call)
        self.mock_store.deploy_application.assert_called_once()
        # Settings file + history file = at least 2 save_file calls
        settings_calls = [
            c for c in self.mock_store.save_file.call_args_list
            if any('marqo_index_settings' in str(a) for a in c[0])
        ]
        self.assertEqual(len(settings_calls), 2)

    def test_update_index_setting_not_found(self):
        """Test that updating a non-existent index raises IndexNotFoundError."""
        index = self.semi_structured_marqo_index(
            name="nonexistent_index",
            schema_name="marqo__nonexistent",
            version=1
        )

        with self.assertRaises(IndexNotFoundError):
            self.vespa_app.update_index_setting(index)

        self.mock_store.deploy_application.assert_not_called()

    def test_update_index_setting_increments_version(self):
        """Test that update_index_setting increments the version."""
        # First save with version=None (gets stored as version=1)
        index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            version=None
        )
        self.vespa_app._index_setting_store.save_index_setting(index)
        stored = self.vespa_app._index_setting_store.get_index("test_index")
        self.assertEqual(stored.version, 1)

        self.mock_store.save_file.reset_mock()

        # update_index_setting should bump version from 1 to 2
        self.vespa_app.update_index_setting(stored)

        saved_index = self.vespa_app._index_setting_store.get_index("test_index")
        self.assertEqual(saved_index.version, 2)

    def test_update_index_setting_none_version(self):
        """Test that update_index_setting handles None version correctly."""
        index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="marqo__test_index",
            version=None
        )
        # save_index_setting with version=None assigns version=1
        self.vespa_app._index_setting_store.save_index_setting(index)
        self.mock_store.save_file.reset_mock()

        # Now the stored index has version=1, update it
        stored = self.vespa_app._index_setting_store.get_index("test_index")
        self.vespa_app.update_index_setting(stored)

        updated = self.vespa_app._index_setting_store.get_index("test_index")
        self.assertEqual(updated.version, 2)


class TestUpdateIndexSettingsApiEndpoint(unittest.TestCase):
    """Tests for the PATCH /indexes/{index_name}/index-settings API endpoint."""

    def setUp(self):
        from marqo.tensor_search import api
        from starlette.testclient import TestClient
        self.api = api

        mock_config = Mock()
        mock_index_mgmt = Mock()
        self._mock_result = {
            "updated": True, "error": False,
            "oldSettings": {}, "newSettings": {},
            "settingsDiff": "", "reason": "Settings updated successfully"
        }
        mock_index_mgmt.update_index_settings_by_settings_dict = Mock(return_value=self._mock_result)
        mock_config.index_management = mock_index_mgmt
        self.mock_config = mock_config
        self.mock_index_mgmt = mock_index_mgmt

        self.api.app.dependency_overrides[self.api.get_config] = lambda: mock_config
        self.client = TestClient(self.api.app, raise_server_exceptions=False)

    def tearDown(self):
        self.api.app.dependency_overrides.clear()

    @patch.dict("os.environ", {"MARQO_ENABLE_OPS_API": "true"})
    def test_update_index_settings_success(self):
        """Test successful PATCH /indexes/{index_name}/index-settings."""
        resp = self.client.patch(
            "/indexes/my_index/index-settings",
            json={"modelProperties": {"dimensions": 384, "type": "hf", "name": "hf/e5-small"}}
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), self._mock_result)
        self.mock_index_mgmt.update_index_settings_by_settings_dict.assert_called_once_with(
            "my_index",
            {"modelProperties": {"dimensions": 384, "type": "hf", "name": "hf/e5-small"}},
            force=False, dry_run=False
        )

    @patch.dict("os.environ", {"MARQO_ENABLE_OPS_API": "true"})
    def test_update_index_settings_invalid_body(self):
        """Test PATCH with missing modelProperties returns validation error."""
        resp = self.client.patch(
            "/indexes/my_index/index-settings",
            json={}
        )

        self.assertIn(resp.status_code, [400, 422])

    @patch.dict("os.environ", {}, clear=False)
    def test_update_index_settings_ops_api_disabled(self):
        """Test PATCH is rejected when MARQO_ENABLE_OPS_API is not set."""
        import os
        os.environ.pop("MARQO_ENABLE_OPS_API", None)

        resp = self.client.patch(
            "/indexes/my_index/index-settings",
            json={"modelProperties": {"dimensions": 384, "type": "hf"}}
        )

        self.assertNotEqual(resp.status_code, 200)
