from unittest.mock import Mock, patch, MagicMock

from marqo.core.exceptions import InternalError, IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import Model
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase


class TestUpdateIndexSettings(MarqoTestCase):
    """Tests for the update_index_settings_by_settings_dict functionality in IndexManagement."""

    def setUp(self):
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_zookeeper_client = Mock()
        self.index_management = IndexManagement(
            vespa_client=self.mock_vespa_client,
            zookeeper_client=self.mock_zookeeper_client,
            enable_index_operations=True
        )

        self.original_model_properties = {
            "type": "open_clip",
            "name": "ViTB-B-16",
            "dimensions": 512
        }
        self.valid_updated_properties = {
            "type": "open_clip",
            "name": "ViTB-B-26",
            "dimensions": 512
        }
        self.invalid_updated_properties = {
            "type": "open_clip",
            "name": "ViTB-B-26",
            "dimensions": 768  # Changed dimensions - invalid
        }

    def test_update_model_properties_by_index_type_and_custom_flag(self):
        """Test updating model properties for different index types and custom flags."""
        test_cases = [
            {
                "name": "semistructured_custom",
                "index_factory": self.semi_structured_marqo_index,
                "custom": True,
            },
            {
                "name": "semistructured_non_custom",
                "index_factory": self.semi_structured_marqo_index,
                "custom": False,
            },
            {
                "name": "structured_custom",
                "index_factory": self.structured_marqo_index,
                "custom": True,
            },
            {
                "name": "structured_non_custom",
                "index_factory": self.structured_marqo_index,
                "custom": False,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                        patch("marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
                    mock_app = MagicMock()
                    mock_vespa_app.return_value = mock_app

                    original_index = test_case["index_factory"](
                        name="test_index",
                        schema_name="test_schema",
                        model=Model(name="default-model", properties=self.original_model_properties, custom=test_case["custom"])
                    )

                    mock_get_index.return_value = original_index

                    result = self.index_management.update_index_settings_by_settings_dict(
                        index_name="test_index",
                        settings_dict={"modelProperties": self.valid_updated_properties}
                    )

                    # Verify result structure
                    self.assertTrue(result["updated"])
                    self.assertFalse(result["error"])
                    self.assertEqual(result["reason"], "Settings updated successfully")
                    self.assertIn("settingsDiff", result)

                    mock_app.update_index_setting.assert_called_once()
                    updated_index_settings = mock_app.update_index_setting.call_args[0][0]

                    original_index_settings_dict = original_index.dict()
                    updated_index_settings_dict = updated_index_settings.dict()

                    # Model properties should be updated
                    self.assertEqual(self.valid_updated_properties, updated_index_settings_dict["model"]["properties"])
                    # Custom should be set to True after update
                    self.assertTrue(updated_index_settings_dict["model"]["custom"])

                    # Version should be updated
                    original_version = original_index_settings_dict.get("version")
                    updated_version = updated_index_settings_dict.get("version")
                    if original_version is None:
                        self.assertEqual(1, updated_version)
                    else:
                        self.assertEqual(original_version + 1, updated_version)

    def test_update_index_settings_validation_errors(self):
        """Test that validation errors are properly returned for invalid model property changes."""
        test_cases = [
            {
                "name": "change_dimensions",
                "updated_properties": {"type": "open_clip", "name": "ViTB-B-16", "dimensions": 768},
                "expected_error_contains": "dimensions",
            },
            {
                "name": "change_type",
                "updated_properties": {"type": "hf", "name": "ViTB-B-16", "dimensions": 512},
                "expected_error_contains": "type",
            },
            {
                "name": "remove_required_key",
                "original_properties": {"type": "open_clip", "name": "ViTB-B-16", "dimensions": 512, "url": "https://example.com/model"},
                "updated_properties": {"type": "open_clip", "name": "ViTB-B-26", "dimensions": 512},
                "expected_error_contains": "must contain all keys",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
                    original_properties = test_case.get("original_properties", self.original_model_properties)
                    original_index = self.semi_structured_marqo_index(
                        name="test_index",
                        schema_name="test_schema",
                        model=Model(name="default-model", properties=original_properties, custom=True)
                    )
                    mock_get_index.return_value = original_index

                    result = self.index_management.update_index_settings_by_settings_dict(
                        index_name="test_index",
                        settings_dict={"modelProperties": test_case["updated_properties"]}
                    )

                    self.assertFalse(result["updated"])
                    self.assertTrue(result["error"])
                    self.assertIn(test_case["expected_error_contains"], result["reason"])

    def test_update_index_settings_disallowed_settings(self):
        """Test that updating with disallowed settings raises InternalError."""
        test_cases = [
            {
                "name": "single_disallowed",
                "settings_dict": {"normalizeEmbeddings": False},
            },
            {
                "name": "multiple_disallowed",
                "settings_dict": {"normalizeEmbeddings": False, "distanceMetric": "cosine"},
            },
            {
                "name": "mixed_allowed_and_disallowed",
                "settings_dict": {
                    "modelProperties": {"type": "open_clip", "name": "ViTB-B-26", "dimensions": 512},
                    "normalizeEmbeddings": False
                },
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
                    original_index = self.semi_structured_marqo_index(
                        name="test_index",
                        schema_name="test_schema",
                        model=Model(name="default-model", properties=self.original_model_properties, custom=True)
                    )
                    mock_get_index.return_value = original_index

                    with self.assertRaises(InternalError) as context:
                        self.index_management.update_index_settings_by_settings_dict(
                            index_name="test_index",
                            settings_dict=test_case["settings_dict"]
                        )

                    self.assertIn("Only the following settings can be updated", str(context.exception))

    def test_update_index_settings_non_existent_index(self):
        """Test updating settings for an index that does not exist."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index:
            mock_get_index.side_effect = IndexNotFoundError("Index non_existent_index not found")

            with self.assertRaises(IndexNotFoundError) as context:
                self.index_management.update_index_settings_by_settings_dict(
                    index_name="non_existent_index",
                    settings_dict={"modelProperties": self.valid_updated_properties}
                )

            self.assertIn("non_existent_index", str(context.exception))

    def test_update_index_settings_no_changes(self):
        """Test that no update occurs when settings are already up to date."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch("marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app

            original_index = self.semi_structured_marqo_index(
                name="test_index",
                schema_name="test_schema",
                model=Model(name="default-model", properties=self.original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            # Try to update with the same properties
            result = self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={"modelProperties": self.original_model_properties}
            )

            self.assertFalse(result["updated"])
            self.assertFalse(result["error"])
            self.assertEqual(result["reason"], "Settings are already up to date")
            mock_app.update_index_setting.assert_not_called()

    def test_dry_run_and_force_combinations(self):
        """Test all combinations of dry_run and force with valid/invalid changes."""
        test_cases = [
            # dry_run=True always prevents deployment regardless of force
            {
                "name": "dry_run=True, force=False, valid",
                "dry_run": True,
                "force": False,
                "properties": "valid",
                "expected_updated": False,
                "expected_error": False,
                "expected_deployed": False,
                "expected_reason_contains": "Dry run",
            },
            {
                "name": "dry_run=True, force=True, valid",
                "dry_run": True,
                "force": True,
                "properties": "valid",
                "expected_updated": False,
                "expected_error": False,
                "expected_deployed": False,
                "expected_reason_contains": "Dry run",
            },
            {
                "name": "dry_run=True, force=False, invalid",
                "dry_run": True,
                "force": False,
                "properties": "invalid",
                "expected_updated": False,
                "expected_error": True,
                "expected_deployed": False,
                "expected_reason_contains": "Dry run - validation would fail",
            },
            {
                "name": "dry_run=True, force=True, invalid",
                "dry_run": True,
                "force": True,
                "properties": "invalid",
                "expected_updated": False,
                "expected_error": True,
                "expected_deployed": False,
                "expected_reason_contains": "Dry run - validation would fail",
            },
            # dry_run=False with force=False - deploys only if valid
            {
                "name": "dry_run=False, force=False, valid",
                "dry_run": False,
                "force": False,
                "properties": "valid",
                "expected_updated": True,
                "expected_error": False,
                "expected_deployed": True,
                "expected_reason_contains": "Settings updated successfully",
            },
            {
                "name": "dry_run=False, force=False, invalid",
                "dry_run": False,
                "force": False,
                "properties": "invalid",
                "expected_updated": False,
                "expected_error": True,
                "expected_deployed": False,
                "expected_reason_contains": "Validation failed",
            },
            # dry_run=False with force=True - always deploys
            {
                "name": "dry_run=False, force=True, valid",
                "dry_run": False,
                "force": True,
                "properties": "valid",
                "expected_updated": True,
                "expected_error": False,
                "expected_deployed": True,
                "expected_reason_contains": "Settings updated successfully",
            },
            {
                "name": "dry_run=False, force=True, invalid",
                "dry_run": False,
                "force": True,
                "properties": "invalid",
                "expected_updated": True,
                "expected_error": True,
                "expected_deployed": True,
                "expected_reason_contains": "Update forced despite validation errors",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                        patch("marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
                    mock_app = MagicMock()
                    mock_vespa_app.return_value = mock_app

                    original_index = self.semi_structured_marqo_index(
                        name="test_index",
                        schema_name="test_schema",
                        model=Model(name="default-model", properties=self.original_model_properties, custom=True)
                    )
                    mock_get_index.return_value = original_index

                    properties = self.valid_updated_properties if test_case["properties"] == "valid" else self.invalid_updated_properties

                    result = self.index_management.update_index_settings_by_settings_dict(
                        index_name="test_index",
                        settings_dict={"modelProperties": properties},
                        dry_run=test_case["dry_run"],
                        force=test_case["force"]
                    )

                    self.assertEqual(result["updated"], test_case["expected_updated"],
                                     f"Expected updated={test_case['expected_updated']}")
                    self.assertEqual(result["error"], test_case["expected_error"],
                                     f"Expected error={test_case['expected_error']}")
                    self.assertIn(test_case["expected_reason_contains"], result["reason"])

                    if test_case["expected_deployed"]:
                        mock_app.update_index_setting.assert_called_once()
                    else:
                        mock_app.update_index_setting.assert_not_called()

    def test_result_contains_diff_and_settings(self):
        """Test that the result contains settingsDiff, oldSettings, and newSettings fields."""
        with patch('marqo.core.index_management.index_management.IndexManagement.get_index') as mock_get_index, \
                patch("marqo.core.index_management.index_management.IndexManagement._get_vespa_application") as mock_vespa_app:
            mock_app = MagicMock()
            mock_vespa_app.return_value = mock_app

            original_index = self.semi_structured_marqo_index(
                name="test_index",
                schema_name="test_schema",
                model=Model(name="default-model", properties=self.original_model_properties, custom=True)
            )
            mock_get_index.return_value = original_index

            result = self.index_management.update_index_settings_by_settings_dict(
                index_name="test_index",
                settings_dict={"modelProperties": self.valid_updated_properties},
                dry_run=True
            )

            # Check all expected fields are present
            self.assertIn("settingsDiff", result)
            self.assertIn("oldSettings", result)
            self.assertIn("newSettings", result)
            self.assertIn("updated", result)
            self.assertIn("error", result)
            self.assertIn("reason", result)

            # Check settings values
            self.assertEqual(result["oldSettings"]["modelProperties"], self.original_model_properties)
            self.assertEqual(result["newSettings"]["modelProperties"], self.valid_updated_properties)

            # Diff should contain unified diff markers
            self.assertIn("---", result["settingsDiff"])
            self.assertIn("+++", result["settingsDiff"])
