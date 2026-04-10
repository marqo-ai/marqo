from marqo.core.index_management.index_management import IndexManagement
from marqo.core.inference.api.exceptions import InferenceError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from tests.integ_tests.marqo_test import MarqoTestCase, TestImageUrls


class TestIndexSettingsUpdate(MarqoTestCase):
    """Integration tests for update_index_settings_by_settings_dict functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_image_index = cls.unstructured_marqo_index_request(
            model=Model(
                name='open_clip/ViT-B-32/laion2b_s34b_b79k',
                properties={
                    'name': 'open_clip/coca_ViT-B-32/laion2b_s13b_b90k',
                    'dimensions': 512,
                    'type': 'open_clip',
                },
                custom=True
            ),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_text_index = cls.unstructured_marqo_index_request(
            model=Model(
                name='hf/all-MiniLM-L6-v2',
                properties={
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "tokens": 256,
                    "type": "hf",
                    "notes": ""
                },
                custom=True
            )
        )

        structured_image_index = cls.structured_marqo_index_request(
            model=Model(
                name='open_clip/ViT-B-32/laion2b_s34b_b79k',
                properties={
                    'name': 'open_clip/coca_ViT-B-32/laion2b_s13b_b90k',
                    'dimensions': 512,
                    'type': 'open_clip',
                },
                custom=True
            ),
            fields=[
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="text_field", type=FieldType.Text)
            ],
            tensor_fields=["image_field", "text_field"]
        )

        structured_text_index = cls.structured_marqo_index_request(
            model=Model(
                name='hf/all-MiniLM-L6-v2',
                properties={
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "tokens": 256,
                    "type": "hf",
                    "notes": ""
                },
                custom=True
            ),
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text)
            ],
            tensor_fields=["text_field"]
        )

        # Index for dry_run and force tests - uses correct model properties from the start
        dry_run_test_index = cls.unstructured_marqo_index_request(
            model=Model(
                name='hf/all-MiniLM-L6-v2',
                properties=get_model_properties("hf/all-MiniLM-L6-v2"),
                custom=True
            )
        )

        cls.indexes = cls.create_indexes([
            unstructured_image_index,
            unstructured_text_index,
            structured_image_index,
            structured_text_index,
            dry_run_test_index
        ])

        cls.unstructured_image_index = unstructured_image_index.name
        cls.unstructured_text_index = unstructured_text_index.name
        cls.structured_image_index = structured_image_index.name
        cls.structured_text_index = structured_text_index.name
        cls.dry_run_test_index = dry_run_test_index.name

    def setUp(self):
        super().setUp()
        self.index_management = IndexManagement(
            self.vespa_client,
            zookeeper_client=self.zookeeper_client,
            enable_index_operations=True,
            deployment_timeout_seconds=30,
            convergence_timeout_seconds=120
        )

    def test_update_model_properties_fixes_inference_errors(self):
        """Test that updating model properties fixes inference errors for different index types."""
        test_cases = [
            {
                "name": "unstructured_image_index",
                "index_name": self.unstructured_image_index,
                "model_name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "documents": [
                    {"_id": "doc1", "image_field": TestImageUrls.IMAGE0.value, "text_field": "A sample text"}
                ],
                "tensor_fields": ["image_field", "text_field"],
                "expected_docs": 1,
                "expected_vectors": 2,
            },
            {
                "name": "unstructured_text_index",
                "index_name": self.unstructured_text_index,
                "model_name": "hf/all-MiniLM-L6-v2",
                "documents": [
                    {"_id": "doc1", "text_field": "A sample text"}
                ],
                "tensor_fields": ["text_field"],
                "expected_docs": 1,
                "expected_vectors": 1,
            },
            {
                "name": "structured_image_index",
                "index_name": self.structured_image_index,
                "model_name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "documents": [
                    {"_id": "doc1", "image_field": TestImageUrls.IMAGE0.value, "text_field": "A sample text"}
                ],
                "tensor_fields": None,  # Structured indexes don't need tensor_fields in AddDocsParams
                "expected_docs": 1,
                "expected_vectors": 2,
            },
            {
                "name": "structured_text_index",
                "index_name": self.structured_text_index,
                "model_name": "hf/all-MiniLM-L6-v2",
                "documents": [
                    {"_id": "doc1", "text_field": "A sample text"}
                ],
                "tensor_fields": None,
                "expected_docs": 1,
                "expected_vectors": 1,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                index_name = test_case["index_name"]
                documents = test_case["documents"]

                # Build AddDocsParams
                add_docs_params_kwargs = {"docs": documents, "index_name": index_name}
                if test_case["tensor_fields"] is not None:
                    add_docs_params_kwargs["tensor_fields"] = test_case["tensor_fields"]

                # Verify that adding documents fails with incorrect model properties
                with self.assertRaises(InferenceError):
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(**add_docs_params_kwargs)
                    )

                # Update to correct model properties
                correct_model_properties = get_model_properties(test_case["model_name"])
                result = self.index_management.update_index_settings_by_settings_dict(
                    index_name=index_name,
                    settings_dict={"modelProperties": correct_model_properties}
                )

                # Verify update result
                self.assertTrue(result["updated"])
                self.assertFalse(result["error"])

                # Adding documents should now succeed
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(**add_docs_params_kwargs)
                )

                # Verify the index was updated correctly
                updated_index = self.index_management.get_index(index_name)
                self.assertEqual(correct_model_properties, updated_index.model.properties)

                # Verify document and vector counts
                index_stats = self.monitoring.get_index_stats_by_name(index_name)
                self.assertEqual(test_case["expected_docs"], index_stats.number_of_documents)
                self.assertEqual(test_case["expected_vectors"], index_stats.number_of_vectors)

    def test_dry_run_does_not_modify_index(self):
        """Test that dry_run=True returns diff information without modifying the index."""
        index_name = self.dry_run_test_index
        original_index = self.index_management.get_index(index_name)
        original_properties = original_index.model.properties.copy()

        test_cases = [
            {
                "name": "valid_changes",
                "updated_properties": {**original_properties, "name": "modified-model-name"},
                "expected_error": False,
                "expected_reason_contains": "Dry run - no changes deployed",
            },
            {
                "name": "invalid_changes_dimension",
                "updated_properties": {**original_properties, "dimensions": 999},
                "expected_error": True,
                "expected_reason_contains": "Dry run - validation would fail",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                updated_properties = test_case["updated_properties"]

                # Perform dry run
                result = self.index_management.update_index_settings_by_settings_dict(
                    index_name=index_name,
                    settings_dict={"modelProperties": updated_properties},
                    dry_run=True
                )

                # Verify result structure
                self.assertFalse(result["updated"])
                self.assertEqual(result["error"], test_case["expected_error"])
                self.assertIn(test_case["expected_reason_contains"], result["reason"])
                self.assertIn("settingsDiff", result)
                self.assertEqual(result["oldSettings"]["modelProperties"], original_properties)
                self.assertEqual(result["newSettings"]["modelProperties"], updated_properties)

                # Verify the index was NOT modified
                current_index = self.index_management.get_index(index_name)
                self.assertEqual(original_properties, current_index.model.properties)

    def test_force_applies_update_despite_validation_error(self):
        """Test that force=True applies the update even when validation fails."""
        index_name = self.dry_run_test_index
        original_index = self.index_management.get_index(index_name)
        original_properties = original_index.model.properties.copy()

        # Create invalid properties (change dimensions - not allowed without force)
        invalid_properties = original_properties.copy()
        invalid_properties["dimensions"] = 999

        # First verify that without force, it fails
        result_without_force = self.index_management.update_index_settings_by_settings_dict(
            index_name=index_name,
            settings_dict={"modelProperties": invalid_properties},
            force=False
        )
        self.assertFalse(result_without_force["updated"])
        self.assertTrue(result_without_force["error"])

        # Now force the update
        result_with_force = self.index_management.update_index_settings_by_settings_dict(
            index_name=index_name,
            settings_dict={"modelProperties": invalid_properties},
            force=True
        )

        # Verify result
        self.assertTrue(result_with_force["updated"])
        self.assertTrue(result_with_force["error"])  # Still has error flag
        self.assertIn("Update forced despite validation errors", result_with_force["reason"])

        # Verify the index WAS modified
        updated_index = self.index_management.get_index(index_name)
        self.assertEqual(invalid_properties, updated_index.model.properties)

        # Restore original properties for other tests
        self.index_management.update_index_settings_by_settings_dict(
            index_name=index_name,
            settings_dict={"modelProperties": original_properties},
            force=True  # Need force because dimensions changed
        )

    def test_no_changes_returns_early(self):
        """Test that updating with identical settings returns early without deployment."""
        index_name = self.dry_run_test_index
        original_index = self.index_management.get_index(index_name)
        original_properties = original_index.model.properties.copy()

        # Update with the same properties
        result = self.index_management.update_index_settings_by_settings_dict(
            index_name=index_name,
            settings_dict={"modelProperties": original_properties}
        )

        # Verify result
        self.assertFalse(result["updated"])
        self.assertFalse(result["error"])
        self.assertEqual(result["reason"], "Settings are already up to date")
