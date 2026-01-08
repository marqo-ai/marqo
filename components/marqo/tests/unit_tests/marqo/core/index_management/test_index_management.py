import unittest
from unittest.mock import Mock, patch, MagicMock

import marqo.version
import marqo.version
from marqo.core.exceptions import InternalError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import Model
from marqo.vespa.vespa_client import VespaClient
from tests.unit_tests.marqo_test import MarqoTestCase


class TestIndexManagementUpdateIndex(MarqoTestCase):
    def setUp(self):
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_zookeeper_client = Mock()
        self.index_management = IndexManagement(
            vespa_client=self.mock_vespa_client,
            zookeeper_client=self.mock_zookeeper_client,
            enable_index_operations=True
        )

    def _create_semi_structured_index(self, name="test_index"):
        """Helper method to create a SemiStructuredMarqoIndex for testing."""
        return self.semi_structured_marqo_index(
            name=name,
            model=Model(name='hf/e5-small-v2'),
            version=1
        )

    def _create_structured_index(self, name="test_index"):
        """Helper method to create a StructuredMarqoIndex for testing."""
        return self.structured_marqo_index(
            name=name,
            schema_name=f"marqo__{name}",
            model=Model(name='hf/e5-small-v2'),
            fields=[],  # StructuredMarqoIndex requires fields list
            tensor_fields=[]  # StructuredMarqoIndex requires tensor_fields list
        )

    def _create_unstructured_index(self, name="test_index"):
        """Helper method to create an UnstructuredMarqoIndex for testing."""
        # UnstructuredMarqoIndex is not in MarqoTestCase, let's create a minimal one
        from marqo.core.models.marqo_index import UnstructuredMarqoIndex, DistanceMetric, VectorNumericType, HnswConfig
        from marqo.core.models.marqo_index import TextPreProcessing, ImagePreProcessing, TextSplitMethod
        import time

        return UnstructuredMarqoIndex(
            name=name,
            schema_name=f"marqo__{name}",
            model=Model(name='hf/e5-small'),
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=128, m=16),
            marqo_version=marqo.version.get_version(),
            created_at=int(time.time()),
            updated_at=int(time.time()),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True,
            filter_string_max_length=100,
        )

    @patch('marqo.core.index_management.index_management.SemiStructuredVespaSchema.generate_vespa_schema')
    def test_update_index_success(self, mock_generate_schema):
        """Test update proceeds when any of the three subset checks fail (tensor_field_map, field_map, name_to_string_array_field_map)."""
        base_existing = self.semi_structured_marqo_index(name="test_index", model=Model(name='hf/e5-small'))

        test_cases = [
            ("tensor_field_map", dict(tensor_field_names=('field1',))),
            ("field_map", dict(lexical_field_names=('field1',))),
            ("name_to_string_array_field_map", dict(string_array_field_names=('field1',)))
        ]

        for case_name, updated_params in test_cases:
            with self.subTest(case=case_name):
                updated_index = self.semi_structured_marqo_index(name="test_index", model=Model(name='hf/e5-small'),
                                                                 **updated_params)
                mock_vespa_app = Mock()
                mock_deployment_lock = MagicMock()
                mock_generate_schema.return_value = "mock_schema"

                self.index_management.get_index = Mock(return_value=base_existing)
                self.index_management._get_vespa_application = Mock(return_value=mock_vespa_app)
                self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)

                self.index_management.update_index(updated_index)

                mock_generate_schema.assert_called_with(updated_index)
                mock_vespa_app.update_index_setting_and_schema.assert_called_with(updated_index, "mock_schema")
                mock_deployment_lock.__enter__.assert_called_once()
                mock_deployment_lock.__exit__.assert_called_once()
                mock_generate_schema.reset_mock()
                mock_vespa_app.reset_mock()

    def test_update_index_skip_if_nothing_to_update(self):
        """Test that update is skipped if the index has no new changes."""
        # Setup
        base_index = self._create_semi_structured_index()

        # Make the updated index a subset of the existing index
        updated_index = base_index.copy(update={
            "tensor_field_map": {"field1": "mapping1"},
            "field_map": {"field2": "mapping2"},
            "name_to_string_array_field_map": {"field3": "mapping3"}
        })

        existing_index = base_index.copy(update={
            "tensor_field_map": {"field1": "mapping1", "field4": "mapping4"},
            "field_map": {"field2": "mapping2", "field5": "mapping5"},
            "name_to_string_array_field_map": {"field3": "mapping3", "field6": "mapping6"}
        })

        mock_vespa_app = Mock()
        mock_deployment_lock = MagicMock()

        # Mock methods
        self.index_management.get_index = Mock(return_value=existing_index)
        self.index_management._get_vespa_application = Mock(return_value=mock_vespa_app)
        self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)

        # Execute
        self.index_management.update_index(updated_index)

        # Verify that deployment lock was not acquired and update was not called
        self.index_management.get_index.assert_called_once_with(updated_index.name)
        self.index_management._vespa_deployment_lock.assert_called_once()
        mock_deployment_lock.__enter__.assert_called_once()
        mock_deployment_lock.__exit__.assert_called_once()
        mock_vespa_app.update_index_setting_and_schema.assert_not_called()

    def test_update_index_raises_internal_error_for_non_semi_structured_index(self):
        """Test that InternalError is raised for non-SemiStructuredMarqoIndex types."""
        test_cases = [
            ("StructuredMarqoIndex", self._create_structured_index()),
            ("UnstructuredMarqoIndex", self._create_unstructured_index())
        ]

        for index_type, index in test_cases:
            with self.subTest(index_type=index_type):
                # Setup
                mock_deployment_lock = MagicMock()
                self.index_management.get_index = Mock(return_value=index)
                self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)

                # Execute & Verify
                with self.assertRaises(InternalError) as context:
                    self.index_management.update_index(index)

                self.assertIn("can not be updated", str(context.exception))
                # Verify deployment lock was acquired
                self.index_management._vespa_deployment_lock.assert_called_once()
                mock_deployment_lock.__enter__.assert_called_once()
                mock_deployment_lock.__exit__.assert_called_once()

    @patch('marqo.core.index_management.index_management.vespa_schema_factory')
    @patch('marqo.core.index_management.index_management.TypeaheadVespaSchema')
    def test_batch_create_indexes_generates_typeahead_schema(self, mock_typeahead_schema_class, mock_vespa_schema_factory):
        """Test that batch_create_indexes properly generates typeahead schemas."""
        # Setup request
        request = self.unstructured_marqo_index_request(
            name="test_index",
            model=Model(name='hf/e5-small-v2')
        )

        # Setup mock returns
        mock_main_schema = "main_schema_content"
        mock_marqo_index = self._create_semi_structured_index("test_index")
        mock_vespa_schema_factory.return_value.generate_schema.return_value = (mock_main_schema, mock_marqo_index)

        # Setup typeahead schema mocks
        mock_updated_index = mock_marqo_index.copy(deep=True, update={"typeahead_schema_name": "marqo__test_index_typeahead"})
        mock_typeahead_schema = "typeahead_schema_content"
        mock_typeahead_instance = Mock()
        mock_typeahead_instance.generate_schema.return_value = (mock_typeahead_schema, mock_updated_index)
        mock_typeahead_schema_class.return_value = mock_typeahead_instance

        # Setup other mocks
        mock_vespa_app = Mock()
        mock_deployment_lock = MagicMock()
        self.index_management._get_vespa_application = Mock(return_value=mock_vespa_app)
        self.index_management._vespa_deployment_lock = Mock(return_value=mock_deployment_lock)

        # Execute
        result = self.index_management.batch_create_indexes([request])

        # Verify TypeaheadVespaSchema was called with the main marqo_index
        mock_typeahead_schema_class.assert_called_once_with(mock_marqo_index)
        mock_typeahead_instance.generate_schema.assert_called_once()

        # Verify batch_add_index_setting_and_schema was called with the correct tuple
        expected_call = [(mock_main_schema, mock_typeahead_schema, mock_updated_index)]
        mock_vespa_app.batch_add_index_setting_and_schema.assert_called_once_with(expected_call)

        # Verify the result contains the updated index with typeahead schema name
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].typeahead_schema_name, "marqo__test_index_typeahead")

        # Verify deployment lock was used
        mock_deployment_lock.__enter__.assert_called_once()
        mock_deployment_lock.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
