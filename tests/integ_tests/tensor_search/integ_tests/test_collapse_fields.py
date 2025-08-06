import os
from unittest import mock

from marqo.core.models.marqo_index import CollapseField, SemiStructuredMarqoIndex
from tests.integ_tests.marqo_test import MarqoTestCase


class TestCollapseFields(MarqoTestCase):
    """Integration tests for collapse fields functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            collapse_fields=[CollapseField(name="parent_id", minGroups=100)]
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
        ])

        cls.default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_index_should_contain_collapse_field_settings(self):
        index = self.index_management.get_index(self.default_text_index.name)
        self.assertIsInstance(index, SemiStructuredMarqoIndex)
        self.assertIsNotNone(index.collapse_fields)
        self.assertEqual(index.collapse_fields[0].name, "parent_id")
        self.assertEqual(index.collapse_fields[0].minGroups, 100)
