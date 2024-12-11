import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestSearchCommon(MarqoTestCase):
    """A class to test common search functionalities for structured and unstructured indexes.

    We should test the shared functionalities between structured and unstructured indexes here to avoid code duplication
    and branching in the test cases."""

    structured_text_index_name = "structured_index_text" + str(uuid.uuid4()).replace('-', '')
    structured_image_index_name = "structured_image_index" + str(uuid.uuid4()).replace('-', '')
    structured_filter_index_name = "structured_filter_index" + str(uuid.uuid4()).replace('-', '')

    unstructured_text_index_name = "unstructured_index_text" + str(uuid.uuid4()).replace('-', '')
    unstructured_image_index_name = "unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.structured_text_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.structured_filter_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "field_a", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "field_b", "type": "text", "features": ["filter"]},
                    {"name": "str_for_filtering", "type": "text", "features": ["filter"]},
                    {"name": "int_for_filtering", "type": "int", "features": ["filter"]},
                    {"name": "long_field_1", "type": "long", "features": ["filter"]},
                    {"name": "double_field_1", "type": "double", "features": ["filter"]},
                    {"name": "array_long_field_1", "type": "array<long>", "features": ["filter"]},
                    {"name": "array_double_field_1", "type": "array<double>", "features": ["filter"]}
                ],
                "tensorFields": ["field_a", "field_b"],
            },
            {
                "indexName": cls.structured_image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_content", "type": "image_pointer"},
                ],
                "tensorFields": ["title", "image_content"],
            }
        ])

        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            }
        ])

        cls.indexes_to_delete = [cls.structured_image_index_name, cls.structured_filter_index_name,
                                 cls.structured_text_index_name, cls.unstructured_image_index_name,
                                 cls.unstructured_text_index_name]

    def test_lexical_query_can_not_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                          {"vector": [2, ] * 384, "weight": 2}]}

        test_case = [
            (None, context, "with context"),
            (None, None, "without context")
        ]
        for index_name in [self.structured_text_index_name, self.unstructured_image_index_name]:
            for query, context, msg in test_case:
                with self.subTest(f"{index_name} - {msg}"):
                    with self.assertRaises(MarqoWebError) as e:
                        res = self.client.index(index_name).search(q=None, context=context, search_method="LEXICAL")
                    self.assertIn("Query(q) is required for lexical search", str(e.exception.message))

    def test_tensor_search_query_can_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                          {"vector": [2, ] * 384, "weight": 2}]}
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            res = self.client.index(index_name).search(q=None, context=context)
            self.assertIn("hits", res)