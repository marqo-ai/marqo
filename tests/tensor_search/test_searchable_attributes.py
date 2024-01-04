import os
from unittest import mock

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestSearchStructured(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.structured_marqo_index_request(
            model = Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_4", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_5", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_6", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_7", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_8", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="int_field_1", type=FieldType.Int,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="float_field_1", type=FieldType.Float,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="bool_field_1", type=FieldType.Bool,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3",
                           "text_field_4", "text_field_5", "text_field_6"]
        )

        default_image_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "text_field_3", "image_field_1", "image_field_2"]
        )

        image_index_with_random_model = cls.structured_marqo_index_request(
            model=Model(name='random/small'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer)
            ],
            tensor_fields=["text_field_1", "text_field_2", "image_field_1"]
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
            default_image_index,
            image_index_with_random_model
        ])

        cls.default_text_index = default_text_index.name
        cls.default_image_index = default_image_index.name
        cls.image_index_with_random_model = image_index_with_random_model.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_searchable_attributes_works(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
            index_name=self.default_text_index,
            docs=[
                {"text_field_1": "exact match field",
                 "text_field_2": "baaadd efgh ",
                 "text_field_3": "some field efgh ",
                 "_id": "1"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "exact match field",
                 "text_field_3": "Random text here efgh ",
                 "_id": "2"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "shouldn't really match",
                 "text_field_3": "exact match field",
                 "_id": "3"},
            ],
            )
        )
        test_cases = [
            ("text_field_1", "exact match field", "1"),
            ("text_field_2", "exact match field", "2"),
            ("text_field_3", "exact match field", "3"),
        ]

        for field_name, query, expected_id in test_cases:
            with self.subTest(field_name):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index,
                    text=query,
                    searchable_attributes=[field_name]
                )
                self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_searchable_attributes_empty_list(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
            index_name=self.default_text_index,
            docs=[
                {"text_field_1": "exact match field",
                 "text_field_2": "baaadd efgh ",
                 "text_field_3": "some field efgh ",
                 "_id": "1"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "exact match field",
                 "text_field_3": "Random text here efgh ",
                 "_id": "2"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "shouldn't really match",
                 "text_field_3": "exact match field",
                 "_id": "3"},
            ],
            )
        )
        for search_method in ["TENSOR", "LEXICAL"]:
            with self.subTest(search_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index,
                    text="exact match field",
                    searchable_attributes=[],
                    search_method=search_method
                )
                self.assertEqual(0, len(res["hits"]))

    def test_searchable_attributes_None(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
            index_name=self.default_text_index,
            docs=[
                {"text_field_1": "exact match field",
                 "text_field_2": "baaadd efgh ",
                 "text_field_3": "some field efgh ",
                 "_id": "1"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "exact match field",
                 "text_field_3": "Random text here efgh ",
                 "_id": "2"},
                {"text_field_1": "shouldn't really match ",
                 "text_field_2": "shouldn't really match",
                 "text_field_3": "exact match field",
                 "_id": "3"},
            ],
            )
        )
        for search_method in ["TENSOR", "LEXICAL"]:
            with self.subTest(search_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index,
                    text="exact match field",
                    searchable_attributes=None,
                    search_method=search_method
                )
                self.assertEqual(3, len(res["hits"]))



