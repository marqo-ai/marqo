import os
from unittest import mock

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.core.models.add_docs_params import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestSearchableAttributes(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_text_index = cls.structured_marqo_index_request(
            model = Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
        )

        semi_structured_text_index = cls.unstructured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
        )

        cls.indexes = cls.create_indexes([
            structured_text_index,
            semi_structured_text_index
        ])

        cls.structured_text_index = structured_text_index.name
        cls.semi_structured_text_index = semi_structured_text_index.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def _add_documents(self, index_name):
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index_name,
                docs=[
                    {"text_field_1": "exact match field",
                     "text_field_2": "baaadd efgh ",
                     "text_field_3": "some field efgh",
                     "_id": "1"},
                    {"text_field_1": "shouldn't really match ",
                     "text_field_2": "exact match field",
                     "text_field_3": "Random text here efgh",
                     "_id": "2"},
                    {"text_field_1": "shouldn't really match ",
                     "text_field_2": "shouldn't really match",
                     "text_field_3": "exact match field",
                     "_id": "3"},
                ],
                tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
                if index_name != self.structured_text_index else None
            )
        )

    def test_searchable_attributes_works(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            with self.subTest(index_name=index_name):
                self._add_documents(index_name)
                test_cases = [
                    ("text_field_1", "exact match field", "1"),
                    ("text_field_2", "exact match field", "2"),
                    ("text_field_3", "exact match field", "3"),
                ]

                for field_name, query, expected_id in test_cases:
                    with self.subTest(field_name):
                        res = tensor_search.search(
                            config=self.config,
                            index_name=index_name,
                            text=query,
                            searchable_attributes=[field_name]
                        )
                        self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_searchable_attributes_works_with_filter(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            with self.subTest(index_name=index_name):
                self._add_documents(index_name)

                test_cases = [
                    (["text_field_1", "text_field_2"], "exact match field", "text_field_3:(some field efgh)", "1"),
                    (["text_field_1", "text_field_2"], "exact match field", "text_field_3:(Random text here efgh)", "2"),
                    (["text_field_2", "text_field_3"], "exact match field", "text_field_2:(shouldn't really match)", "3"),
                ]
                for search_method in ["TENSOR", "LEXICAL"]:
                    for searchable_attributes, query, filter_string, expected_id in test_cases:
                        with self.subTest(f"{search_method}-{searchable_attributes}-{filter_string}"):
                            res = tensor_search.search(
                                config=self.config,
                                index_name=index_name,
                                text=query,
                                searchable_attributes=searchable_attributes,
                                filter=filter_string
                            )
                            self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_searchable_attributes_empty_list(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            with self.subTest(index_name=index_name):
                self._add_documents(index_name)
                for search_method in ["TENSOR", "LEXICAL"]:
                    with self.subTest(search_method):
                        res = tensor_search.search(
                            config=self.config,
                            index_name=index_name,
                            text="exact match field",
                            searchable_attributes=[],
                            search_method=search_method
                        )
                        self.assertEqual(0, len(res["hits"]))

    def test_searchable_attributes_empty_list_with_filter(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            with self.subTest(index_name=index_name):
                self._add_documents(index_name)

                filter_string_cases = ["text_field_1:(exact match field)", "text_field_2:(exact match field)",
                                       "text_field_3:(exact match field)"]

                for search_method in ["TENSOR", "LEXICAL"]:
                    for filter_string in filter_string_cases:
                        with self.subTest(f"{search_method}-{filter_string}"):
                            res = tensor_search.search(
                                config=self.config,
                                index_name=index_name,
                                text="exact match field",
                                searchable_attributes=[],
                                search_method=search_method,
                                filter=filter_string
                            )
                            self.assertEqual(0, len(res["hits"]))

    def test_searchable_attributes_None(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            with self.subTest(index_name=index_name):
                self._add_documents(index_name)
                for search_method in ["TENSOR", "LEXICAL"]:
                    with self.subTest(search_method):
                        res = tensor_search.search(
                            config=self.config,
                            index_name=index_name,
                            text="exact match field",
                            searchable_attributes=None,
                            search_method=search_method
                        )
                        self.assertEqual(3, len(res["hits"]))

    def test_searchable_attributes_behaves_the_same_way_for_different_types_of_indexes(self):
        for index_name in [self.structured_text_index, self.semi_structured_text_index]:
            self._add_documents(index_name)

        searchable_attributes_test_cases = [
            None,
            ["text_field_1"],
            ["text_field_1", "text_field_2"],
            ["text_field_2", "text_field_3"],
            ["text_field_1", "text_field_2", "text_field_3"],
        ]

        self.maxDiff = None
        for search_method in ["TENSOR", "LEXICAL"]:
            for searchable_attributes in searchable_attributes_test_cases:
                with self.subTest(search_method):
                    res_structured = tensor_search.search(
                        config=self.config,
                        index_name=self.structured_text_index,
                        text="exact match field",
                        searchable_attributes=searchable_attributes,
                        search_method=search_method
                    )

                    res_semi_structured = tensor_search.search(
                        config=self.config,
                        index_name=self.semi_structured_text_index,
                        text="exact match field",
                        searchable_attributes=searchable_attributes,
                        search_method=search_method
                    )

                    self.assertCountEqual(res_structured['hits'], res_semi_structured['hits'],
                                          msg=f'Search result differs for search method: {search_method} and '
                                              f'searchable attributes: {searchable_attributes}')

