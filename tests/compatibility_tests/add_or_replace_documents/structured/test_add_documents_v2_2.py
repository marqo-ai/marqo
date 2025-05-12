import traceback

import pytest
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.2.0')
class TestAddDocumentsv2_2(BaseCompatibilityTestCase):
    structured_index_name = "test_add_doc_api_structured_index_2_2_0"

    indexes_to_test_on = [
        {
            "indexName": structured_index_name,
            "type": "structured",
            "vectorNumericType": "float",
            "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 2,
                "splitOverlap": 0,
                "splitMethod": "sentence",
            },
            "imagePreprocessing": {"patchMethod": None},
            "allFields": [
                {"name": "text_field", "type": "text", "features": ["lexical_search"]},
                {"name": "caption", "type": "text", "features": ["lexical_search", "filter"]},
                {"name": "tags", "type": "array<text>", "features": ["filter"]},
                {"name": "image_field", "type": "image_pointer"},
                {"name": "my_int", "type": "int", "features": ["score_modifier"]},
                # this field maps the above image field and text fields into a multimodal combination.
                {
                    "name": "multimodal_field",
                    "type": "multimodal_combination",
                    "dependentFields": {"image_field": 0.9, "text_field": 0.1},
                },
                {"name": "boolean_field", "type": "bool"},
                {"name": "float_field_1", "type": "float"},
                {"name": "array_int_field_1", "type": "array<int>"},
                {"name": "array_float_field_1", "type": "array<float>"},
                {"name": "array_long_field_1", "type": "array<long>"},
                {"name": "array_double_field_1", "type": "array<double>"},
                {"name": "long_field_1", "type": "long"},
                {"name": "double_field_1", "type": "double"},
            ],
            "tensorFields": ["multimodal_field"],
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {"efConstruction": 512, "m": 16},
            }
        }]

    text_docs = [{
        "text_field": "The Travels of Marco Polo",
        "caption": "A 13th-century travelogue describing the travels of Polo",
        "tags": ["wow", "this", "is", "awesome"],
        "my_int": 123,
        "boolean_field": True,
        "float_field_1": 1.23,
        "array_int_field_1": [1, 2, 3],
        "array_float_field_1": [1.23, 2.34, 3.45],
        "array_long_field_1": [1234567890, 2345678901, 3456789012],
        "array_double_field_1": [1.234567890, 2.345678901, 3.456789012],
        "long_field_1": 1234567890,
        "double_field_1": 1.234567890,

        "_id": "article_602"
    },
    {
        "text_field": "Extravehicular Mobility Unit (EMU)",
        "caption": "The EMU is a spacesuit that provides environmental protection",
        "tags": ["space", "EMU", "NASA", "astronaut"],
        "my_int": 354,
        "boolean_field": True,
        "float_field_1": 1.56,
        "array_int_field_1": [4, 5, 6],
        "array_float_field_1": [1.14, 2.21, 3.31],
        "array_long_field_1": [3456789012, 1234567890, 2345678901],
        "array_double_field_1": [1.234567890, 2.345678901, 3.456789012],
        "long_field_1": 1234567890,
        "double_field_1": 1.234567890,
        "_id": "article_603"
    }]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().setUpClass()

    def prepare(self):
        self.logger.debug(f"Creating indexes {self.indexes_to_test_on} in test case: {self.__class__.__name__}")
        self.create_indexes(self.indexes_to_test_on)

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')

        errors = []  # Collect errors to report them at the end

        for index in self.indexes_to_test_on:
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs)
                else:
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs,
                                                                tensor_fields = ["Description", "Genre", "Title"])
            except Exception as e:
                errors.append((index, traceback.format_exc()))

        all_results = {}

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}

            for doc in self.text_docs:
                try:
                    doc_id = doc['_id']
                    all_results[index_name][doc_id] = self.client.index(index_name).get_document(doc_id)
                except Exception as e:
                    errors.append((index, traceback.format_exc()))

        if errors:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")
        self.save_results_to_file(all_results)

    def test_add_doc(self):
        self.logger.info(f"Running test_add_doc on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions


        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            for doc in self.text_docs:
                doc_id = doc['_id']
                try:
                    with self.subTest(index=index_name, doc_id=doc_id):
                        expected_doc = stored_results[index_name][doc_id]
                        actual_doc = self.client.index(index_name).get_document(doc_id)
                        self.assertEqual(expected_doc, actual_doc)

                except Exception as e:
                    test_failures.append((index_name, doc_id, traceback.format_exc()))

        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}, doc_id {doc_id}: {error}"
                for idx, doc_id, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")