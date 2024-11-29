import pytest
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
class TestAddDocuments(BaseCompatibilityTestCase):
    structured_index_name = "add_doc_api_test_structured_index"
    unstructured_index_name = "add_doc_api_test_unstructured_index"

    indexes_to_test_on = [{
        "indexName": structured_index_name,
        "type": "structured",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "normalizeEmbeddings": False,
        "allFields": [
            {"name": "title", "type": "text"},
            {"name": "content", "type": "text"},
            {"name": "int_field_1", "type": "int"},
            {"name": "float_field_1", "type": "float"},
            {"name": "long_field_1", "type": "long"},
            {"name": "double_field_1", "type": "double"},
            {"name": "array_int_field_1", "type": "array<int>"},
            {"name": "array_float_field_1", "type": "array<float>"},
            {"name": "array_long_field_1", "type": "array<long>"},
            {"name": "array_double_field_1", "type": "array<double>"},
            {"name": "custom_vector_field_1", "type": "custom_vector", "features": ["lexical_search", "filter"]},
        ],
        "tensorFields": ["title", "content", "custom_vector_field_1"],
    },
        {
        "indexName": unstructured_index_name,
        "type": "unstructured",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "normalizeEmbeddings": False,
    }]

    text_docs = [{
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing the travels of Polo",
        "Genre": "History"
    },
    {
        "Title": "Extravehicular Mobility Unit (EMU)",
        "Description": "The EMU is a spacesuit that provides environmental protection",
        "_id": "article_591",
        "Genre": "Science"
    }]
    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def prepare(self):
        self.logger.info(f"Creating indexes {self.indexes_to_test_on} in test case: {self.__class__.__name__}")
        self.create_indexes(self.indexes_to_test_on)

        try:
            self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
            for index in self.indexes_to_test_on:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs)
                else:
                    self.client.index(index['indexName']).add_documents(documents = self.text_docs,
                                                                        tensor_field = ["Description"])

            self.logger.debug(f"Finished running prepare method for test case: {self.__class__.__name__}")
        except Exception as e:
            raise e

        all_results = {}

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}

            for doc in self.text_docs:
                doc_id = doc['_id']
                all_results[index_name][doc_id] = self.client.index(index_name).get_document(doc_id)

        self.save_results_to_file(all_results)

    def test_add_doc(self):
        self.logger.info(f"Running test_add_doc on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        for index in self.indexes_to_test_on:
            index_name = index['indexName']

            for doc in self.text_docs:
                doc_id = doc['_id']
                expected_doc = stored_results[index_name][doc_id]
                self.logger.debug(f"Printing expected doc {expected_doc}")
                actual_doc = self.client.index(index_name).get_document(doc_id)
                self.logger.debug(f"Printing actual doc {expected_doc}")

                self.assertEqual(expected_doc, actual_doc)