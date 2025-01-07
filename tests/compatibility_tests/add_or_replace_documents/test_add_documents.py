import traceback

import pytest
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
class TestAddDocumentsv2_0(BaseCompatibilityTestCase):
    """
    This class tests the add_documents API on both structured and unstructured indexes
    """
    structured_index_name = "test_add_doc_api_structured_index"
    unstructured_index_name = "test_add_doc_api_unstructured_index"

    indexes_to_test_on = [{
        "indexName": structured_index_name,
        "type": "structured",
        "normalizeEmbeddings": True,
        "allFields": [
            {"name": "Title", "type": "text"},
            {"name": "Description", "type": "text"},
            {"name": "Genre", "type": "text"},
        ],
        "tensorFields": ["Title", "Description", "Genre"],
    },
        {
        "indexName": unstructured_index_name,
        "type": "unstructured",
        "normalizeEmbeddings": True,
    }]

    text_docs = [{
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing the travels of Polo",
        "Genre": "History",
        "_id": "article_602"
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
                        self.logger.debug(f"Printing expected doc {expected_doc}")
                        actual_doc = self.client.index(index_name).get_document(doc_id)
                        self.assertEqual(expected_doc, actual_doc)

                except Exception as e:
                    test_failures.append((index_name, doc_id, traceback.format_exc()))

        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index_name: {idx}, doc_id: {doc_id}: {error}"
                for idx, doc_id, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n {failure_message}")