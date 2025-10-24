import traceback
from typing import List, Dict

import pytest
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase
from marqo.errors import MarqoWebError


@pytest.mark.marqo_version('2.0.0')
class TestDocumentAPIv2_0(BaseCompatibilityTestCase):
    """
    This class tests document API operations on both structured and unstructured indexes:
    - get_document
    - delete_documents
    - add_documents

    All operations are performed in a single test method to control execution sequence
    and avoid interference with other tests.
    """
    structured_index_name = "test_doc_api_structured_index"
    unstructured_index_name = "test_doc_api_unstructured_index"

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

    new_docs = [{
        "Title": "The Odyssey",
        "Description": "Ancient Greek epic poem attributed to Homer",
        "Genre": "Epic poetry",
        "_id": "article_701"
    },
        {
            "Title": "Quantum Computing Basics",
            "Description": "An introduction to quantum computing principles",
            "_id": "article_702",
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
        self.logger.info(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)

        self.logger.info(f'Feeding documents to {self.indexes_to_test_on}')

        errors = []  # Collect errors to report them at the end

        for index in self.indexes_to_test_on:
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.text_docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.text_docs,
                                                                                   tensor_fields=["Description",
                                                                                                  "Genre", "Title"])
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
            self.logger.error(
                f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")
        self.save_results_to_file(all_results)

    def test_document_api(self):
        """
        Tests all document API operations in a controlled sequence:
        1. get_document - Verifies initial documents can be retrieved
        2. delete_documents - Tests document deletion functionality
        3. add_documents - Tests adding new documents after deletion
        """
        for index in self.indexes_to_test_on:
            index_name = index['indexName']

            # Step 1: Test get_document
            with self.subTest(index=index_name, operation="get_document"):
                stored_results = self.load_results_from_file()

                for doc in self.text_docs:
                    doc_id = doc['_id']
                    expected_doc = stored_results[index_name][doc_id]
                    actual_doc = self.client.index(index_name).get_document(doc_id)
                    self.assertEqual(expected_doc, actual_doc)

            # Step 2: Test delete_documents
            with self.subTest(index=index_name, operation="delete_documents"):
                doc_ids = [doc['_id'] for doc in self.text_docs]

                # Delete documents
                delete_result = self.client.index(index_name).delete_documents(ids=doc_ids)
                self.assertEqual('succeeded', delete_result['status'])
                self.assertEqual(len(doc_ids), len(delete_result.get('items', [])))

                # Verify documents are deleted
                for doc_id in doc_ids:
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).get_document(doc_id)
                    self.assertEqual(404, e.exception.status_code)

            # Step 3: Test add_documents
            with self.subTest(index=index_name, operation="add_documents"):
                # Add new documents
                if index.get("type") is not None and index.get('type') == 'structured':
                    add_result = self.client.index(index_name=index_name).add_documents(documents=self.new_docs)
                else:
                    add_result = self.client.index(index_name=index_name).add_documents(
                        documents=self.new_docs,
                        tensor_fields=["Description", "Genre", "Title"]
                    )

                self.assertEqual(False, add_result['errors'])

                # Verify documents are added correctly
                for doc in self.new_docs:
                    doc_id = doc['_id']
                    retrieved_doc = self.client.index(index_name).get_document(doc_id)
                    self.assertEqual(doc, retrieved_doc)
