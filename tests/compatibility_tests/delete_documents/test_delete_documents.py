import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.0.0')
class TestDeleteDocuments(BaseCompatibilityTestCase):
    structured_index_name = "test_delete_api_structured_index"
    unstructured_index_name = "test_delete_api_unstructured_index"

    indexes_to_test_on = [{
        "indexName": structured_index_name,
        "type": "structured",
        "allFields": [
            {"name": "title", "type": "text"},
            {"name": "content", "type": "text"},
        ],
        "tensorFields": ["title", "content"],
        },
        {
            "indexName": unstructured_index_name,
            "type": "unstructured",
        }
    ]

    text_docs = [
        {
            "title": "The Travels of Marco Polo",
            "Description": "A 13th-century travelogue describing the travels of Polo",
            "Genre": "History",
            "_id": "article_602"
    },
        {
            "title": "Extravehicular Mobility Unit (EMU)",
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

        try:
            self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
            for index in self.indexes_to_test_on:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs)
                else:
                    self.client.index(index['indexName']).add_documents(documents = self.text_docs,
                                                                        tensor_fields = ["Description"])

            self.logger.debug(f"Finished running prepare method for test case: {self.__class__.__name__}")
        except Exception as e:
            raise Exception(f"Exception occurred while adding documents to indexes {e}") from e

    def test_delete_document(self):
        self.logger.info(f"Running test_delete_document on {self.__class__.__name__}")
        for index in self.indexes_to_test_on:
            index_name = index['indexName']

            result = self.client.index(index_name).delete_documents(["article_602", "article_591"])
            self.logger.debug(f"Result: {result}")
            assert result["index_name"] == index_name
            assert result["type"] == "documentDeletion"
            assert result["details"] == {
                "receivedDocumentIds": 2,
                "deletedDocuments": 2
            }
            assert len(result["items"]) == 2

            for item in result["items"]:
                if item["_id"] in { "article_602", "article_591"}:
                    assert item["status"] == 200
                    assert item["result"] == "deleted"