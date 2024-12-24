import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.3.0')
class TestUpdateDocuments(BaseCompatibilityTestCase):
    structured_index_name = "update_doc_api_test_structured_index"

    indexes_to_test_on = [{
        "indexName": structured_index_name,
        "type": "structured",
        "allFields": [
            {"name": "img", "type": "image_pointer"},
            {"name": "title", "type": "text"},
            {"name": "label", "type": "text", "features": ["filter"]},
        ],
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
        "tensorFields": ["img", "title"]
    }]

    text_docs = [
        {
            "img": "https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/data/image0.jpg?raw=true",
            "title": "A lady taking a phote",
            "label": "lady",
            "_id": "1",
        },
        {
            "img": "https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/data/image1.jpg?raw=true",
            "title": "A plane flying in the sky",
            "label": "airplane",
            "_id": "2",
        },
    ]

    mappings = {}
    tensor_fields = {}

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
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            with self.subTest(indexName = index_name):
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs)
                else:
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs, mappings = self.mappings, tensor_fields = self.tensor_fields)

    def test_update_doc(self):
        self.logger.info(f"Running test_update_doc on {self.__class__.__name__}")

        test_failures = [] # this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                with self.subTest(indexName = index_name):
                    result = self.client.index(index_name).update_documents(
                        [{"_id": "1", "label": "person"}, {"_id": "2", "label": "plane"}]
                    )
                    self.logger.debug(f"Printing result {result}")
                    assert result["index_name"] == self.structured_index_name
                    assert len(result["items"]) == 2
                    assert result["errors"] == False
                    for item in result["items"]:
                        if item["_id"] in {"1", "2"}:
                            assert item["status"] == 200
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")