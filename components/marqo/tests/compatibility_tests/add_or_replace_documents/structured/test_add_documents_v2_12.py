import traceback
from inspect import trace

import pytest
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.12.0')
class TestAddDocumentsv2_12(BaseCompatibilityTestCase):
    structured_index_name = "test_add_doc_api_structured_index_2_12_0"

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
                {"name": "image_field", "type": "image_pointer"},
                {"name": "video_field_1", "type": "video_pointer"}, #TODO: write this example for video_pointer and audio_pointers
                {"name": "audio_field_1", "type": "audio_pointer"},
                {"name": "text_field_3", "type": "text", "features": ["lexical_search"]},
            ],
            "tensorFields": ["video_field_1", "audio_field_1", "image_field", "text_field_3"],
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {"efConstruction": 512, "m": 16},
            }
        }]

    documents = [
        {
            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
            "_id": "3"
        },
        {
            "text_field_3": "hello there Padawan. Today you will begin your training to be a Jedi",
            "_id": "4"
        },
    ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().setUpClass()

    def prepare(self):
        self.create_indexes(self.indexes_to_test_on)

        errors = []  # Collect errors to report them at the end

        for index in self.indexes_to_test_on:
            self.logger.info(
                f"Feeding document to index {index.get('indexName')} in test case: {self.__class__.__name__}")
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name = index['indexName']).add_documents(documents = self.documents)
            except Exception as e:
                errors.append((index, traceback.format_exc()))

        all_results = {}

        for index in self.indexes_to_test_on:
            self.logger.debug(f'Getting documents from {index.get("indexName")}')
            index_name = index['indexName']
            all_results[index_name] = {}

            for doc in self.documents:
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
            for doc in self.documents:
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