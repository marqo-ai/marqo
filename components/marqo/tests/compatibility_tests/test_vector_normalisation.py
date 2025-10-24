import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.13.0')
@pytest.mark.skip_marqo_version('2.17.0')
class CompatibilityTestVectorNormalisation(BaseCompatibilityTestCase):
    text_index_with_normalize_embeddings_true = "add_doc_api_test_structured_index_with_normalize_embeddings_true"

    DEFAULT_DIMENSIONS = 384
    custom_vector = [1.0 for _ in range(DEFAULT_DIMENSIONS)]
    expected_custom_vector_after_normalization = [0.05103103816509247 for _ in range(DEFAULT_DIMENSIONS)]
    index_metadata = {
                "indexName": text_index_with_normalize_embeddings_true,
                "type": "structured",
                "normalizeEmbeddings": True,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
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
                    {"name": "custom_vector_field_1", "type": "custom_vector",
                     "features": ["lexical_search", "filter"]},
                ],
                "tensorFields": ["title", "content", "custom_vector_field_1"],
            }

    indexes_to_test_on = [text_index_with_normalize_embeddings_true]

    # We need to set indexes_to_delete variable in an overriden tearDownClass() method
    # So that when the test method has finished running, pytest is able to delete the indexes added in
    # prepare method of this class
    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().tearDownClass()


    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().setUpClass()

    def prepare(self):
        # Create structured and unstructured indexes and add some documents, set normalise embeddings to true
        # Add documents
        self.logger.debug(f"Creating indexes {self.text_index_with_normalize_embeddings_true}")
        self.create_indexes([self.index_metadata])

        try:
            add_docs_res_normalized = self.client.index(index_name=self.text_index_with_normalize_embeddings_true).add_documents(
                documents=[
                    {
                        "custom_vector_field_1": {
                            "content": "custom vector text",
                            "vector": self.custom_vector,
                        },
                        "content": "normal text",
                        "_id": "doc1",
                    },
                    {
                        "content": "second doc",
                        "_id": "doc2"
                    }
                ])

            result = self.client.index(index_name = self.text_index_with_normalize_embeddings_true).get_document(document_id="doc1", expose_facets=True)
            self.logger.debug(f"Added documents to index: {add_docs_res_normalized}")
            self.save_results_to_file(result)
            self.logger.debug(f'Ran prepare mode test for {self.text_index_with_normalize_embeddings_true} inside test class {self.__class__.__name__}')
        except Exception as e:
            self.logger.error(f"Exception occurred while adding documents / getting documents to / from index {self.text_index_with_normalize_embeddings_true}. When the corresponding test runs, it is expected to fail."
                              f"Exception traceback was: {traceback.format_exc()}")

    def test_custom_vector_doc_in_normalized_embedding_true(self):
        # This runs on to_version
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions
        result_from_prepare_mode = self.load_results_from_file()
        for index_name in self.indexes_to_test_on:
            with self.subTest(index=index_name):
                self.logger.debug(f"Processing index: {index_name}")
                try:
                    with self.subTest(index=index_name):
                        doc_res_normalized = self.client.index(index_name).get_document(
                        document_id="doc1",
                        expose_facets=True)
                        self.assertEqual(doc_res_normalized["custom_vector_field_1"], "custom vector text")
                        self.assertEqual(doc_res_normalized['_tensor_facets'][0]["custom_vector_field_1"], "custom vector text")
                        self._compare_results(result_from_prepare_mode, doc_res_normalized)
                except Exception as e:
                    test_failures.append((index_name, traceback.format_exc()))
        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}: {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def _compare_results(self, expected_result, actual_result):
        """Compare two search results and assert if they match."""
        self.assertEqual(expected_result, actual_result, f"Results do not match. Expected: {expected_result}, Got: {actual_result}")