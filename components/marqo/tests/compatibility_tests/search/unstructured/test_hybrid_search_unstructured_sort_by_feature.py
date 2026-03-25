import traceback
from inspect import trace

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.11.0')
class TestHybridSearchUnstructuredSortByFeature(BaseCompatibilityTestCase):
    """
    In 2.21.0, the sortBy feature update the index schema. The schema will be updated if we try to
    add documents to an index with new fields that requires a schema update. The update is supposed to be
    backward compatible, meaning that the index should still work with older versions of Marqo. Therefore, this test
    checks if te old indexes with updated schema can still search correctly.
    """

    image_model = 'open_clip/ViT-B-32/laion2b_s34b_b79k'
    multimodal_weights = {"image_field": 0.9, "text_field": 0.1}
    mappings = {
        "multimodal_field": {
            "type": "multimodal_combination",
            "weights": multimodal_weights,
        }
    }
    tensor_fields = ["multimodal_field", "text_field", "image_field"]

    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index_hybrid_sort_by_feature",
        "model": image_model,
        "treatUrlsAndPointersAsImages": True,
    }

    hybrid_search_params = {
        "retrievalMethod": "disjunction",
        "rankingMethod": "rrf",
        "alpha": 0.3,
        "rrfK": 60,
        "searchableAttributesLexical": ["text_field"],
        "searchableAttributesTensor": ['image_field', 'multimodal_field'],
        "scoreModifiersTensor": {
            "add_to_score": [{"field_name": "my_int", "weight": 0.01}]
        },
        "scoreModifiersLexical": {
            "add_to_score": [{"field_name": "my_int", "weight": 0.01}]
        },
    }

    docs = [
        {
            '_id': f"example_doc_1",
            'text_field': 'Man riding a horse',
            'image_field': 'https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg',
            'tags': ['man', 'horse'],
            'my_int': 1
        },
        {
            "_id": f"example_doc_2",
            "text_field": "Flying Plane",
            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            'tags': ['plane'],
            'my_int': 2
        },
        {
            "_id": f"example_doc_3",
            "text_field": "Traffic light",
            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg",
            'tags': ['light'],
            'caption': 'example_doc_3'
        },
        {
            "_id": f"example_doc_4",
            "text_field": "Red Bus",
            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg",
            'tags': ['bus', 'red'],
            'caption': 'example_doc_4'
        }
    ]

    extra_docs = [
        {
            '_id': f"example_doc_5",
            'text_field': 'Woman looking at phone',
            'image_field': 'https://marqo-assets.s3.amazonaws.com/tests/images/image0.jpg',
            'tags': ['women', 'phone'],
            'my_int': 3
        },
        {
            "_id": f"example_doc_6",
            "text_field": "Woman skiing",
            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
            'tags': ['ski'],
            'caption': 'example_doc_6'
        },
    ]
    indexes_to_test_on = [unstructured_index_metadata]
    queries = ["travel", "horse light", "travel with plane"]
    search_methods = ["HYBRID"]
    result_keys = search_methods # Set the result keys to be the same as search methods for easy comparison
    searchable_attributes = {"TENSOR": ['image_field', 'multimodal_field'], "LEXICAL": ['text_field']}

    # We need to set indexes_to_delete variable in an overriden tearDownClass() method
    # So that when the test method has finished running, pytest is able to delete the indexes added in
    # prepare method of this class
    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().setUpClass()

    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        self.logger.debug(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)
        add_doc_errors = []  # Collect add_doc_errors to report them at the end

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,
                                                                               mappings=self.mappings,
                                                                               tensor_fields=self.tensor_fields)
            except Exception as e:
                add_doc_errors.append((index, traceback.format_exc()))

        all_results = {}
        search_errors = []  # Collect search errors to report them at the end
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}

            self.client.index(index_name).add_documents(
                documents=[{
                    "_id": f"extra_doc_1",
                    "extra_text_field": "Extra document for schema update",
                }], tensor_fields=["extra_text_field"]
            ) # Add extra documents to the index to trigger a schema update

            # Delete the extra document to ensure it does not affect the search results
            self.client.index(index_name).delete_documents(["extra_doc_1"])

            # For each index, store results for different search methods
            for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
                try:
                    if index.get("type") is not None and index.get("type") == 'structured':
                        if search_method == 'HYBRID':
                            result = self.client.index(index_name).search(q=query, search_method=search_method, hybrid_parameters=self.hybrid_search_params)
                        else:
                            result = self.client.index(index_name).search(q=query, search_method=search_method, searchable_attributes=self.searchable_attributes[search_method])
                    else:
                        result = self.client.index(index_name).search(q=query, search_method=search_method)
                    all_results[index_name][result_key] = result
                except Exception as e:
                    search_errors.append((query, search_method, index_name, traceback.format_exc()))
        if add_doc_errors:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in add_doc_errors
            ])
            self.logger.error(f"Some subtests failed: \n {failure_message}. When the corresponding test runs for this index, it is expected to fail")
        if search_errors:
            failure_message = "\n".join([
                f"Failure in query {query}, search_method {search_method}, idx: {idx} : {error}"
                for query, search_method, idx, error in search_errors
            ])
            self.logger.error(f"Some subtests failed:\n {failure_message}. When the corresponding test runs for this index, it is expected to fail")
        self.save_results_to_file(all_results)

    def test_search(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        for index in self.indexes_to_test_on:
            index_name = index['indexName']

            # For each index, search for different queries and compare results
            for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
                try:
                    if index.get("type") is not None and index.get("type") == 'structured':
                        if search_method == 'HYBRID':
                            result = self.client.index(index_name).search(q=query, search_method=search_method, hybrid_parameters=self.hybrid_search_params)
                        else:
                            result = self.client.index(index_name).search(q=query, search_method=search_method, searchable_attributes=self.searchable_attributes[search_method])
                    else:
                        result = self.client.index(index_name).search(q=query, search_method=search_method)
                    self._compare_search_results(stored_results[index_name][result_key], result)

                except Exception as e:
                    test_failures.append((query, search_method, index_name, traceback.format_exc()))

            if test_failures:
                failure_message = "\n".join([
                    f"Failure in query {query}, search_method {search_method}, idx: {idx} : {error}"
                    for query, search_method, idx, error in test_failures
                ])
                self.fail(f"Some subtests failed:\n{failure_message}")