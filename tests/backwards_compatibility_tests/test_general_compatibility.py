import pytest

from base_compatibility_test_case import BaseCompatibilityTestCase
from marqo_test import MarqoTestCase
import marqo


@pytest.mark.marqo_version('2.11.0') #TODO: Check this again
class GeneralCompatibilityTest(BaseCompatibilityTestCase):

    image_model = 'open_clip/ViT-B-32/laion2b_s34b_b79k'
    multimodal_weights = {"image_field": 0.9, "text_field": 0.1}
    mappings = {
        "multimodal_field": {
            "type": "multimodal_combination",
            "weights": multimodal_weights,
        }
    }
    tensor_fields = ["multimodal_field", "text_field", "image_field"]
    structured_index_metadata = {
        "indexName": "structured-index-2-11",
        "type": "structured",
        "vectorNumericType": "float",
        "model": image_model,
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
                "dependentFields": multimodal_weights,
            },
        ],
        "tensorFields": tensor_fields,
        "annParameters": {
            "spaceType": "prenormalized-angular",
            "parameters": {"efConstruction": 512, "m": 16},
        },
    }

    unstructured_index_metadata = {
        "indexName": "unstructured-index-2-11",
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
            '_id': 'example_doc_1',
            'text_field': 'Man riding a horse',
            'image_field': 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg',
            'tags': ['man', 'horse'],
            'my_int': 1
        },
        {
            "_id": f"example_doc_2",
            "text_field": "Flying Plane",
            "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            'tags': ['plane'],
            'my_int': 2
        },
        {
            "_id": f"example_doc_3",
            "text_field": "Traffic light",
            "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
            'tags': ['light'],
            'caption': 'example_doc_3'
        },
        {
            "_id": f"example_doc_4",
            "text_field": "Red Bus",
            "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
            'tags': ['bus', 'red'],
            'caption': 'example_doc_4'
        }
    ]

    extra_docs = [
        {
            '_id': 'example_doc_5',
            'text_field': 'Woman looking at phone',
            'image_field': 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg',
            'tags': ['women', 'phone'],
            'my_int': 3
        },
        {
            "_id": f"example_doc_6",
            "text_field": "Woman skiing",
            "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg",
            'tags': ['ski'],
            'caption': 'example_doc_6'
        },
    ]
    indexes_to_test_on = [structured_index_metadata, unstructured_index_metadata]
    queries = ["travel", "horse light", "travel with plane"]
    search_methods = ["TENSOR", "LEXICAL", "HYBRID"]
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
        try:
            self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
            for index in self.indexes_to_test_on:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,
                                                                                   mappings=self.mappings,
                                                                                   tensor_fields=self.tensor_fields)
            self.logger.debug(f'Ran prepare method for {self.indexes_to_test_on} inside test class {self.__class__.__name__}')
        except Exception as e:
            self.logger.error(f"Exception occurred while adding documents {e}")
            raise e

        all_results = {}
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}

            # For each index, store results for different search methods
            for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
                if index.get("type") is not None and index.get("type") == 'structured':
                    if search_method == 'HYBRID':
                        result = self.client.index(index_name).search(q=query, search_method=search_method, hybrid_parameters=self.hybrid_search_params)
                    else:
                        result = self.client.index(index_name).search(q=query, search_method=search_method, searchable_attributes=self.searchable_attributes[search_method])
                else:
                    result = self.client.index(index_name).search(q=query, search_method=search_method)
                all_results[index_name][result_key] = result

        # store the result of search across all structured & unstructured indexes
        self.save_results_to_file(all_results)

    def test_search(self):
        """Run search queries and compare the results with the stored results."""

        stored_results = self.load_results_from_file()
        for index in self.indexes_to_test_on:
            index_name = index['indexName']

            # For each index, search for different queries and compare results
            for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
                if index.get("type") is not None and index.get("type") == 'structured':
                    if search_method == 'HYBRID':
                        result = self.client.index(index_name).search(q=query, search_method=search_method, hybrid_parameters=self.hybrid_search_params)
                    else:
                        result = self.client.index(index_name).search(q=query, search_method=search_method, searchable_attributes=self.searchable_attributes[search_method])
                else:
                    result = self.client.index(index_name).search(q=query, search_method=search_method)

                self._compare_search_results(stored_results[index_name][result_key], result)


    def _compare_search_results(self, expected_result, actual_result):
        """Compare two search results and assert if they match."""
        # We compare just the hits because the result contains other fields like processingTime which changes in every search API call.
        self.assertEqual(expected_result.get("hits"), actual_result.get("hits"), f"Results do not match. Expected: {expected_result}, Got: {actual_result}")