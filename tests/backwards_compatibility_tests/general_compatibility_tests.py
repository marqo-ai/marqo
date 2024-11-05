import pytest

from base_test_case import BaseCompatibilityTestCase
from marqo_test import MarqoTestCase
import marqo


@pytest.mark.marqo_version('2.11.0') #TODO: Check this again
class GeneralCompatibilityCompatibilityTest(BaseCompatibilityTestCase):

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

    hybrid_search_prarms = {
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
    result_keys = ['unstructured_tensor', 'unstructured_lexical', 'unstructured_hybrid']


    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = marqo.Client(**cls.client_settings)
        print("Client has been initialized:", cls.client)
        print(f"Creating indexes {cls.indexes_to_test_on}")
        cls.create_indexes(cls.indexes_to_test_on)
        cls.indexes_to_delete = [cls.structured_index_metadata['indexName'],
                                 cls.unstructured_index_metadata['indexName']]

    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        try:
            print(f'Feeding documents to {self.indexes_to_test_on}')
            for index in self.indexes_to_test_on:
                if index['type'] == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,
                                                                                   mappings=self.mappings,
                                                                                   tensor_fields=self.tensor_fields)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
        except Exception as e:
            print(f"Exception occurred while adding documents {e}")

        unstructured_results = {}
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
            result = self.client.index(self.unstructured_index_metadata['indexName']).search(q=query, search_method=search_method)
            unstructured_results[result_key] = result

        # store the result of search across all structured & unstructured indexes
        self.save_results_to_file(unstructured_results)

    def test_search(self):
        """Run search queries and compare the results with the stored results."""

        stored_results = self.load_results_from_file()

        for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
            # Run the search again with the same parameters
            current_result = self.client.index(index_name=self.unstructured_index_metadata['indexName']).search(q=query, search_method=search_method)
            # Compare the current result with the stored result
            self.compare_results(stored_results[result_key], current_result)

    def compare_results(self, expected_result, actual_result):
        """Compare two search results and assert if they match."""
        assert expected_result == actual_result, f"Results do not match. Expected: {expected_result}, Got: {actual_result}"
