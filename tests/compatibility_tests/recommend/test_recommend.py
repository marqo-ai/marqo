import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.5.0')
class TestRecommend(BaseCompatibilityTestCase):

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
        "indexName": "test_recommend_api_structured_index",
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
        "indexName": "test_recommend_api_unstructured_index",
        "model": image_model,
        "treatUrlsAndPointersAsImages": True,
    }

    docs = [
        {
            '_id': f"example_doc_1",
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

    indexes_to_test_on = [structured_index_metadata, unstructured_index_metadata]

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
        errors = []  # Collect errors to report them at the end
        self.create_indexes(self.indexes_to_test_on)
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
                errors.append((index, str(e)))


        all_results = {}
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                all_results[index_name] = {}

                result = self.client.index(index_name).recommend(
                    documents = ["example_doc_1", "example_doc_2"],
                    limit = 10,
                    offset = 0,
                    show_highlights = True,
                    attributes_to_retrieve=["text_field", "tags", "caption"]
                )
                all_results[index_name] = result
            except Exception as e:
                errors.append((index, traceback.format_exc()))
        if errors:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")

        # store the result of search across all structured & unstructured indexes
        self.save_results_to_file(all_results)
        self.logger.debug(f'Ran prepare method for {self.indexes_to_test_on} inside test class {self.__class__.__name__}')

    def test_recommender(self):
        self.logger.info(f"Running test_recommender on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                actual_result = self.client.index(index_name).recommend(
                    documents = ["example_doc_1", "example_doc_2"],
                    limit = 10,
                    offset = 0,
                    show_highlights = True,
                    attributes_to_retrieve=["text_field", "tags", "caption"]
                )
                expected_result = stored_results[index_name]
                self.logger.debug(f"Printing expected_result {expected_result}")
                self.logger.debug(f"Printing actual_result {actual_result}")
                self.assert_search_results_match(expected_result, actual_result)
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))
        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}: {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def compare_search_results(self, expected, actual, ignore_fields=None):
        """
        Compare search results while ignoring order of items in hits and specific fields.

        Args:
            expected (dict): Expected search result
            actual (dict): Actual search result
            ignore_fields (list): Fields to ignore in comparison (e.g. ['processingTimeMs'])

        Returns:
            bool: True if results match ignoring order, False otherwise
        """
        if ignore_fields is None:
            ignore_fields = ['processingTimeMs', 'query', 'limit', 'offset']

        # Create copies and remove ignored fields
        expected_copy = expected.copy()
        actual_copy = actual.copy()

        for field in ignore_fields:
            expected_copy.pop(field, None)
            actual_copy.pop(field, None)

        # Get hits from both results
        expected_hits = expected_copy.pop('hits', [])
        actual_hits = actual_copy.pop('hits', [])

        # Compare non-hits parts
        if expected_copy != actual_copy:
            return False

        # Convert hits to tuples of sorted items for comparison
        def hit_to_comparable(hit):
            def convert_value(v):
                if isinstance(v, dict):
                    # Convert dictionary to sorted tuple of (key, converted_value) pairs
                    return tuple(sorted((k, convert_value(val)) for k, val in v.items()))
                elif isinstance(v, list):
                    # Convert list to tuple of converted values
                    return tuple(sorted(convert_value(x) for x in v))
                return v

            # Convert the hit dictionary into a sorted tuple of (key, converted_value) pairs
            return tuple(sorted(
                (k, convert_value(v))
                for k, v in hit.items()
            ))

        # Convert hits to sets of comparable tuples
        expected_set = {hit_to_comparable(hit) for hit in expected_hits}
        actual_set = {hit_to_comparable(hit) for hit in actual_hits}

        return expected_set == actual_set

    def assert_search_results_match(self, expected, actual, ignore_fields=None):
        """
        Assert that search results match while ignoring order.

        Args:
            expected (dict): Expected search result
            actual (dict): Actual search result
            ignore_fields (list): Fields to ignore in comparison
        """
        assert self.compare_search_results(expected, actual, ignore_fields), \
            f"Results do not match when comparing ignoring order.\nExpected: {expected}\nGot: {actual}"