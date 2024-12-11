import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
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

    indexes_to_test_on = [structured_index_metadata, unstructured_index_metadata]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        self.logger.info(f"Creating indexes {self.indexes_to_test_on}")
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
                errors.append((index, str(e)))
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
                self._compare_hits(expected_result, actual_result)
            except Exception as e:
                test_failures.append((index_name, str(e)))
        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}, doc_id {doc_id}: {error}"
                for idx, doc_id, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def _compare_hits(self, expected_result, actual_result):
        self.assertEqual(expected_result.get("hits"), actual_result.get("hits"), f"Results do not match. Expected: {expected_result}, Got: {actual_result}")