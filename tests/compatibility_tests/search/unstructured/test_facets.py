import traceback
from inspect import trace

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.18.0')
class TestFacets(BaseCompatibilityTestCase):
    image_model = 'open_clip/ViT-B-32/laion2b_s34b_b79k'
    tensor_fields = ["title", "description"]

    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index",
        "model": image_model,
        "treatUrlsAndPointersAsImages": True,
    }

    EXAMPLE_FASHION_DOCUMENTS = [
        {
            "_id": "1",
            "title": "Slim Fit Denim Jacket",
            "brand": "SnugNest",
            "description": "A timeless piece with a modern slim-fit design, perfect for casual layering.",
            "color": "yellow",
            "size": "S",
            "style": "casual",
            "price": 83.42
        },
        {
            "_id": "2",
            "title": "Classic Cotton Shirt",
            "brand": "SnugNest",
            "description": "Comfortable and breathable cotton shirt suitable for everyday wear.",
            "color": "red",
            "size": "M",
            "style": "partywear",
            "price": 49.03
        },
        {
            "_id": "3",
            "title": "High-Waisted Skirt",
            "brand": "PulseWear",
            "description": "Elegant skirt with a high waistline and flattering silhouette.",
            "color": "coral",
            "size": "L",
            "style": "streetwear",
            "price": 1.2
        },
        {
            "_id": "4",
            "title": "Knitted Winter Sweater",
            "brand": "SprintX",
            "description": "Chunky knit sweater designed for warmth and comfort in cold seasons.",
            "color": "red",
            "size": "Free",
            "style": "loungewear",
            "price": 92.99
        },
        {
            "_id": "5",
            "title": "Casual Linen Trousers",
            "brand": "PulseWear",
            "description": "Relaxed-fit trousers crafted from lightweight linen for maximum comfort.",
            "color": "charcoal",
            "size": "M",
            "style": "partywear",
            "price": 88.14
        },
        {
            "_id": "6",
            "title": "Embroidered Kurta",
            "brand": "RetroHue",
            "description": "Traditional kurta with intricate embroidery for festive occasions.",
            "color": "green",
            "size": "S",
            "style": "streetwear",
            "price": 81.33
        },
        {
            "_id": "7",
            "title": "Floral Summer Dress",
            "brand": "SnugNest",
            "description": "Breezy and lightweight dress ideal for sunny summer days.",
            "color": "green",
            "size": "XS",
            "style": "streetwear",
            "price": 28.71
        },
        {
            "_id": "8",
            "title": "Athletic Running Shorts",
            "brand": "PulseWear",
            "description": "Performance shorts made from moisture-wicking fabric for workouts.",
            "color": "green",
            "size": "Free",
            "style": "biker",
            "price": 73.88
        },
        {
            "_id": "9",
            "title": "Hooded Windbreaker",
            "brand": "CozyCore",
            "description": "Windproof and waterproof jacket with adjustable hood.",
            "color": "charcoal",
            "size": "S",
            "style": "streetwear",
            "price": 55.54
        },
        {
            "_id": "10",
            "title": "Fleece Zip-Up Hoodie",
            "brand": "SnugNest",
            "description": "Super soft fleece hoodie for a relaxed and cozy look.",
            "color": "gray",
            "size": "M",
            "style": "loungewear",
            "price": 49.3
        }
    ]
    for doc in EXAMPLE_FASHION_DOCUMENTS:
        doc["tags"] = [
            f"color:{doc['color']}",
            f"size:{doc['size']}",
            f"style:{doc['style']}",
            f"brand:{doc['brand']}",
        ]
    queries = ["travel", "horse light", "travel with plane"]
    search_methods = [("tensor", "tensor"), ("lexical", "lexical"), ("disjunction", "rrf"), ("tensor", "lexical"), ("lexical", "tensor")]
    result_keys = search_methods # Set the result keys to be the same as search methods for easy comparison

    # We need to set indexes_to_delete variable in an overriden tearDownClass() method
    # So that when the test method has finished running, pytest is able to delete the indexes added in
    # prepare method of this class
    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [cls.unstructured_index_metadata['indexName']]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [cls.unstructured_index_metadata['indexName']]
        super().setUpClass()

    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        self.logger.debug(f"Creating index {self.unstructured_index_metadata['indexName']}")
        self.create_indexes([self.unstructured_index_metadata])
        errors = []  # Collect errors to report them at the end

        self.logger.debug(f'Feeding documents to index')
        try:
            self.client.index(index_name=self.unstructured_index_metadata['indexName']).add_documents(documents=self.EXAMPLE_FASHION_DOCUMENTS,
                                                                           tensor_fields=self.tensor_fields)
        except Exception as e:
            errors.append((self.unstructured_index_metadata, traceback.format_exc()))

        if errors:
            failure_message = "\n".join([
                f"Failure while Feeding documents to idx: {idx} : {error}"
                for idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")

        # store the result of search across all structured & unstructured indexes
        all_results = self.do_searches()
        self.save_results_to_file(all_results)


    def do_searches(self):
        all_results = {}
        errors = []  # Redefining to Collect errors related to search to report them at the end
        # Loop through queries, search methods, and result keys to populate unstructured_results
        index_name = self.unstructured_index_metadata['indexName']
        all_results[index_name] = {}

        # For each index, store results for different search methods
        for query, search_method, result_key in zip(self.queries, self.search_methods, self.result_keys):
            try:
                result = self.client.index(index_name).search(q=query, search_method="HYBRID", hybrid_parameters={
                    "searchMethod": search_method[0],
                    "retrievalMethod": search_method[1],
                },
                facets={
                    "fields": {
                        "color": {"type": "string"},
                        "brand": {"type": "string", "maxResults": 2},
                        "price": {"type": "number"},
                        "tags": {"type": "array"},
                    },
                    "maxResults": 1000,
                    "maxDepth": 1000,
                }
            )
                all_results[index_name][result_key] = result
            except Exception as e:
                errors.append((query, search_method, index_name, traceback.format_exc()))

        if errors:
            failure_message = "\n".join([
                f"Failure in query {query}, search_method {search_method}, idx: {idx} : {error}"
                for query, search_method, idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")

        return all_results

    def test_search(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        all_results = self.do_searches()
        for index_name, results in all_results.items():
            for search_method, result in results.items():
                # Compare the actual result with the expected result
                expected_result = stored_results[index_name][search_method]
                try:
                    self._compare_search_results(expected_result, result)
                except AssertionError as e:
                    test_failures.append((index_name, search_method, str(e)))
                except Exception as e:
                    test_failures.append((index_name, search_method, str(e)))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in query {query}, search_method {search_method}, idx: {idx} : {error}"
                for query, search_method, idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def assert_dict_almost_equal(self, d1, d2, places=1):
        """Compare two dictionaries with numeric values allowing for small differences."""
        self.assertEqual(d1.keys(), d2.keys(), "Dictionaries have different keys")
        for key in d1:
            if isinstance(d1[key], dict):
                self.assert_dict_almost_equal(d1[key], d2[key], places)
            elif isinstance(d1[key], (int, float)) and isinstance(d2[key], (int, float)):
                self.assertAlmostEqual(d1[key], d2[key], places=places,
                                       msg=f"Values differ for key {key}: {d1[key]} != {d2[key]}")
            else:
                self.assertEqual(d1[key], d2[key],
                                 f"Values differ for key {key}: {d1[key]} != {d2[key]}")

    def _compare_search_results(self, expected_result, actual_result):
        """Compare two search results and assert if they match."""
        # We compare just the hits because the result contains other fields like processingTime which changes in every search API call.
        self.assertEqual(expected_result.get("hits"), actual_result.get("hits"), f"Results do not match. Expected: {expected_result}, Got: {actual_result}")
        # Use this for facets due to numeric values like "avg" that can have small rounding errors.
        self.assert_dict_almost_equal(expected_result["facets"], actual_result["facets"], places=2)