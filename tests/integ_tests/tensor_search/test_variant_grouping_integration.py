import os
import random
import time
from typing import Dict, List, Any
from unittest import mock

import pytest

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.marqo_index import Model, VariantGrouping
from marqo.tensor_search import tensor_search
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.api_models import VariantGroupingParameters
from marqo.tensor_search.enums import SearchMethod


class TestVariantGroupingIntegration(MarqoTestCase):
    """Integration tests for variant grouping functionality using Vespa grouping queries"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes - for now, use this and handle grouping field issue differently
        unstructured_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            variant_grouping=VariantGrouping(variantGroupField="product_id", minGroup=20),
        )

        cls.indexes = cls.create_indexes([
            unstructured_index,
        ])

        cls.unstructured_index = cls.indexes[0]

    def setUp(self) -> None:
        super().setUp()

        # Add variant documents
        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_index.name,
                docs=self._create_variant_test_data(),
                use_existing_tensors=False,
                tensor_fields=['variant_title'],
            )
        )
        self.assertFalse(res.errors)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu",
            "MARQO_MAX_CPU_MODEL_MEMORY": "15"
        })
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def _create_variant_test_data(self) -> List[Dict[str, Any]]:
        """Create test data with multiple variants per product for testing grouping"""
        products = [
            {
                "product_id": "nike_air_force",
                "title": "Nike Air Force 1",
                "brand": "Nike",
                "collection": ["Sneakers", "On Sale"],
                "description": "Classic basketball shoe",
                "price": 100.0,
            },
            {
                "product_id": "adidas_stan_smith",
                "title": "Adidas Stan Smith",
                "brand": "Adidas",
                "collection": ["Sneakers", "On Sale"],
                "description": "Iconic tennis shoe",
                "price": 80.0,
            },
            {
                "product_id": "converse_chuck",
                "title": "Converse Chuck Taylor All Star",
                "brand": "Converse",
                "collection": ["Sneakers", "New Arrival"],
                "description": "Classic canvas shoe",
                "price": 60.0,
            },
        ]

        color_variants = ['white', 'black', 'navy', 'red', 'blue', 'pink']
        size_variants = ['9', '10', '11', '12', '13']

        all_variants = []

        for product in products:
            for color in color_variants:
                for size in size_variants:
                    all_variants.append({
                        **product,
                        'color': color,
                        'size': size,
                        'variant_title': f'{product["title"]} - {color} - Size: {size}',
                        'stock': 0 if color == 'white' and size == '10' else 5,
                        '_id': f'{product["product_id"]}_{color}_{size}'
                    })

        return all_variants

    def test_search_without_grouping_should_return_all_variants(self):
        for search_method in [SearchMethod.HYBRID]:
            with self.subTest(search_method=search_method):
                # Search without grouping - should return all variants
                regular_results = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_index.name,
                    search_method=search_method,
                    text="sneakers",
                    result_count=3,
                    ensure_diversity=True
                    # hybrid_parameters=HybridParameters(
                    #     retrievalMethod="lexical",
                    #     rankingMethod="tensor"
                    # )
                )

                # Verify we get multiple variants per product
                self.assertEqual(len(regular_results["hits"]), 8)  # All 8 variants

                # Count unique products in regular results
                regular_product_ids = {hit["product_id"] for hit in regular_results["hits"]}
                self.assertEqual(len(regular_product_ids), 3)  # 3 unique products

    def test_basic_variant_grouping(self):
        """Test that grouping returns one result per product_id"""
        for search_method in [SearchMethod.HYBRID]:
            with self.subTest(search_method=search_method):
                grouped_results = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_index.name,
                    text="sneakers",
                    result_count=5,
                    offset=0,
                    search_method=search_method,
                    ensure_diversity=True,
                    # variant_grouping=VariantGroupingParameters(
                    #     maxVariantsPerGroup=2,
                    #     variantGroupField="product_id"
                    # )
                )

                print([f'{hit["_id"]}: {hit["_score"]}' for hit in grouped_results["hits"]])

                # Verify grouping returns one result per product
                self.assertEqual(len(grouped_results["hits"]), 3, f'assert error with {search_method}')  # One per product
                grouped_product_ids = {hit["product_id"] for hit in grouped_results["hits"]}
                self.assertEqual(len(grouped_product_ids), 3)  # All unique


    def test_variant_specific_search_with_grouping(self):
        """Test that variant-specific queries work correctly with grouping"""
        
        # Search for specific color
        white_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="white sneakers",
            result_count=10
        )
        
        # Verify we get some results (semantic search may not return only white variants)
        self.assertGreater(len(white_results["hits"]), 0)
                
        # Test with grouping enabled
        from marqo.tensor_search.models.api_models import VariantGroupingParameters
        from marqo.tensor_search.enums import SearchMethod
        white_grouped_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="white sneakers", 
            result_count=10,
            search_method=SearchMethod.HYBRID,
            variant_grouping=VariantGroupingParameters(
                maxVariantsPerGroup=1,
                variantGroupField="product_id"
            )
        )
        
        # Should return max one variant per product
        product_ids_seen = set()
        for hit in white_grouped_results["hits"]:
            self.assertNotIn(hit["product_id"], product_ids_seen)
            product_ids_seen.add(hit["product_id"])

    def test_variant_grouping_with_filters(self):
        """Test grouping works correctly with filter queries"""
        # Filter by price range
        price_filtered_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="sneakers",
            filter="price:[80 TO 100]",
            result_count=10
        )
        
        # Verify price filtering works
        for hit in price_filtered_results["hits"]:
            self.assertGreaterEqual(hit["price"], 80.0)
            self.assertLessEqual(hit["price"], 100.0)
            
        # Test with grouping + filters
        from marqo.tensor_search.models.api_models import VariantGroupingParameters
        from marqo.tensor_search.enums import SearchMethod
        grouped_filtered_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="sneakers",
            filter="price:[80 TO 100]", 
            result_count=10,
            search_method=SearchMethod.HYBRID,
            variant_grouping=VariantGroupingParameters(
                maxVariantsPerGroup=1,
                variantGroupField="product_id"
            )
        )
        
        # Verify grouping + filtering works together
        product_ids_seen = set()
        for hit in grouped_filtered_results["hits"]:
            self.assertNotIn(hit["product_id"], product_ids_seen)
            product_ids_seen.add(hit["product_id"])
            self.assertGreaterEqual(hit["price"], 80.0)
            self.assertLessEqual(hit["price"], 100.0)

    def test_variant_grouping_respects_max_variants_setting(self):
        """Test that max_variants_per_product setting is respected"""
        
        # Test different max_variants_per_product values
        from marqo.tensor_search.models.api_models import VariantGroupingParameters
        from marqo.tensor_search.enums import SearchMethod
        for max_variants in [1, 2, 3]:
            grouped_results = tensor_search.search(
                config=self.config,
                index_name=self.unstructured_index.name,
                text="Nike Air Force",
                result_count=10,
                search_method=SearchMethod.HYBRID,
                variant_grouping=VariantGroupingParameters(
                    maxVariantsPerGroup=max_variants,
                    variantGroupField="product_id"
                )
            )
            
            # Count variants for Nike Air Force product
            nike_variants = [hit for hit in grouped_results["hits"] 
                            if hit["product_id"] == "nike_air_force"]
            self.assertLessEqual(len(nike_variants), max_variants)
        
        # For now, just verify we have the test data
        all_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="Nike Air Force",
            result_count=10
        )
        nike_variants = [hit for hit in all_results["hits"] 
                        if hit.get("product_id") == "nike_air_force"]
        self.assertEqual(len(nike_variants), 3)  # Should have 3 Nike variants

    def test_variant_grouping_performance(self):
        """Test that grouping doesn't significantly impact search performance"""
        
        # Measure regular search performance
        start_time = time.perf_counter()
        regular_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="sneakers",
            result_count=10
        )
        regular_latency = time.perf_counter() - start_time
        
        # Verify we got results
        self.assertGreater(len(regular_results["hits"]), 0)
        
        # Measure grouped search performance 
        from marqo.tensor_search.models.api_models import VariantGroupingParameters
        from marqo.tensor_search.enums import SearchMethod
        start_time = time.perf_counter()
        grouped_results = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_index.name,
            text="sneakers",
            result_count=10,
            search_method=SearchMethod.HYBRID,
            variant_grouping=VariantGroupingParameters(
                maxVariantsPerGroup=1,
                variantGroupField="product_id"
            )
        )
        grouped_latency = time.perf_counter() - start_time
        
        # Grouping should not add more than 50% latency overhead
        self.assertLess(grouped_latency, regular_latency * 1.5)
        
        # For now just verify regular search is reasonably fast (< 1 second)
        self.assertLess(regular_latency, 1.0, "Regular search should complete in under 1 second")


if __name__ == "__main__":
    pytest.main([__file__])