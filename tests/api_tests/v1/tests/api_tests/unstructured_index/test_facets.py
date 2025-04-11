import uuid
from marqo.client import Client
from marqo.errors import MarqoWebError
from tests.marqo_test import MarqoTestCase, EXAMPLE_FASHION_DOCUMENTS

class TestFacets(MarqoTestCase):
    text_index_name = "api_test_structured_index_text" + str(uuid.uuid4()).replace('-', '')
    unstructured_text_index_name = "api_test_unstructured_index_text" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)
        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "text_field_1", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_2", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_3", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "add_field_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "add_field_2", "type": "float", "features": ["score_modifier"]},
                    {"name": "mult_field_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "mult_field_2", "type": "float", "features": ["score_modifier"]}
                ],
                "tensorFields": ["text_field_1", "text_field_2", "text_field_3"]
            },
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            }
        ])
        cls.indexes_to_delete = [cls.text_index_name, cls.unstructured_text_index_name]

    def setUp(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)
        self.client.index(self.unstructured_text_index_name).add_documents(
            EXAMPLE_FASHION_DOCUMENTS,
            tensor_fields=["title", "description"]
        )

    def test_facets_structured_index_fails(self):
        """Verify facets fail on structured indexes"""
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.text_index_name).search(
                "shirt",
                search_method="HYBRID",
                facets={"fields": {"color": {"type": "string"}}}
            )
        self.assertIn("Facets are only supported for unstructured indexes", str(e.exception))

    def test_facets_non_hybrid_search_fails(self):
        """Verify facets only work with HYBRID search method"""
        for search_method in ["lexical", "tensor"]:
            with self.assertRaises(MarqoWebError) as e:
                self.client.index(self.unstructured_text_index_name).search(
                    "shirt",
                    search_method=search_method,
                    facets={"fields": {"color": {"type": "string"}}}
                )
            self.assertIn("Facets can only be provided for 'HYBRID' search", str(e.exception))

    def test_single_string_facet(self):
        """Test getting a single string facet"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            facets={"fields": {"color": {"type": "string"}}}
        )
        self.assertIn("facets", res)
        self.assertIn("color", res["facets"])
        self.assertGreater(len(res["facets"]["color"]), 0)

    def test_multiple_facets(self):
        """Test getting multiple facets simultaneously"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            facets={
                "fields": {
                    "color": {"type": "string"},
                    "brand": {"type": "string"},
                    "style": {"type": "string"}
                }
            }
        )
        self.assertIn("facets", res)
        self.assertEqual(len(res["facets"]), 3)

    def test_number_facet(self):
        """Test getting numeric facet with metrics"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            facets={"fields": {"price": {"type": "number"}}}
        )
        self.assertIn("facets", res)
        self.assertIn("price", res["facets"])
        for metric in ["min", "max", "avg", "count", "sum"]:
            self.assertIn(metric, res["facets"]["price"])

    def test_number_facet_with_ranges(self):
        """Test getting numeric facet with custom ranges"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            facets={
                "fields": {
                    "price": {
                        "type": "number",
                        "ranges": [
                            {"from": 0, "to": 50, "name": "cheap"},
                            {"from": 50, "name": "expensive"}
                        ]
                    }
                }
            }
        )
        self.assertIn("facets", res)
        self.assertIn("price", res["facets"])
        self.assertEqual(len(res["facets"]["price"]), 2)

    def test_overlapping_ranges_fails(self):
        """Test that overlapping ranges raise an error"""
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.unstructured_text_index_name).search(
                "shirt",
                search_method="HYBRID",
                facets={
                    "fields": {
                        "price": {
                            "type": "number",
                            "ranges": [
                                {"from": 0, "to": 60},
                                {"from": 50, "to": 100}
                            ]
                        }
                    }
                }
            )
        self.assertIn("Range configurations must not overlap", str(e.exception))

    def test_non_number_facet_with_ranges_fails(self):
        """Test that ranges are only allowed for number facets"""
        for facet_type in ["string", "array"]:
            with self.assertRaises(MarqoWebError) as e:
                self.client.index(self.unstructured_text_index_name).search(
                    "shirt",
                    search_method="HYBRID",
                    facets={
                        "fields": {
                            "color": {
                                "type": facet_type,
                                "ranges": [
                                    {"from": 0, "to": 50}
                                ]
                            }
                        }
                    }
                )
            self.assertIn("Ranges can only be used for 'number' facets", str(e.exception))

    def test_facets_with_filter_exclusions(self):
        """Test facets with filter term exclusions"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            facets={
                "fields": {
                    "color": {
                        "type": "string",
                        "excludeTerms": ["price:[100 TO 200]"]
                    }
                }
            },
            filter_string="price:[100 TO 200]"
        )
        self.assertIn("facets", res)
        self.assertIn("color", res["facets"])

    def test_filter_exclusions_without_filter_fails(self):
        """Test that exclude terms require a filter string"""
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.unstructured_text_index_name).search(
                "shirt",
                search_method="HYBRID",
                facets={
                    "fields": {
                        "color": {
                            "type": "string",
                            "excludeTerms": ["nonexistent:value"]
                        }
                    }
                }
            )
        self.assertIn("Exclude terms can only be used with a filter string.", str(e.exception))

    def test_invalid_filter_exclusions_fails(self):
        """Test that exclude terms must be present in filter string"""
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.unstructured_text_index_name).search(
                "shirt",
                search_method="HYBRID",
                filter_string="existent:notvalue",
                facets={
                    "fields": {
                        "color": {
                            "type": "string",
                            "excludeTerms": ["nonexistent:value"]
                        }
                    }
                }
            )
        self.assertIn("that do not appear in the filter string", str(e.exception))

    def test_invalid_facet_parameters_fail(self):
        """Test that invalid facet parameters raise errors"""
        invalid_params = ["fields.to_", "fields.from_", "max_results", "max_depth", "fields.max_results"]
        for param in invalid_params:
            with self.assertRaises(MarqoWebError) as e:
                facets = {"fields": {"color": {"type": "string"}}}
                if param.startswith("fields."):
                    facets["fields"]["color"][param[6:]] = 100
                else:
                    facets[param] = 100
                self.client.index(self.unstructured_text_index_name).search(
                    "shirt",
                    search_method="HYBRID",
                    facets=facets
                )
            self.assertIn("extra fields not permitted", str(e.exception).lower())

    def test_track_total_hits_default(self):
        """Test track_total_hits parameter defaults to False"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID"
        )
        self.assertNotIn("totalHits", res)

    def test_track_total_hits_enabled(self):
        """Test track_total_hits parameter returns total hits count"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            track_total_hits=True
        )
        self.assertIn("totalHits", res)
        self.assertIsInstance(res["totalHits"], int)
        self.assertGreater(res["totalHits"], 0)

    def test_track_total_hits_with_filter(self):
        """Test track_total_hits with filter returns correct count"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            track_total_hits=True,
            filter_string="price:[100 TO 200]"
        )
        self.assertIn("totalHits", res)
        filtered_res = self.client.index(self.unstructured_text_index_name).search(
            "shirt",
            search_method="HYBRID",
            filter_string="price:[100 TO 200]"
        )
        self.assertEqual(res["totalHits"], len(filtered_res["hits"]))

    def test_track_total_hits_no_results(self):
        """Test track_total_hits when there are no matching documents"""
        res = self.client.index(self.unstructured_text_index_name).search(
            "nonexistentquery123456",
            search_method="HYBRID",
            hybrid_parameters={
                "retrievalMethod": "lexical",
                "rankingMethod": "lexical"
            },
            track_total_hits=True
        )
        self.assertIn("totalHits", res)
        self.assertEqual(res["totalHits"], 0)
        self.assertEqual(len(res["hits"]), 0)

    def test_facets_filter_string_parsing_edge_cases(self):
        """Tests handling of filter strings with different parentheses placements in facet queries via API"""
        test_cases = [
            {
                "name": "Filter string with parentheses around the whole string",
                "filter": "(price:[49 TO 49.1] AND NOT color:red)",
                "exclude_terms": ["color:red"]
            },
            {
                "name": "Filter string with parentheses around term",
                "filter": "price:[49 TO 49.1] AND (NOT color:red)",
                "exclude_terms": ["color:red"]
            },
            {
                "name": "Filter string with parentheses around value",
                "filter": "price:[49 TO 49.1] AND NOT color:(red)",
                "exclude_terms": ["color:(red)"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                res = self.client.index(self.unstructured_text_index_name).search(
                    "shirt",
                    search_method="HYBRID",
                    filter_string=test_case["filter"],
                    facets={
                        "fields": {
                            "color": {
                                "type": "string",
                                "excludeTerms": test_case["exclude_terms"]
                            }
                        }
                    }
                )
                self.assertIn("facets", res)
                self.assertIn("color", res["facets"])

    def test_non_existing_array_field_raises_error(self):
        """Test that searching a non-existing array field raises an error"""
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.unstructured_text_index_name).search(
                "shirt",
                search_method="HYBRID",
                facets={
                    "fields": {
                        "non_existing_field": {
                            "type": "array"
                        }
                    }
                }
            )
        self.assertIn("is not present in any index document", str(e.exception))
