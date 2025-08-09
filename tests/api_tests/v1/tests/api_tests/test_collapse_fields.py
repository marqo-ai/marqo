import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError

class TestCollapseFields(MarqoTestCase):
    unstructured_text_index_name = "unstructured_index_text" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)
        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "collapseFields": [
                    {"name": "parent_id", "minGroups": 100}
                ]
            },
        ])

        cls.indexes_to_delete = [cls.unstructured_text_index_name]

    def test_collapse_fields_is_in_index_settings(self):
        index_name = self.unstructured_text_index_name
        index_settings = self.client.index(index_name).get_settings()
        self.assertTrue("collapseFields" in index_settings)
        self.assertEqual(len(index_settings["collapseFields"]), 1)
        self.assertEqual(index_settings["collapseFields"][0], {"name": "parent_id", "minGroups": 100})

    def test_add_documents_mixed_batch_with_collapse_field_errors(self):
        """Test that valid documents succeed while invalid ones fail in same batch"""
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
            {"_id": "invalid1", "title": "Invalid document - missing field"},
            {"_id": "valid2", "title": "Valid document 2", "parent_id": "group_2"},
            {"_id": "invalid2", "title": "Invalid document - wrong type", "parent_id": 456}
        ]

        res = self.client.index(self.unstructured_text_index_name).add_documents(
            docs, tensor_fields=[]
        )
        print(res)

        # Verify failed documents
        failed_items = [item for item in res['items'] if item['status'] != 200]
        self.assertEqual(2, len(failed_items), "Expected 2 failed documents")
        self.assertIn("Document missing required field 'parent_id'", failed_items[0]['message'])
        self.assertIn("Field 'parent_id' must be of type string", failed_items[1]['message'])

        # Verify successful documents
        successful_items = [item for item in res['items'] if item['status'] == 200]
        self.assertEqual(2, len(successful_items), "Expected 2 successful documents")

        successful_ids = {item['_id'] for item in successful_items}
        self.assertEqual({"valid1", "valid2"}, successful_ids)

        # Verify we can retrieve the parent_id back
        valid_docs = self.client.index(self.unstructured_text_index_name).get_documents(
            document_ids=list(successful_ids))

        self.assertFalse(valid_docs['errors'])
        self.assertEqual(2, len(valid_docs['results']))
        for doc in valid_docs['results']:
            expected_parent_id = "group_1" if doc["_id"] == "valid1" else "group_2"
            self.assertEqual(expected_parent_id, doc["parent_id"])

    def test_partial_update_of_collapse_field_does_not_work(self):
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
        ]

        index_name = self.unstructured_text_index_name

        res = self.client.index(index_name).add_documents(
            docs, tensor_fields=[]
        )

        self.assertFalse(res['errors'])
        self.assertEqual(1, len(res['items']))

        update_res = self.client.index(index_name).update_documents([
            {"_id": "valid1", "parent_id": "group_2"}
        ])

        self.assertTrue(update_res['errors'])
        self.assertEqual(400, update_res['items'][0]['status'])

        # TODO please note that this is not working due to a side effect that partial update treats all string fields
        #  as lexical fields. Ideally, partial updates should treat collapse differently to avoid confusing error msg.
        self.assertIn("parent_id of type str does not exist in the original document. "
                      "Marqo does not support adding new lexical fields in partial updates", update_res['items'][0]['error'])

        doc = self.client.index(index_name).get_document(document_id="valid1")

        self.assertEqual(doc["parent_id"], "group_1")

    def test_search_with_invalid_collapse_field_raises_error(self):
        """Test that search with invalid collapse field name raises error"""

        index_name = self.unstructured_text_index_name

        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(index_name).search(
                q="test query",
                search_method="HYBRID",
                collapse_fields=[{"name": "non_existent_field"}]
            )

        self.assertIn("Field 'non_existent_field' is not configured as a collapse field for this index",
                      str(cm.exception))

    def test_search_with_valid_collapse_field_succeeds(self):
        """Test that search with valid collapse field name succeeds"""
        # Add some test documents
        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(10) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor"),
            ("tensor", "tensor"),
            ("tensor", "lexical"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 30,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    limit=6
                )

                # Verify the search executed successfully and only contain 1 doc from each group
                self.assertEqual(5, len(res["hits"]))  # there's only 5 groups, so at most 5 results
                self.assertEqual(set([f"group_{g}" for g in range(5)]), set([hit['parent_id'] for hit in res["hits"]]))

    def test_filter(self):
        # Add some test documents
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": g + 1,
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor"),
            ("tensor", "tensor"),
            ("tensor", "lexical"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 30,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    filter_string="price:[* TO 3] AND (color:red OR color:yellow)",
                    limit=6
                )

                self.assertEqual(3, len(res["hits"]))
                self.assertEqual(3, len(set([hit['parent_id'] for hit in res["hits"]])))

                for hit in res["hits"]:
                    self.assertLessEqual(hit["price"], 3)
                    self.assertIn(hit["color"], ("red", "yellow"))

    def test_facets(self):
        # Add some test documents
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": float(g + 1.1),
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor"),
            ("tensor", "tensor"),
            ("tensor", "lexical"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 30,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    filter_string="price:[0 TO 4] AND (color:red OR color:yellow)",
                    facets={"fields": {
                        "price": {"type": "number", "ranges": [
                            {"from": 0, "to": 2},
                            {"from": 2, "to": 4}
                        ]},
                        "color": {"type": "string"}
                    }},
                    limit=6
                )

                self.assertEqual(3, len(res["hits"]))
                self.assertDictEqual({'count': 1}, res["facets"]["price"]["0.0:2.0"])
                self.assertDictEqual({'count': 2}, res["facets"]["price"]["2.0:4.0"])
                self.assertDictEqual({'red': {'count': 3}, 'yellow': {'count': 3}}, res["facets"]["color"])

    def test_pagination(self):
        # Add some test documents
        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(10) for g in range(10)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        test_cases = [
            # ("disjunction", "rrf"), # FIXME dup can only be fixed by pagination fix
            ("lexical", "lexical"),
            # ("lexical", "tensor"),  # FIXME dup and missing doc
            ("tensor", "tensor"),
            ("tensor", "lexical"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                page_1_res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 100,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    limit=6
                )

                self.assertEqual(6, len(page_1_res["hits"]))
                page_1_res_groups = set([hit['parent_id'] for hit in page_1_res["hits"]])
                self.assertEqual(6, len(page_1_res_groups))

                page_2_res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 100,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    limit=6,
                    offset=6
                )

                self.assertEqual(4, len(page_2_res["hits"]))
                page_2_res_groups = set([hit['parent_id'] for hit in page_2_res["hits"]])
                self.assertEqual(4, len(page_2_res_groups))

                print(retrieval_method, ranking_method, page_1_res_groups, page_2_res_groups)
                self.assertEqual(10, len(page_1_res_groups.union(page_2_res_groups)))

    def test_sort_by(self):
        # Add some test documents
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": g + 1,
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        res = self.client.index(index_name).search(
            q="test",
            search_method="HYBRID",
            hybrid_parameters={
                "rerankDepthTensor": 30,
            },
            collapse_fields=[{"name": "parent_id"}],
            sort_by={
                "fields": [{"fieldName": "price", "order": "desc"}],
            },
            filter_string="price:[* TO 3] AND (color:red OR color:yellow)",
            limit=6
        )

        self.assertEqual(3, len(res["hits"]))
        self.assertListEqual(["group_2", "group_1", "group_0"], [hit['parent_id'] for hit in res["hits"]])

        for hit in res["hits"]:
            self.assertLessEqual(hit["price"], 3)
            self.assertIn(hit["color"], ("red", "yellow"))


    def test_relevance_cutoff(self):
        # 30 documents designed for "machine learning artificial intelligence algorithms" query
        test_docs = [
            # === HIGH RELEVANCE (10 docs) - Contains ALL 5 query words ===
            {"_id": "h1", "parent_id": "group_0",
             "content": "Machine learning algorithms in artificial intelligence enable systems to adapt by processing data efficiently.",
             "sort_value": 8.1},
            {"_id": "h2", "parent_id": "group_0",
             "content": "Artificial intelligence relies on machine learning algorithms to build predictive models from large datasets.",
             "sort_value": 9.2},
            {"_id": "h3", "parent_id": "group_0",
             "content": "Researchers develop artificial intelligence machine learning algorithms to improve decision-making processes.",
             "sort_value": 7.4},
            {"_id": "h4", "parent_id": "group_0",
             "content": "Scalable artificial intelligence frameworks integrate machine learning algorithms for real-time data analysis.",
             "sort_value": 9.8},
            {"_id": "h5", "parent_id": "group_1",
             "content": "Modern artificial intelligence and machine learning algorithms optimize operational workflows across industries.",
             "sort_value": 6.5},
            {"_id": "h6", "parent_id": "group_1",
             "content": "Sophisticated artificial intelligence machine learning algorithms optimize data mining operations effectively.",
             "sort_value": 8.9},
            {"_id": "h7", "parent_id": "group_1",
             "content": "Cutting-edge artificial intelligence machine learning algorithms accelerate data processing in cloud platforms.",
             "sort_value": 5.3},
            {"_id": "h8", "parent_id": "group_2",
             "content": "Enterprise artificial intelligence solutions embed machine learning algorithms to enhance user experiences.",
             "sort_value": 9.0},
            {"_id": "h9", "parent_id": "group_2",
             "content": "Robust artificial intelligence machine learning algorithms improve data quality assessment procedures.",
             "sort_value": 7.8},
            {"_id": "h10", "parent_id": "group_2",
             "content": "Innovative artificial intelligence and machine learning algorithms revolutionize data analytics workflows.",
             "sort_value": 8.4},

            # === MEDIUM RELEVANCE (10 docs) - Contains EXACTLY 3 of the 5 query words ===
            # (e.g., {machine, learning, algorithms} or {artificial, intelligence, learning}, etc.)
            {"_id": "m1",  "parent_id": "group_3",
             "content": "Machine learning algorithms process financial time series for forecasting market trends.",
             "sort_value": 64},
            {"_id": "m2",  "parent_id": "group_3",
             "content": "Artificial intelligence algorithms underpin recommendation engines in e-commerce platforms.",
             "sort_value": 6.7},
            {"_id": "m3",  "parent_id": "group_3",
             "content": "Artificial intelligence learning models adapt to new user behaviors in real time.",
             "sort_value": 4.3},
            {"_id": "m4",  "parent_id": "group_3",
             "content": "Machine and artificial intelligence technologies converge to create autonomous robotic systems.",
             "sort_value": 7.1},
            {"_id": "m5",  "parent_id": "group_4",
             "content": "Machine learning artificial neural networks mimic animal brain structures.",
             "sort_value": 6.2},
            {"_id": "m6",  "parent_id": "group_4",
             "content": "Advanced machine learning algorithms accelerate computational biology research.",
             "sort_value": 5.9},
            {"_id": "m7",  "parent_id": "group_4",
             "content": "Distributed artificial intelligence systems leverage algorithms for parallel decision making.",
             "sort_value": 4.8},
            {"_id": "m8",  "parent_id": "group_4",
             "content": "Deep learning frameworks support neural architectures and optimization algorithms.",
             "sort_value": 7.5},
            {"_id": "m9",  "parent_id": "group_5",
             "content": "Evolutionary algorithms integrate with machine frameworks for adaptive problem solving.",
             "sort_value": 6.0},
            {"_id": "m10",  "parent_id": "group_5",
             "content": "Artificial learning simulations test intelligence benchmarks under controlled conditions.",
             "sort_value": 4.1},

            # === LOW RELEVANCE ===
            # 5 docs with exactly 1 query word, matching the word counts of l1–l5
            {"_id": "l1",  "parent_id": "group_6",
             "content": "Engineers use machine tools for precise cutting.",
             "sort_value": 65},

            {"_id": "l2",  "parent_id": "group_6",
             "content": "Innovators encourage collaborative learning environments to foster team growth.",
             "sort_value": 2.7},  # 9 words, contains "learning"

            {"_id": "l3",  "parent_id": "group_6",
             "content": "Manufacturers produce artificial components designed precisely for specialized industrial applications.",
             "sort_value": 1.4},  # 10 words, contains "artificial"

            {"_id": "l4",  "parent_id": "group_6",
             "content": "Local units value human intelligence during critical decision making.",
             "sort_value": 100},  # 9 words, contains "intelligence"

            {"_id": "l5",  "parent_id": "group_6",
             "content": "Researchers propose algorithms optimized specifically to accelerate image processing tasks.",
             "sort_value": 60},  # 10 words, contains "algorithms"

            # === Irrelevant ===
            # 5 docs with 0 words from the query
            {"_id": "l6",  "parent_id": "group_7",
             "content": "Bright morning sunlight streamed through the quiet study room.",
             "sort_value": 2.1},

            {"_id": "l7",  "parent_id": "group_7",
             "content": "Surprising weather patterns emerged across the town.",
             "sort_value": 70},

            {"_id": "l8",  "parent_id": "group_7",
             "content": "Vibrant wildflowers adorned the rolling hills during summer.",
             "sort_value": 1.9},

            {"_id": "l9",  "parent_id": "group_7",
             "content": "Chilly autumn breeze painted golden leaves across streets.",
             "sort_value": 24},

            {"_id": "l10",  "parent_id": "group_7",
             "content": "The ancient manuscript revealed hidden stories from forgotten civilizations.",
             "sort_value": 5.6}

            # group 0-2 are of high relevance, 3-5 are of medium relevance, 6-7 are of low relevance
        ]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            test_docs, tensor_fields=["content"]
        )

        res = self.client.index(index_name).search(
            q="machine learning artificial intelligence algorithms",
            search_method="HYBRID",
            hybrid_parameters={
                "rerankDepthTensor": 30,
            },
            collapse_fields=[{"name": "parent_id"}],
            sort_by={
                "fields": [{"fieldName": "sort_value", "order": "desc"}],
            },
            relevance_cutoff={
                "method": "mean_std_dev",
                "parameters": {"stdDevFactor": 0.5}
            },
            limit=6
        )

        # Verify we only return 1 doc for each group
        unique_groups = set([hit['parent_id'] for hit in res['hits']])
        self.assertEqual(len(unique_groups), len(res['hits']))

        # Verify we only return docs with high relevance
        for group in unique_groups:
            self.assertIn(group, ['group_0', 'group_1', 'group_2'])

    def test_score_modifiers(self):
        # Add some test documents
        docs = [{"_id": f"doc{g}{i:02}", "rating": i+1, "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(5) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        score_modifiers = {
            "multiply_score_by": [{"field_name": "rating", "weight": 1}]
        }

        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor"),
            ("tensor", "tensor"),
            ("tensor", "lexical"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = self.client.index(index_name).search(
                    q="test",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": retrieval_method,
                        "rankingMethod": ranking_method,
                        "rerankDepthTensor": 30,
                        "scoreModifiersTensor": score_modifiers if ranking_method != "lexical" else None,
                        "scoreModifiersLexical": score_modifiers if ranking_method != "tensor" or retrieval_method != "tensor" else None,
                    },
                    collapse_fields=[{"name": "parent_id"}],
                    limit=6
                )

                # Verify that the result only contains doc with rating 5
                self.assertTrue(all([hit['rating'] == 5 for hit in res['hits']]))

    def test_filter_by_collapse_field(self):
        # Add some test documents
        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(5) for g in range(5)]

        index_name = self.unstructured_text_index_name

        self.client.index(index_name).add_documents(
            docs, tensor_fields=["title"]
        )

        hybrid_res = self.client.index(index_name).search(
            q="test",
            search_method="HYBRID",
            hybrid_parameters={
                "retrievalMethod": "lexical",
                "rankingMethod": "lexical",
            },
            filter_string="parent_id:group_1",
            limit=10
        )

        # Verify the search returns all docs in one group
        self.assertEqual(5, len(hybrid_res["hits"]))
        self.assertEqual(set([f"doc1{i:02}" for i in range(5)]), set([hit['_id'] for hit in hybrid_res["hits"]]))

        # verify lexical search also works
        lexical_res = self.client.index(index_name).search(
            q="*",
            search_method="LEXICAL",
            filter_string="parent_id:group_1",
            limit=10
        )

        # Verify the search returns all docs in one group
        self.assertEqual(5, len(lexical_res["hits"]))
        self.assertEqual(set([f"doc1{i:02}" for i in range(5)]), set([hit['_id'] for hit in lexical_res["hits"]]))