from unittest import mock

import os
import pytest

from marqo.api.exceptions import InvalidArgError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import CollapseField, SemiStructuredMarqoIndex
from marqo.core.models.marqo_index import Model
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel, RelevanceCutoffMethod, \
    MeanStdParameters
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField
from tests.integ_tests.marqo_test import MarqoTestCase


class TestCollapseFields(MarqoTestCase):
    """Integration tests for collapse fields functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            collapse_fields=[CollapseField(name="parent_id", minGroups=3)],
            model=Model(name="hf/all-MiniLM-L6-v2")
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
        ])

        cls.default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_index_should_contain_collapse_field_settings(self):
        """Test that collapse field in the index creation request is persisted"""
        index = self.index_management.get_index(self.default_text_index.name)
        self.assertIsInstance(index, SemiStructuredMarqoIndex)
        self.assertIsNotNone(index.collapse_fields)
        self.assertEqual(index.collapse_fields[0].name, "parent_id")
        self.assertEqual(index.collapse_fields[0].min_groups, 3)

    def test_add_documents_mixed_batch_with_collapse_field_errors(self):
        """Test that valid documents succeed while invalid ones fail in same batch"""
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
            {"_id": "invalid1", "title": "Invalid document - missing field"},
            {"_id": "valid2", "title": "Valid document 2", "parent_id": "group_2"},
            {"_id": "invalid2", "title": "Invalid document - wrong type", "parent_id": 456}
        ]

        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=[]
            )
        )

        # Verify failed documents
        failed_items = [item for item in res.items if item.status != 200]
        self.assertEqual(2, len(failed_items), "Expected 2 failed documents")
        self.assertIn("Document missing required field 'parent_id'", failed_items[0].message)
        self.assertIn("Field 'parent_id' must be of type string", failed_items[1].message)

        # Verify successful documents
        successful_items = [item for item in res.items if item.status == 200]
        self.assertEqual(2, len(successful_items), "Expected 2 successful documents")

        successful_ids = {item.id for item in successful_items}
        self.assertEqual({"valid1", "valid2"}, successful_ids)

        # Verify we can retrieve the parent_id back
        valid_docs = tensor_search.get_documents_by_ids(config=self.config, index_name=self.default_text_index.name,
                                                        document_ids=successful_ids)

        self.assertFalse(valid_docs.errors)
        self.assertEqual(2, len(valid_docs.results))
        for doc in valid_docs.results:
            expected_parent_id = "group_1" if doc["_id"] == "valid1" else "group_2"
            self.assertEqual(expected_parent_id, doc["parent_id"])

    def test_partial_update_of_collapse_field_does_not_work(self):
        """Test that partial update on the collapse field fails"""
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
        ]

        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=[]
            )
        )

        self.assertFalse(res.errors)
        self.assertEqual(1, len(res.items))

        update_res = self.config.document.partial_update_documents_by_index_name(
            index_name=self.default_text_index.name,
            partial_documents=[{"_id": "valid1", "parent_id": "group_2"}])

        self.assertTrue(update_res.errors)
        self.assertEqual(400, update_res.items[0].status)

        # TODO please note that this is not working due to a side effect that partial update treats all string fields
        #  as lexical fields. Ideally, partial updates should treat collapse differently to avoid confusing error msg.
        self.assertIn("parent_id of type str does not exist in the original document. "
                      "Marqo does not support adding new lexical fields in partial updates", update_res.items[0].error)

        doc = tensor_search.get_document_by_id(config=self.config, index_name=self.default_text_index.name,
                                               document_id="valid1")

        self.assertEqual(doc["parent_id"], "group_1")

    def test_search_with_invalid_collapse_field_raises_error(self):
        """Test that search with invalid collapse field name raises error"""
        with self.assertRaises(InvalidArgError) as cm:
            tensor_search.search(
                config=self.config,
                index_name=self.default_text_index.name,
                text="test query",
                search_method="HYBRID",
                collapse=CollapseModel(name="non_existent_field")
            )

        self.assertIn("Field 'non_existent_field' is not configured as a collapse field for this index",
                      str(cm.exception))

    def test_search_with_valid_collapse_field_succeeds(self):
        """Test that search with valid collapse field name succeeds"""

        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}", "group": g}
                for i in range(10) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=50,  # tensor-tensor will have fewer hits if we do not increase this, why?
                    ),
                    # parent id is not added here, it will be added in the query for collapsing, but not in the result
                    attributes_to_retrieve=["title", "group"],
                    collapse=CollapseModel(name="parent_id"),
                    result_count=6
                )

                # there's only 5 groups, so only 5 results
                self.assertEqual(5, len(res["hits"]))
                # only contain 1 doc from each group
                self.assertEqual(set(range(5)), set([hit['group'] for hit in res["hits"]]))
                # parent_id is not returned
                self.assertTrue(all("parent_id" not in hit for hit in res["hits"]))

    def test_filter(self):
        """Test that filtering works with search with collapse field"""
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": g + 1,
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=10,  # tensor-tensor will have fewer hits if we do not increase this, why?
                    ),
                    collapse=CollapseModel(name="parent_id"),
                    filter="price:[* TO 3] AND (color:red OR color:yellow)",
                    result_count=6
                )

                self.assertEqual(3, len(res["hits"]))  # there's only 5 groups, so at most 5 results
                # 5 hits should have different group_ids
                self.assertEqual(3, len(set([hit['parent_id'] for hit in res["hits"]])))

                for hit in res["hits"]:
                    self.assertLessEqual(hit["price"], 3)
                    self.assertIn(hit["color"], ("red", "yellow"))

    def test_facets(self):
        """Test that facets query works with search with collapse field"""
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": float(g + 1.1),
                 "rating": int(g + 1) if g % 2 == 0 else float(g + 1.5),  # mix of float and int
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=10,  # tensor-tensor will have fewer hits if we do not increase this, why?
                    ),
                    collapse=CollapseModel(name="parent_id"),
                    filter="price:[0 TO 4] AND (color:red OR color:yellow)",
                    facets=FacetsParameters(
                        fields={
                            "price": FieldFacetsConfiguration(type="number", ranges=[
                                {"from": 0, "to": 2},
                                {"from": 2, "to": 4},
                            ]),
                            "rating": FieldFacetsConfiguration(type="number", ranges=[
                                {"from": 0, "to": 2},
                                {"from": 2, "to": 4},
                            ]),
                            "color": FieldFacetsConfiguration(type="string")
                        }
                    ),
                    track_total_hits=True,
                    result_count=6
                )

                self.assertEqual(3, len(res["hits"]))
                self.assertDictEqual({'red': {'count': 3}, 'yellow': {'count': 3}}, res["facets"]["color"])

                # prices are [1.1, 2.1, 3.1]
                self.assertDictEqual({'count': 1}, res["facets"]["price"]["0.0:2.0"])
                self.assertDictEqual({'count': 2}, res["facets"]["price"]["2.0:4.0"])

                # ratings are [1, 2.5, 3]
                self.assertDictEqual({'count': 2}, res["facets"]["rating"]["2.0:4.0"])
                # FIXME mixed int and float rating confuses Vespa, 0.0:2.0 in the float field returns 2 instead of 0
                self.assertDictEqual({'count': 3}, res["facets"]["rating"]["0.0:2.0"])

                # Test that the hit count returns the count of unique collapse field value
                self.assertEqual(3, res['totalHits'])

    @pytest.mark.skip_for_multinode("Pagination result is not consistent across different Vespa infrastructures")
    def test_pagination(self):
        """Test that pagination works with search with collapse field"""
        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(10) for g in range(10)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        test_cases = [
            # (RetrievalMethod.Disjunction, RankingMethod.RRF),  # FIXME dup can only be fixed by pagination fix
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            # (RetrievalMethod.Lexical, RankingMethod.Tensor),  # FIXME dup and missing doc
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                page_1_res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=100,  # set a large value to expand the tensor retrieval set
                    ),
                    collapse=CollapseModel(name="parent_id"),
                    result_count=6
                )

                self.assertEqual(6, len(page_1_res["hits"]))
                page_1_res_groups = set([hit['parent_id'] for hit in page_1_res["hits"]])
                self.assertEqual(6, len(page_1_res_groups))

                page_2_res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=100,  # set a large value to expand the tensor retrieval set
                    ),
                    collapse=CollapseModel(name="parent_id"),
                    offset=6,
                    result_count=6
                )

                self.assertEqual(4, len(page_2_res["hits"]))
                page_2_res_groups = set([hit['parent_id'] for hit in page_2_res["hits"]])
                self.assertEqual(4, len(page_2_res_groups))
                self.assertEqual(10, len(page_1_res_groups.union(page_2_res_groups)))

    def test_sort_by(self):
        """Test that sort by param works with search with collapse field"""
        colors = ['white', 'red', 'green', 'yellow', 'blue']
        docs = [{"_id": f"doc{g}{i:02}",
                 "title": f"Test document {g}{i:02}",
                 "parent_id": f"group_{g}",
                 "price": g + 1,
                 "color": colors[i % 5]
                 } for i in range(10) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="test",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                rerankDepthTensor=10,
            ),
            sort_by=SortByModel(fields=[
                SortByField(field_name="price", order="desc"),
            ], min_sort_candidates=18),
            collapse=CollapseModel(name="parent_id"),
            filter="price:[* TO 3] AND (color:red OR color:yellow)",
            result_count=6
        )

        self.assertEqual(3, len(res["hits"]))
        # all hits should have different group_ids
        self.assertListEqual(["group_2", "group_1", "group_0"], [hit['parent_id'] for hit in res["hits"]])

        for hit in res["hits"]:
            self.assertLessEqual(hit["price"], 3)
            self.assertIn(hit["color"], ("red", "yellow"))

    def test_relevance_cutoff(self):
        """Test that relevance cutoff param works with search with collapse field"""
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
            {"_id": "m1", "parent_id": "group_3",
             "content": "Machine learning algorithms process financial time series for forecasting market trends.",
             "sort_value": 64},
            {"_id": "m2", "parent_id": "group_3",
             "content": "Artificial intelligence algorithms underpin recommendation engines in e-commerce platforms.",
             "sort_value": 6.7},
            {"_id": "m3", "parent_id": "group_3",
             "content": "Artificial intelligence learning models adapt to new user behaviors in real time.",
             "sort_value": 4.3},
            {"_id": "m4", "parent_id": "group_3",
             "content": "Machine and artificial intelligence technologies converge to create autonomous robotic systems.",
             "sort_value": 7.1},
            {"_id": "m5", "parent_id": "group_4",
             "content": "Machine learning artificial neural networks mimic animal brain structures.",
             "sort_value": 6.2},
            {"_id": "m6", "parent_id": "group_4",
             "content": "Advanced machine learning algorithms accelerate computational biology research.",
             "sort_value": 5.9},
            {"_id": "m7", "parent_id": "group_4",
             "content": "Distributed artificial intelligence systems leverage algorithms for parallel decision making.",
             "sort_value": 4.8},
            {"_id": "m8", "parent_id": "group_4",
             "content": "Deep learning frameworks support neural architectures and optimization algorithms.",
             "sort_value": 7.5},
            {"_id": "m9", "parent_id": "group_5",
             "content": "Evolutionary algorithms integrate with machine frameworks for adaptive problem solving.",
             "sort_value": 6.0},
            {"_id": "m10", "parent_id": "group_5",
             "content": "Artificial learning simulations test intelligence benchmarks under controlled conditions.",
             "sort_value": 4.1},

            # === LOW RELEVANCE ===
            # 5 docs with exactly 1 query word, matching the word counts of l1–l5
            {"_id": "l1", "parent_id": "group_6",
             "content": "Engineers use machine tools for precise cutting.",
             "sort_value": 65},

            {"_id": "l2", "parent_id": "group_6",
             "content": "Innovators encourage collaborative learning environments to foster team growth.",
             "sort_value": 2.7},  # 9 words, contains "learning"

            {"_id": "l3", "parent_id": "group_6",
             "content": "Manufacturers produce artificial components designed precisely for specialized industrial applications.",
             "sort_value": 1.4},  # 10 words, contains "artificial"

            {"_id": "l4", "parent_id": "group_6",
             "content": "Local units value human intelligence during critical decision making.",
             "sort_value": 100},  # 9 words, contains "intelligence"

            {"_id": "l5", "parent_id": "group_6",
             "content": "Researchers propose algorithms optimized specifically to accelerate image processing tasks.",
             "sort_value": 60},  # 10 words, contains "algorithms"

            # === Irrelevant ===
            # 5 docs with 0 words from the query
            {"_id": "l6", "parent_id": "group_7",
             "content": "Bright morning sunlight streamed through the quiet study room.",
             "sort_value": 2.1},

            {"_id": "l7", "parent_id": "group_7",
             "content": "Surprising weather patterns emerged across the town.",
             "sort_value": 70},

            {"_id": "l8", "parent_id": "group_7",
             "content": "Vibrant wildflowers adorned the rolling hills during summer.",
             "sort_value": 1.9},

            {"_id": "l9", "parent_id": "group_7",
             "content": "Chilly autumn breeze painted golden leaves across streets.",
             "sort_value": 24},

            {"_id": "l10", "parent_id": "group_7",
             "content": "The ancient manuscript revealed hidden stories from forgotten civilizations.",
             "sort_value": 5.6}

            # group 0-2 are of high relevance, 3-5 are of medium relevance, 6-7 are of low relevance
        ]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=test_docs,
                tensor_fields=["content"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="machine learning artificial intelligence algorithms",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                rerankDepthTensor=10,
            ),
            sort_by=SortByModel(fields=[
                SortByField(field_name="sort_value", order="desc"),
            ]),
            relevance_cutoff=RelevanceCutoffModel(method=RelevanceCutoffMethod.MeanStdDev,
                                                  parameters=MeanStdParameters(stdDevFactor=0.5)),
            collapse=CollapseModel(name="parent_id"),
            result_count=6
        )

        # Verify we only return 1 doc for each group
        unique_groups = set([hit['parent_id'] for hit in res['hits']])
        self.assertEqual(len(unique_groups), len(res['hits']))

        # Verify we only return docs with high relevance
        for group in unique_groups:
            self.assertIn(group, ['group_0', 'group_1', 'group_2'])

    @pytest.mark.skip_for_multinode
    def test_score_modifiers(self):
        """Test that score modifiers work with search with collapse field"""
        docs = [
            {"_id": f"doc{g}{i:02}", "rating": i + 1, "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
            for i in range(5) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        score_modifiers = ScoreModifierLists(multiply_score_by=[ScoreModifierOperator(field_name="rating", weight=1)])

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="test",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                        rerankDepthTensor=25,  # make this deep enough to see high rating docs
                        scoreModifiersTensor=score_modifiers if ranking_method in [RankingMethod.Tensor,
                                                                                   RankingMethod.RRF] else None,
                        scoreModifiersLexical=score_modifiers if ranking_method != RankingMethod.Tensor or retrieval_method != RetrievalMethod.Tensor else None,
                    ),
                    result_count=6,
                    collapse=CollapseModel(name="parent_id"),
                )

                # Verify that the result only contains doc with rating 5
                self.assertTrue(all([hit['rating'] == 5 for hit in res['hits']]))

    def test_filter_by_collapse_field(self):
        """Test that filtering on collapse field works for both lexical search and hybrid lexical-lexical search"""
        docs = [{"_id": f"doc{g}{i:02}", "title": f"Test document {g}{i:02}", "parent_id": f"group_{g}"}
                for i in range(5) for g in range(5)]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        hybrid_res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="test",
            search_method="HYBRID",
            filter="parent_id:group_1",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
            ),
            result_count=10
        )

        # Verify the search returns all docs in one group
        self.assertEqual(5, len(hybrid_res["hits"]))
        self.assertEqual(set([f"doc1{i:02}" for i in range(5)]), set([hit['_id'] for hit in hybrid_res["hits"]]))

        # verify lexical search also works
        lexical_res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="*",
            search_method="LEXICAL",
            filter="parent_id:group_1",
            result_count=10
        )

        # Verify the search returns all docs in one group
        self.assertEqual(5, len(lexical_res["hits"]))
        self.assertEqual(set([f"doc1{i:02}" for i in range(5)]), set([hit['_id'] for hit in lexical_res["hits"]]))


class TestCollapseWithSortByFeature(MarqoTestCase):
    """Integration tests for collapse fields with sort by functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            collapse_fields=[CollapseField(name="category", minGroups=2)]
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
        ])

        cls.default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

    def _add_shoe_documents(self):
        """Add sample shoe documents across two category groups with varying prices."""
        docs = [
            # Category A - 5 variants with different prices
            {"_id": "shoe_a1", "title": "Running Shoe Alpha", "category": "shoes_a", "price": 120.0, "cost": 95.0,
             "brand": "nike"},
            {"_id": "shoe_a2", "title": "Running Shoe Beta", "category": "shoes_a", "price": 89.99, "cost": 70.0,
             "brand": "adidas"},
            {"_id": "shoe_a3", "title": "Running Shoe Gamma", "category": "shoes_a", "price": 150.0, "cost": 110.0,
             "brand": "nike"},
            {"_id": "shoe_a4", "title": "Running Shoe Delta", "category": "shoes_a", "price": 65.0, "cost": 50.0},
            {"_id": "shoe_a5", "title": "Running Shoe Epsilon", "category": "shoes_a", "price": 200.0, "cost": 160.0,
             "brand": "puma"},
            # Category B - 5 variants with different prices
            {"_id": "shoe_b1", "title": "Hiking Boot Alpha", "category": "shoes_b", "price": 180.0, "cost": 140.0,
             "brand": "merrell"},
            {"_id": "shoe_b2", "title": "Hiking Boot Beta", "category": "shoes_b", "price": 75.0, "cost": 55.0,
             "brand": "columbia"},
            {"_id": "shoe_b3", "title": "Hiking Boot Gamma", "category": "shoes_b", "price": 220.0, "cost": 170.0,
             "brand": "merrell"},
            {"_id": "shoe_b4", "title": "Hiking Boot Delta", "category": "shoes_b", "price": 99.0, "cost": 78.0},
            {"_id": "shoe_b5", "title": "Hiking Boot Epsilon", "category": "shoes_b", "price": 55.0, "cost": 40.0,
             "brand": "columbia"},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

    # ---- Scenario 1: Collapse sort by works ----

    def test_collapse_sort_by_price_asc(self):
        """Scenario 1a: Collapse with sortBy asc returns the cheapest variant per category."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe Delta",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=10
        )

        self.assertEqual(["shoe_a4", "shoe_b5"], [hit["_id"] for hit in res["hits"]])
        # _originalId is None for shoe_a4 because it's the cheapest in its category, not replacement
        self.assertEqual([None, "shoe_b4"], [hit.get("_originalId") for hit in res["hits"]])

    def test_collapse_sort_by_price_desc(self):
        """Scenario 1b: Collapse with sortBy desc returns the most expensive variant per category."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="desc")])
            ),
            result_count=10
        )

        self.assertEqual(["shoe_a5", "shoe_b3"], [hit["_id"] for hit in res["hits"]])

    def test_collapse_sort_by_different_numerical_field(self):
        """Scenario 1c: Collapse sortBy using cost instead of price."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="cost", order="asc")])
            ),
            result_count=10
        )

        self.assertEqual(["shoe_a4", "shoe_b5"], [hit["_id"] for hit in res["hits"]])

    def test_collapse_sort_by_with_filter(self):
        """Scenario 1d: Collapse sortBy combined with a filter."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            filter="price:[100 TO 200]",
            result_count=10
        )

        self.assertEqual(["shoe_a1", "shoe_b1"], [hit["_id"] for hit in res["hits"]])

    def test_collapse_sort_by_across_retrieval_methods(self):
        """Scenario 1e: Collapse sortBy works across different retrieval/ranking methods."""
        # group_a matches "premium running shoe" on all 3 words, group_b matches on 2 ("running shoe")
        docs = [
                   {"_id": f"doc_a{i}", "title": f"Premium running shoe model {i}", "category": "group_a",
                    "price": float(10 * (i + 1))}
                   for i in range(5)
               ] + [
                   {"_id": f"doc_b{i}", "title": f"Running shoe basic model {i}", "category": "group_b",
                    "price": float(10 * (i + 1))}
                   for i in range(5)
               ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.default_text_index.name,
                    text="premium running shoe",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                    ),
                    collapse=CollapseModel(
                        name="category",
                        sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
                    ),
                    result_count=10
                )

                # Both groups should return their cheapest doc (price=10.0)
                self.assertEqual(["doc_a0", "doc_b0"], sorted([hit["_id"] for hit in res["hits"]]))

    # ---- Scenario 2: No collapse sort by, based on original relevancy ----

    def test_collapse_without_sort_by_returns_relevancy_based(self):
        """Scenario 2: Collapse without sortBy returns the most relevant doc per group (default behavior)."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe Alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(name="category"),
            result_count=10
        )

        # alpha should match shoes_a1 and shoes_b1 best in their respective groups due to higher relevance
        self.assertEqual(["shoe_a1", "shoe_b1"], [hit["_id"] for hit in res["hits"]])

    # ---- Scenario 3: Some documents missing the sort by field ----

    def test_collapse_sort_by_with_some_docs_missing_sort_field(self):
        """Scenario 3: When some docs in a collapse group are missing the sort field,
        the group with missing field falls back to relevance-based selection."""
        docs = [
            {"_id": "a1", "title": "Running shoe premium model", "category": "group_a", "price": 100.0},
            {"_id": "a2", "title": "Running shoe budget model", "category": "group_a", "price": 50.0},
            {"_id": "a3", "title": "Running shoe deluxe model", "category": "group_a", "price": 200.0},
            {"_id": "b1", "title": "Casual leather sandal basic", "category": "group_b"},
            {"_id": "b2", "title": "Casual leather sandal premium", "category": "group_b"},
            {"_id": "b3", "title": "Casual leather sandal deluxe", "category": "group_b"},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe premium",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=10
        )

        hit_ids = [hit["_id"] for hit in res["hits"]]
        # group_a has price, so sort_by picks the cheapest (a2, price=50.0),
        # group_b falls back to relevance (b2 best match)
        self.assertEqual(["a2", "b2"], hit_ids)

    def test_collapse_sort_by_with_mixed_missing_field_within_group(self):
        """Scenario 3b: When some docs within a group have the sort field and some don't,
        the collapse still works. The relevance-based phase picks the representative,
        and the sort-based phase only processes groups whose representative has the sort field."""
        docs = [
            {"_id": "a1", "title": "Running shoe premium model", "category": "group_a", "price": 100.0},
            {"_id": "a2", "title": "Running shoe budget model", "category": "group_a"},  # missing price
            {"_id": "a3", "title": "Running shoe deluxe model", "category": "group_a", "price": 30.0},
            {"_id": "b1", "title": "Casual leather sandal basic budget", "category": "group_b", "price": 80.0},
            {"_id": "b2", "title": "Casual leather sandal premium", "category": "group_b", "price": 20.0},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe budget",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=10
        )

        hits = [hit["_id"] for hit in res["hits"]]
        # Note: group_a representative is a2 (best relevance), which lacks price,
        # so group_a falls back to relevance (a2).
        # group_b representative is b1 (best relevance), which has price, so sort_by picks b2 (cheapest).
        self.assertEqual(["a2", "b2"], hits)

    # ---- Scenario 4: All documents missing the sort by field ----

    def test_collapse_sort_by_all_docs_missing_sort_field(self):
        """Scenario 4: When ALL documents lack the sort by field, collapse falls back
        entirely to relevance-based selection (no sort-based phase runs)."""
        docs = [
            {"_id": "a1", "title": "Running shoe premium model", "category": "group_a"},
            {"_id": "a2", "title": "Running shoe budget model", "category": "group_a"},
            {"_id": "b1", "title": "Running Casual leather sandal basic", "category": "group_b"},
            {"_id": "b2", "title": "Casual leather sandal", "category": "group_b"},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe budget",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="void_field", order="asc")])
            ),
            result_count=10
        )

        hits = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(["a2", "b1"], hits)

    # ---- Scenario 5: Sort by field is a string field (non-numerical) ----

    def test_collapse_sort_by_string_field_falls_back_to_relevance(self):
        """Scenario 5: Collapse sortBy on a string field. The Vespa collapse_sort_value rank profile
        uses numeric tensor operations (marqo__score_modifiers), so string fields are not stored in that
        tensor. When the sort field is a string, the sort-based phase cannot find numeric values to sort,
        and the result falls back to relevance-based selection. We verify the search still succeeds and
        returns one representative per group."""
        docs = [
            {"_id": "a1", "title": "Running shoe premium model", "category": "group_a", "brand": "nike"},
            {"_id": "a2", "title": "Running shoe budget model", "category": "group_a", "brand": "adidas"},
            {"_id": "a3", "title": "Running shoe deluxe model", "category": "group_a", "brand": "puma"},
            {"_id": "b1", "title": "Casual leather sandal basic", "category": "group_b", "brand": "merrell"},
            {"_id": "b2", "title": "Casual leather sandal premium", "category": "group_b", "brand": "columbia"},
            {"_id": "b3", "title": "Casual leather sandal deluxe", "category": "group_b", "brand": "salomon"},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe premium model",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="brand", order="asc")])
            ),
            result_count=10
        )
        self.assertEqual(["a1", "b2"], [hit["_id"] for hit in res["hits"]])

    def test_collapse_sort_by_with_always_fetch_variants(self):
        """Scenario 5b: With alwaysFetchVariants=True, the sort-by phase runs even when the
        first-phase representative has a non-numeric (or missing) sort field value.

        In collect_document_ids, the check is:
            if always_fetch_variants or isinstance(value, (int, float))
        Without the flag, groups whose representative lacks a numeric sort field value skip the
        sort-by phase. With the flag, all groups enter the sort-by phase regardless.

        Setup: group_a's most relevant doc (a1) has no price field, group_b's most relevant doc (b1)
        has price. Without alwaysFetchVariants, group_a skips the sort-by phase (representative is a1).
        With alwaysFetchVariants=True, group_a enters the sort-by phase and picks the cheapest (a3)."""
        docs = [
            {"_id": "a1", "title": "Running shoe budget model", "category": "group_a"},
            {"_id": "a2", "title": "Running shoe premium model", "category": "group_a", "price": 100.0},
            {"_id": "a3", "title": "Running shoe deluxe model", "category": "group_a", "price": 30.0},

            {"_id": "b1", "title": "Casual leather sandal basic budget", "category": "group_b", "price": 80.0},
            {"_id": "b2", "title": "Casual leather sandal premium", "category": "group_b", "price": 20.0},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        with self.subTest("alwaysFetchVariants=False"):
            # group_a representative is a1 (best lexical match for "running shoe budget"), which lacks price
            # → sort-by skipped for group_a → a1 stays as representative
            # group_b representative is b1 (best relevance), which has price → sort-by picks b2 (cheapest)
            res_without = tensor_search.search(
                config=self.config,
                index_name=self.default_text_index.name,
                text="running shoe budget",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0,
                ),
                collapse=CollapseModel(
                    name="category",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order="asc")],
                    )
                ),
                result_count=10
            )
            # group_a falls back to relevance (a1, no price), group_b sort-by picks b2 (cheapest)
            self.assertEqual(["a1", "b2"], [hit["_id"] for hit in res_without["hits"]])

        with self.subTest("alwaysFetchVariants=True"):
            # group_a representative is still a1 (no price), but always_fetch_variants forces the sort-by phase
            # → sort-by picks a3 (cheapest at 30.0)
            # group_b is the same: sort-by picks b2 (cheapest at 20.0)
            res_with = tensor_search.search(
                config=self.config,
                index_name=self.default_text_index.name,
                text="running shoe budget",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0,
                ),
                collapse=CollapseModel(
                    name="category",
                    sort_by=CollapseSortBy(
                        fields=[CollapseSortByField(fieldName="price", order="asc")],
                        alwaysFetchVariants=True,
                    )
                ),
                result_count=10
            )
            # Now group_a enters sort-by phase and picks a3 (cheapest at 30.0), group_b still picks b2
            self.assertEqual(["a3", "b2"], [hit["_id"] for hit in res_with["hits"]])

    def test_collapse_without_sort_by_has_no_original_id(self):
        """When collapse is used without sortBy, hits should NOT have _originalId
        since no merge/replacement occurs."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe Alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(name="category"),
            result_count=10
        )

        for hit in res["hits"]:
            self.assertNotIn("_originalId", hit)

    # ---- Scenario 6: Collapse sort by with main sort by ----

    def test_collapse_sort_by_with_main_sort_by(self):
        """Scenario 6: When both main sortBy and collapse sortBy are provided,
        the main sortBy controls group ordering in the relevance phase while collapse sortBy
        picks the representative within each group in the sort phase."""
        self._add_shoe_documents()

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            sort_by=SortByModel(
                fields=[SortByField(fieldName="price", order="desc")],
                minSortCandidates=10
            ),
            result_count=10
        )

        # Collapse sortBy picks cheapest per group. Main sortBy price desc orders groups by
        # their first-phase representative's price: shoes_b max=220 > shoes_a max=200
        self.assertEqual(["shoe_b5", "shoe_a4"], [hit["_id"] for hit in res["hits"]])

    # ---- Scenario 7: Collapse sort by with main sort by + disableIfMainSortByFields ----

    def test_collapse_sort_by_disabled_when_main_sort_by_matches_disable_list(self):
        """Scenario 7: When the main query sortBy field is in disableIfMainSortByFields,
        collapse sortBy is pruned (set to None), so collapse.sortBy is not used.
        This requires going through the SearchQuery validator."""

        # Now actually execute the search with the pruned collapse model
        self._add_shoe_documents()
        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(
                    fields=[CollapseSortByField(field_name="price", order="asc")],
                    disable_if_main_sort_by_fields={"price"},  # main sortBy is on "price", so this triggers pruning
                )
            ),
            sort_by=SortByModel(
                fields=[SortByField(fieldName="price", order="desc")],
                minSortCandidates=10
            ),
            result_count=10
        )

        # Main sortBy orders by price desc. Collapse sortBy was pruned so representatives are relevance-based.
        # shoes_b1 has higher max price (180) than shoes_a (120), so shoes_b1 first.
        self.assertEqual(["shoe_b1", "shoe_a1"], [hit["_id"] for hit in res["hits"]])
        for hit in res["hits"]:
            # Since collapse.sortBy was pruned, representatives are relevance-based, so no _originalId set
            self.assertNotIn("_originalId", hit)

    def test_collapse_sort_by_kept_when_main_sort_by_not_in_disable_list(self):
        """Scenario 7b: When the main query sortBy field is NOT in disableIfMainSortByFields,
        collapse sortBy is preserved and the sort-based representative selection still works."""
        from marqo.tensor_search.models.api_models import SearchQuery

        search_query = SearchQuery(
            q="shoe",
            searchMethod="HYBRID",
            hybridParameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapseFields=[CollapseModel(
                name="category",
                sort_by=CollapseSortBy(
                    fields=[CollapseSortByField(fieldName="price", order="asc")],
                    disableIfMainSortByFields={"cost"},  # price is NOT in this set
                ),
            )],
            sortBy=SortByModel(
                fields=[SortByField(fieldName="price", order="desc")]
            ),
            limit=10
        )

        # Validator should NOT prune collapse.sortBy because "price" is not in {"cost"}
        self.assertIsNotNone(search_query.collapse_fields[0].sort_by)
        self.assertEqual("price", search_query.collapse_fields[0].sort_by.fields[0].field_name)

        # Execute the search with collapse sortBy preserved
        self._add_shoe_documents()

        collapse = search_query.collapse_fields[0]
        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=collapse,
            sort_by=SortByModel(
                fields=[SortByField(fieldName="price", order="desc")],
                minSortCandidates=10
            ),
            result_count=10
        )

        # Collapse sortBy asc picks cheapest per group. Main sortBy price desc orders groups by
        # their first-phase representative's price: shoes_b max=220 > shoes_a max=200
        self.assertEqual(["shoe_b5", "shoe_a4"], [hit["_id"] for hit in res["hits"]])

    # ---- Scenario 8: Other edge cases ----

    def test_collapse_sort_by_single_group(self):
        """Scenario 8a: Collapse sortBy when all documents belong to a single group."""
        docs = [
            {"_id": f"doc{i}", "title": f"Alpha product variant {i}", "category": "only_group",
             "price": float(10 * (i + 1))}
            for i in range(5)
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="Alpha product",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=10
        )

        self.assertEqual(["doc0"], [hit["_id"] for hit in res["hits"]])

    def test_collapse_sort_by_with_attributes_to_retrieve(self):
        """Scenario 8b: Collapse sortBy respects attributes_to_retrieve."""
        self._add_shoe_documents()

        original_attributes_to_retrieve = ["title", "price"]

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="desc")])
            ),
            attributes_to_retrieve=original_attributes_to_retrieve,  # "price" is included
            result_count=10
        )

        self.assertEqual(["title", "price"], original_attributes_to_retrieve)
        self.assertEqual(["shoe_a5", "shoe_b3"], [hit["_id"] for hit in res["hits"]])
        hit_0 = res["hits"][0]
        expected_hit_0 = {
            "_id": "shoe_a5",
            "title": "Running Shoe Epsilon",
            "price": 200.0,
            "_originalId": "shoe_a1",
            "_highlights": [{}],
            "_score": hit_0["_score"],  # score is dynamic
        }

        self.assertEqual(expected_hit_0, hit_0)
        hit_1 = res["hits"][1]
        expected_hit_1 = {
            "_id": "shoe_b3",
            "title": "Hiking Boot Gamma",
            "price": 220.0,
            "_highlights": [{}],
            "_originalId": "shoe_b1",
            "_score": hit_1["_score"],  # score is dynamic
        }
        self.assertEqual(expected_hit_1, hit_1)

    def test_collapse_sort_by_works_when_sort_field_not_in_attributes_to_retrieve(self):
        """Collapse sortBy should work even when the sort_by field is not in attributes_to_retrieve.

        The sort_by field (e.g., 'price') is automatically added to attributes_to_retrieve internally
        so that collect_parent_ids() can access the field value for sorting decisions.
        """
        self._add_shoe_documents()

        original_attributes_to_retrieve = ["title"]

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running shoe alpha",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            attributes_to_retrieve=original_attributes_to_retrieve,
            result_count=10
        )

        # Ensure original attributes_to_retrieve is unchanged
        self.assertEqual(["title"], original_attributes_to_retrieve)

        self.assertEqual(["shoe_a4", "shoe_b5"], [hit["_id"] for hit in res["hits"]])
        hit_0 = res["hits"][0]
        expected_hit_0 = {
            "_id": "shoe_a4",
            "title": "Running Shoe Delta",
            "_originalId": "shoe_a1",
            "_highlights": [{}],
            "_score": hit_0["_score"],  # score is dynamic
        }

        self.assertEqual(expected_hit_0, hit_0)
        hit_1 = res["hits"][1]
        expected_hit_1 = {
            "_id": "shoe_b5",
            "title": "Hiking Boot Epsilon",
            "_highlights": [{}],
            "_originalId": "shoe_b1",
            "_score": hit_1["_score"],  # score is dynamic
        }
        self.assertEqual(expected_hit_1, hit_1)

    def test_collapse_sort_by_many_groups(self):
        """Scenario 8c: Collapse sortBy with many groups and pagination."""
        docs = [
            {"_id": f"doc{g}{i}", "title": f"Alpha product variant {g} {i}",
             "category": f"group_{g}", "price": float(10 * (i + 1))}
            for g in range(8) for i in range(3)
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="Alpha product",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=5
        )

        # Should get at most 5 groups (limited by result_count)
        self.assertLessEqual(len(res["hits"]), 5)
        # Each hit should have the cheapest price for its group (10.0)
        for hit in res["hits"]:
            self.assertEqual(10.0, hit["price"])
        # All groups should be unique
        categories = [hit["category"] for hit in res["hits"]]
        self.assertEqual(len(categories), len(set(categories)))

    def test_collapse_search_sort_by_with_facets(self):
        self._add_shoe_documents()

        collapse_search_results_without_sort_by = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
            ),
            facets=FacetsParameters(
                fields={"price": FieldFacetsConfiguration(type="number")}
            ),
            result_count=10
        )

        collapse_search_results_with_sort_by = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="shoe",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            facets=FacetsParameters(
                fields={"price": FieldFacetsConfiguration(type="number")}
            ),
            result_count=10
        )

        # Facets should be identical regardless of collapse sort_by usage
        self.assertEqual(
            collapse_search_results_without_sort_by["facets"],
            collapse_search_results_with_sort_by["facets"]
        )


class TestCollapseSortByTieBreaker(MarqoTestCase):
    """Integration tests for collapse fields with sort by functionality in tie-breaking scenarios.

    This test class must pass on the multi-shard environment to ensure that the collapse sort by logic correctly
    handles tie-breaking when multiple documents have the same sort field value. The tests cover scenarios where
    multiple documents within a group have the same price, and we verify that the collapse sort by consistently
    selects the same representative document based on relevance as a tie-breaker.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            collapse_fields=[CollapseField(name="category", minGroups=2)]
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
        ])

        cls.default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

    def _add_shoe_documents(self):
        """Add sample shoe documents across two category groups with varying prices."""
        basic_docs = [
            # Two basic shoes
            {"_id": "shoe_a1", "title": "Running Shoe Alpha", "category": "shoes_a", "price": 120.0, "cost": 95.0,
             "brand": "nike"},
            {"_id": "shoe_b1", "title": "Walking boot Alpha", "category": "shoes_b", "price": 89.99, "cost": 70.0,
             "brand": "adidas"},
        ]

        # A lot of identical variants with the same low cost to force ties in collapse sort by
        shoe_a_variants = [
            {
                "_id": f"shoe_a{i}", "title": f"variants",
                "category": "shoes_a", "price": 120.0, "cost": 15.0,
            }
            for i in range(2, 10)
        ]

        # A lot of identical variants with the same low cost to force ties in collapse sort by
        shoe_b_variants = [
            {
                "_id": f"shoe_b{i}", "title": f"variants",
                "category": "shoes_b", "price": 120.0, "cost": 1.0,
            }
            for i in range(2, 6)
        ]

        docs = basic_docs + shoe_a_variants + shoe_b_variants
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

    @pytest.mark.skip_for_multinode(reason="We don't have a reliable tie-breaker for multi-nodes.")
    def test_collapse_sort_by_with_ties(self):
        """When multiple documents within a group have the same sort field value (price),
        the collapse sort by should consistently select the same representative based on a tie-breaker."""
        self._add_shoe_documents()

        def get_results():
            res = tensor_search.search(
                config=self.config,
                index_name=self.default_text_index.name,
                text="running shoe boot alpha",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0,
                ),
                collapse=CollapseModel(
                    name="category",
                    sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="cost", order="asc")])
                ),
                result_count=10
            )
            return [hit["_id"] for hit in res["hits"]] + [hit.get("_originalId") for hit in res["hits"]]

        first_result = get_results()

        for i in range(10):
            # Run the same search multiple times to verify that the same representative is consistently selected
            result = get_results()
            self.assertEqual(
                first_result, result,
                f"Collapse sort by with ties should consistently select the same representative "
                f"document in the single node setup, however got different results across runs. "
                f"Expected result: {first_result}, Returned result: {result}, in the run {i + 1}/10"
            )

    def test_collapse_sort_by_keeps_relevance_hit_on_tie_asc(self):
        """When all documents in a group share the same sort value with order=asc,
        the sorted variant is not strictly cheaper, so the relevance representative is kept"""
        docs = [
            # shoes_a: all same price -> tie
            {"_id": "shoe_a1", "title": "Running Shoe Alpha", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a2", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a3", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a4", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a5", "title": "variants", "category": "shoes_a", "price": 100.0},
            # shoes_b: all same price -> tie
            {"_id": "shoe_b1", "title": "Hiking Boot Alpha", "category": "shoes_b", "price": 100.0},
            {"_id": "shoe_b2", "title": "variants", "category": "shoes_b", "price": 100.0},
            {"_id": "shoe_b3", "title": "variants", "category": "shoes_b", "price": 100.0},
        ]
        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running alpha shoe boot",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="asc")])
            ),
            result_count=10
        )
        self.assertEqual(["shoe_a1", "shoe_b1"], [hit["_id"] for hit in res["hits"]])
        for hits in res["hits"]:
            self.assertNotIn("_originalId", hits)

    def test_collapse_sort_by_keeps_relevance_hit_on_tie_desc(self):
        """When all documents in a group share the same sort value with order=desc,
        the sorted variant is not strictly higher, so the relevance representative is kept."""
        docs = [
            {"_id": "shoe_a1", "title": "Running Shoe Alpha", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a2", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a3", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a4", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_a5", "title": "variants", "category": "shoes_a", "price": 100.0},
            {"_id": "shoe_b1", "title": "Hiking Boot Alpha", "category": "shoes_b", "price": 100.0},
            {"_id": "shoe_b2", "title": "variants", "category": "shoes_b", "price": 100.0},
            {"_id": "shoe_b3", "title": "variants", "category": "shoes_b", "price": 100.0},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="running alpha shoe boot",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0,
            ),
            collapse=CollapseModel(
                name="category",
                sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price", order="desc")])
            ),
            result_count=10
        )
        self.assertEqual(["shoe_a1", "shoe_b1"], [hit["_id"] for hit in res["hits"]])
        for hits in res["hits"]:
            self.assertNotIn("_originalId", hits)