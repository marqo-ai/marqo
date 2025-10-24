import uuid

import numpy as np
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestScoreModifierSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.unstructured_score_modifier_index_name = ("unstructured_score_modifier"
                                                      + str(uuid.uuid4()).replace('-', ''))
        cls.structured_score_modifier_index_name = "structured_score_modifier" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.unstructured_score_modifier_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion400m_e31"
            },
            {
                "indexName": cls.structured_score_modifier_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/laion400m_e31",
                "allFields": [
                    {"name": "text_field", "type": "text", "features": ["lexical_search"]},
                    {"name": "image_field", "type": "image_pointer"},
                    {"name": "multiply_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "multiply_2", "type": "float", "features": ["score_modifier"]},
                    {"name": "add_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "add_2", "type": "float", "features": ["score_modifier"]},
                ],
                "tensorFields": ["text_field", "image_field"]
            }
        ]
        )

        cls.indexes_to_delete = [cls.structured_score_modifier_index_name,
                                 cls.unstructured_score_modifier_index_name]

    def test_score_modifier_search_results(self):
        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            for _ in range(10):
                # Generate 8 random values to test score modifiers
                multiply_1_value, multiply_1_weight, multiply_2_value, multiply_2_weight, \
                    add_1_value, add_1_weight, add_2_value, add_2_weight = \
                    np.round(np.random.uniform(-10, 10, 8), 2)

                doc = {
                    "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue_small.png",
                    "text_field": "Marqo can support vector search",
                    "multiply_1": multiply_1_value,
                    "multiply_2": multiply_2_value,
                    "add_1": add_1_value,
                    "add_2": add_2_value,
                }

                score_modifiers = {
                    "multiply_score_by":
                        [{"field_name": "multiply_1", "weight": multiply_1_weight},
                         {"field_name": "multiply_2", "weight": multiply_2_weight}],
                    "add_to_score":
                        [{"field_name": "add_1", "weight": add_1_weight},
                         {"field_name": "add_2", "weight": add_2_weight}]
                }
                msg = (f"{'unstructured' if index_name.startswith('un') else 'structured'}, doc = {doc}, "
                       f"score_modifiers = {score_modifiers}")

                with self.subTest(msg):
                    self.clear_indexes(self.indexes_to_delete)
                    res = self.client.index(index_name).add_documents(
                        documents=[doc], tensor_fields=["text_field", "image_field"] if msg.startswith("un") else None,
                    )
                    self.assertEqual(1, self.client.index(index_name).get_stats()["numberOfDocuments"])

                    original_res = self.client.index(index_name).search(q="test_search", score_modifiers=None)
                    modifiers_res = self.client.index(index_name).search(q="test_search",
                                                                         score_modifiers=score_modifiers)
                    original_score = original_res["hits"][0]["_score"]
                    modifiers_score = modifiers_res["hits"][0]["_score"]

                    expected_score = original_score * multiply_1_value * multiply_1_weight * multiply_2_value \
                                     * multiply_2_weight + add_1_value * add_1_weight + add_2_value * add_2_weight

                    self.assertAlmostEqual(expected_score, modifiers_score, delta=1e-3)

    def test_invalid_score_modifiers_format(self):
        invalid_score_modifiers = {
            # typo in multiply score by
            "multiply_score_bys":
                [{"field_name": "multiply_1",
                  "weight": 1, },
                 {"field_name": "multiply_2", }],
            "add_to_score": [
                {"field_name": "add_1", "weight": 4,
                 },
                {"field_name": "add_2", "weight": 1,
                 }]
        }

        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            with self.subTest(index_name):
                with self.assertRaises(MarqoWebError) as e:
                    res = self.client.index(index_name).search(
                        "query", score_modifiers=invalid_score_modifiers
                    )

                self.assertIn("score_modifiers", str(e.exception.message))


    def test_valid_score_modifiers_format(self):
        valid_score_modifiers_list = [
            {
                # missing add to score
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing multiply score by
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1, },
                     {"field_name": "multiply_2"}],
            }]

        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            for valid_score_modifiers in valid_score_modifiers_list:
                with self.subTest(f"{index_name} - {valid_score_modifiers}"):
                    self.client.index(index_name).search("test", score_modifiers=valid_score_modifiers)

    def test_hybrid_search_rrf_score_modifiers_with_rerank_depth(self):
        """
        Test that hybrid search with RRF can use root level score_modifiers and rerank_depth
        """

        docs_list = [
            {"_id": "both1", "text_field": "dogs", "multiply_1": -1, "add_1": -1},           # HIGH tensor, LOW lexical
            {"_id": "tensor1", "text_field": "puppies", "multiply_1": 2, "add_1": 2},         # MID tensor
            {"_id": "tensor2", "text_field": "random words", "multiply_1": 3, "add_1": 3},    # LOW tensor
        ]

        for test_index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            with self.subTest(index=test_index_name):
                self.client.index(test_index_name).add_documents(
                    docs_list,
                    tensor_fields=["text_field"] if "unstr" in test_index_name else None)

                # Get unmodified scores
                # Unmodified result order should be: both1, tensor1, tensor2
                unmodified_results = self.client.index(test_index_name).search(q="dogs", search_method="HYBRID",limit=3)
                unmodified_scores = {hit["_id"]: hit["_score"] for hit in unmodified_results["hits"]}
                self.assertEqual(["both1", "tensor1", "tensor2"], [hit["_id"] for hit in unmodified_results["hits"]])

                # Get modified scores (rank all 3)
                # Modified result order should be: tensor2, tensor1, both1
                score_modifiers = {
                    "multiply_score_by": [
                        {"field_name": "multiply_1", "weight": 1}
                    ],
                    "add_to_score": [
                        {"field_name": "add_1", "weight": 1}
                    ]
                }
                with self.subTest(f"Case 1: limit == rerankCount == hits.size()"):
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=3, rerank_depth=3, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor2", "tensor1", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], 3*unmodified_scores["tensor2"] + 3)
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], 2*unmodified_scores["tensor1"] + 2)
                    self.assertAlmostEqual(modified_results["hits"][2]["_score"], -1*unmodified_scores["both1"] - 1)

                    # Get modified scores (rank only 1). Only both1 should be rescored (goes to the bottom)
                    # Modified result order should be: tensor1, tensor2, both1
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=3, rerank_depth=1, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor1", "tensor2", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], unmodified_scores["tensor1"])     # unmodified
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], unmodified_scores["tensor2"])     # unmodified
                    self.assertAlmostEqual(modified_results["hits"][2]["_score"], -1*unmodified_scores["both1"] - 1)    # modified

                with self.subTest(f"Case 2: limit < rerankCount < hits.size()"):
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=1, rerank_depth=2, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], -1*unmodified_scores["both1"] - 1)

                with self.subTest(f"Case 3: limit == rerankCount < hits.size()"):
                    # tensor2 never appears, because it is the 3rd result out of tensor
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=2, rerank_depth=2, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor1", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], 2*unmodified_scores["tensor1"] + 2)
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], -1*unmodified_scores["both1"] - 1)

                with self.subTest(f"Case 4: limit < hits.size() < rerankCount"):
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=2, rerank_depth=10, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor1", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], 2 * unmodified_scores["tensor1"] + 2)
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], -1 * unmodified_scores["both1"] - 1)

                with self.subTest(f"Case 5: rerankCount < hits.size() < limit"):
                    # tensor2 remains unmodified.
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=10, rerank_depth=2, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor1", "tensor2", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], 2 * unmodified_scores["tensor1"] + 2)
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], unmodified_scores["tensor2"])
                    self.assertAlmostEqual(modified_results["hits"][2]["_score"], -1 * unmodified_scores["both1"] - 1)

                with self.subTest("Case 6: rerankCount == 0"):
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID", hybrid_parameters={"verbose": True},
                        limit=3, rerank_depth=0, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["both1", "tensor1", "tensor2"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], unmodified_scores["both1"])
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], unmodified_scores["tensor1"])
                    self.assertAlmostEqual(modified_results["hits"][2]["_score"], unmodified_scores["tensor2"])

                with self.subTest("Case 7: No rerankCount"):
                    modified_results = self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        limit=3, score_modifiers=score_modifiers
                    )
                    self.assertEqual(["tensor2", "tensor1", "both1"], [hit["_id"] for hit in modified_results["hits"]])
                    self.assertAlmostEqual(modified_results["hits"][0]["_score"], 3 * unmodified_scores["tensor2"] + 3)
                    self.assertAlmostEqual(modified_results["hits"][1]["_score"], 2 * unmodified_scores["tensor1"] + 2)
                    self.assertAlmostEqual(modified_results["hits"][2]["_score"], -1 * unmodified_scores["both1"] - 1)

    def test_global_score_modifiers_wrong_retrieval_or_ranking_fails(self):
        """
        Test that providing score modifiers at the root level for non-RRF hybrid search fails
        """
        # TODO: remove when we support this
        for test_index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            with self.subTest(index=test_index_name):
                for retrieval_method, ranking_method in [
                    ("tensor", "tensor"),
                    ("tensor", "lexical"),
                    ("lexical", "tensor"),
                    ("lexical", "lexical"),
                ]:
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(test_index_name).search(
                            q="dogs", search_method="HYBRID",
                            hybrid_parameters={
                                "retrievalMethod": retrieval_method,
                                "rankingMethod": ranking_method,
                            },
                            score_modifiers={"multiply_score_by": [{"field_name": "multiply_1", "weight": 1}],
                                             "add_to_score": [{"field_name": "add_1", "weight": 1}]},
                        )
                    self.assertEqual(400, e.exception.status_code)
                    self.assertIn("only supported for hybrid search if \\'rankingMethod\\' is \\'RRF\\'", str(e.exception.message))

    def test_negative_rerank_depth_fails(self):
        """
        Tests that creating a search query with rerank_depth fails if the value is negative.
        """
        for test_index_name in [self.unstructured_score_modifier_index_name,
                                self.structured_score_modifier_index_name]:
            with self.assertRaises(MarqoWebError) as e:
                with self.subTest(index=test_index_name):
                    self.client.index(test_index_name).search(
                        q="dogs", search_method="HYBRID",
                        rerank_depth=-5
                    )
            self.assertEqual(422, e.exception.status_code)
            self.assertIn("rerankDepth cannot be negative",
                          str(e.exception.message))

