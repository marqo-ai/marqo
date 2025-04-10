import copy
import uuid
from unittest import mock

import marqo
from marqo import enums
from marqo.client import Client
from marqo.enums import SearchMethods
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase, EXAMPLE_FASHION_DOCUMENTS


class TestStructuredHybridSearch(MarqoTestCase):
    text_index_name = "api_test_structured_index_text" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "api_test_structured_image_index_image" + str(uuid.uuid4()).replace('-', '')
    unstructured_text_index_name = "api_test_unstructured_index_text" + str(uuid.uuid4()).replace('-', '')
    unstructured_image_index_name = "api_test_unstructured_image_index_image" + str(uuid.uuid4()).replace('-', '')

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
                "indexName": cls.image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "text_field_1", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_2", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_3", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_field_1", "type": "image_pointer"},
                    {"name": "image_field_2", "type": "image_pointer"},
                    {"name": "list_field_1", "type": "array<text>", "features": ["filter"]}
                ],
                "tensorFields": ["text_field_1", "text_field_2", "text_field_3", "image_field_1", "image_field_2"],
            },
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai",
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name, cls.unstructured_text_index_name,
                                 cls.unstructured_image_index_name]

    def setUp(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

        self.docs_list = [
            # similar semantics to dogs
            {"_id": "doc1", "text_field_1": "dogs"},
            {"_id": "doc2", "text_field_1": "puppies"},
            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
            {"_id": "doc4", "text_field_1": "huskies"},
            {"_id": "doc5", "text_field_1": "four-legged animals"},

            # shares lexical token with dogs
            {"_id": "doc6", "text_field_1": "hot dogs"},
            {"_id": "doc7", "text_field_1": "dogs is a word"},
            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
            {"_id": "doc9", "text_field_1": "dogs random words"},
            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
            {"_id": "doc13", "text_field_2": "canines canines"},
        ]
            
    @staticmethod
    def strip_marqo_fields(doc, strip_id=True):
        """Strips Marqo fields from a returned doc to get the original doc"""
        copied = copy.deepcopy(doc)

        strip_fields = ["_highlights", "_score"]
        if strip_id:
            strip_fields += ["_id"]

        for to_strip in strip_fields:
            del copied[to_strip]

        return copied

    def test_hybrid_search_with_custom_vector_query(self):
        """
        Custom Vectory q should work similar to None q with a context vector
        """
        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                    if "unstructured" in index_name else None
                )
                sample_vector = [0.5 for _ in range(384)]

                res_custom_vector = self.client.index(index_name).search(
                    q={"customVector": {"content": None, "vector": sample_vector}},
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": "tensor",
                        "rankingMethod": "tensor"
                    }
                )

                res_context = self.client.index(index_name).search(
                    q=None,
                    search_method="TENSOR",
                    context={"tensor": [{"vector": sample_vector, "weight": 1}]}
                )
                self.assertEqual(len(res_custom_vector["hits"]), len(res_context["hits"]))
                for i in range(len(res_custom_vector["hits"])):
                    self.assertEqual(res_custom_vector["hits"][i]["_id"], res_context["hits"][i]["_id"])

    def test_hybrid_search_disjunction_rrf_zero_alpha_same_as_lexical(self):
        """
        Tests that hybrid search with:
        retrievalMethod = "disjunction"
        rankingMethod = "rrf"
        alpha = 0.0

        is the same as a lexical search (in terms of result order).
        """

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                hybrid_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": "disjunction",
                        "rankingMethod": "rrf",
                        "alpha": 0
                    },
                    limit=10
                )

                lexical_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="LEXICAL",
                    limit=10
                )

                self.assertEqual(len(hybrid_res["hits"]), len(lexical_res["hits"]))
                for i in range(len(hybrid_res["hits"])):
                    self.assertEqual(hybrid_res["hits"][i]["_id"], lexical_res["hits"][i]["_id"])

    def test_hybrid_search_disjunction_rrf_one_alpha_same_as_tensor(self):
        """
        Tests that hybrid search with:
        retrievalMethod = "disjunction"
        rankingMethod = "rrf"
        alpha = 1.0

        is the same as a tensor search (in terms of result order).
        """

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                hybrid_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": "disjunction",
                        "rankingMethod": "rrf",
                        "alpha": 1
                    },
                    limit=10
                )

                tensor_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="TENSOR",
                    limit=10
                )

                self.assertEqual(len(hybrid_res["hits"]), len(tensor_res["hits"]))
                for i in range(len(hybrid_res["hits"])):
                    self.assertEqual(hybrid_res["hits"][i]["_id"], tensor_res["hits"][i]["_id"])

    def test_hybrid_search_searchable_attributes(self):
        """
        Tests that searchable attributes work as expected for all methods
        """
        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                with self.subTest("retrieval: disjunction, ranking: rrf"):
                    hybrid_res = self.client.index(index_name).search(
                        "puppies",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "alpha": 0.5,
                            "searchableAttributesLexical": ["text_field_2"],
                            "searchableAttributesTensor": ["text_field_2"]
                        },
                        limit=10
                    )
                    self.assertEqual(len(hybrid_res["hits"]), 3)  # Only 3 documents have text_field_2 at all
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")  # puppies puppies in text field 2
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc13")
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc11")

                with self.subTest("retrieval: lexical, ranking: tensor"):
                    hybrid_res = self.client.index(index_name).search(
                        "puppies",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "lexical",
                            "rankingMethod": "tensor",
                            "searchableAttributesLexical": ["text_field_2"]
                        },
                        limit=10
                    )
                    self.assertEqual(len(hybrid_res["hits"]),
                                        1)  # Only 1 document has puppies in text_field_2. Lexical retrieval will only get this one.
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")

                with self.subTest("retrieval: tensor, ranking: lexical"):
                    hybrid_res = self.client.index(index_name).search(
                        "puppies",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "tensor",
                            "rankingMethod": "lexical",
                            "searchableAttributesTensor": ["text_field_2"]
                        },
                        limit=10
                    )
                    self.assertEqual(len(hybrid_res["hits"]),
                                        3)  # Only 3 documents have text field 2. Tensor retrieval will get them all.
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")
                    # doc11 and doc13 has score 0, so their order is non-deterministic
                    self.assertSetEqual({'doc11', 'doc13'}, {hit["_id"] for hit in hybrid_res["hits"][1:]})

    def test_hybrid_search_score_modifiers(self):
        """
        Tests that score modifiers work as expected for all methods
        """
        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                # Add documents
                self.client.index(index_name).add_documents(
                    [
                        {"_id": "doc6", "text_field_1": "HELLO WORLD"},
                        {"_id": "doc7", "text_field_1": "HELLO WORLD", "add_field_1": 1.0},  # third
                        {"_id": "doc8", "text_field_1": "HELLO WORLD", "mult_field_1": 2.0},  # second highest score
                        {"_id": "doc9", "text_field_1": "HELLO WORLD", "mult_field_1": 3.0},  # highest score
                        {"_id": "doc10", "text_field_1": "HELLO WORLD", "mult_field_2": 3.0},  # lowest score
                    ],
                    tensor_fields=["text_field_1"] if "unstructured" in index_name else None
                )

                with self.subTest("retrieval: lexical, ranking: tensor"):
                    hybrid_res = self.client.index(index_name).search(
                        "HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "lexical",
                            "rankingMethod": "tensor",
                            "scoreModifiersTensor":{
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                        },
                        limit=10
                    )
                    # TODO investigate why the score modifier has changed
                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")  # highest score (score*10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][0]["_score"], 30.0, places=4)
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # (score*10*2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_score"], 20.0, places=4)
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # (score + 5*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_score"], 6.0, places=4)
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")  # (score)
                    self.assertAlmostEqual(hybrid_res["hits"][3]["_score"], 1.0, places=4)
                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_score"], -30.0, places=4)

                with self.subTest("retrieval: tensor, ranking: lexical"):
                    hybrid_res = self.client.index(index_name).search(
                        "HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "tensor",
                            "rankingMethod": "lexical",
                            "scoreModifiersLexical":{
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 2}
                                ]
                            },
                        },
                        limit=10
                    )
                    self.assertIn("hits", hybrid_res)

                    base_lexical_score = hybrid_res["hits"][3]["_score"]
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")  # highest score (score*10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][0]["_score"], base_lexical_score * 10 * 3)
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # second highest score (score*10*2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_score"], base_lexical_score * 10 * 2)
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # third highest score (score + 2*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_score"], base_lexical_score + 2 * 1)
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")  # ORIGINAL SCORE
                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_score"], base_lexical_score * -10 * 3)

                with self.subTest("retrieval: disjunction, ranking: rrf"):
                    hybrid_res = self.client.index(index_name).search(
                        "HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "scoreModifiersLexical":{
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                            "scoreModifiersTensor":{
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                        },
                        limit=10
                    )
                    self.assertIn("hits", hybrid_res)

                    # Score without score modifiers
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")  # (score)
                    base_lexical_score = hybrid_res["hits"][3]["_lexical_score"]
                    base_tensor_score = hybrid_res["hits"][3]["_tensor_score"]

                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")  # highest score (score*10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][0]["_lexical_score"], base_lexical_score * 10 * 3)
                    self.assertEqual(hybrid_res["hits"][0]["_tensor_score"], base_tensor_score * 10 * 3)

                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # (score*10*2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_lexical_score"], base_lexical_score * 10 * 2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_tensor_score"], base_tensor_score * 10 * 2)

                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # (score + 5*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_lexical_score"], base_lexical_score + 5 * 1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_tensor_score"], base_tensor_score + 5 * 1)

                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_lexical_score"], base_lexical_score * -10 * 3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_tensor_score"], base_tensor_score * -10 * 3)

    def test_hybrid_search_same_retrieval_and_ranking_matches_original_method(self):
        """
        Tests that hybrid search with:
        retrievalMethod = "lexical", rankingMethod = "lexical" and
        retrievalMethod = "tensor", rankingMethod = "tensor"

        Results must be the same as lexical search and tensor search respectively.
        """

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                test_cases = [
                    ("lexical", "lexical"),
                    ("tensor", "tensor")
                ]

                for retrievalMethod, rankingMethod in test_cases:
                    with self.subTest(retrieval=retrievalMethod, ranking=rankingMethod):
                        hybrid_res = self.client.index(index_name).search(
                            "dogs",
                            search_method="HYBRID",
                            hybrid_parameters={
                                "retrievalMethod": retrievalMethod,
                                "rankingMethod": rankingMethod
                            },
                            limit=10
                        )

                        base_res = self.client.index(index_name).search(
                            "dogs",
                            search_method=retrievalMethod,     # will be either lexical or tensor
                            limit=10
                        )

                        self.assertEqual(len(hybrid_res["hits"]), len(base_res["hits"]))
                        for i in range(len(hybrid_res["hits"])):
                            self.assertEqual(hybrid_res["hits"][i]["_id"], base_res["hits"][i]["_id"])

    def test_hybrid_search_with_filter(self):
        """
        Tests that filter is applied correctly in hybrid search.
        """

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                test_cases = [
                    ("disjunction", "rrf"),
                    ("lexical", "lexical"),
                    ("tensor", "tensor")
                ]

                for retrievalMethod, rankingMethod in test_cases:
                    with self.subTest(retrieval=retrievalMethod, ranking=rankingMethod):
                        hybrid_res = self.client.index(index_name).search(
                            "dogs",
                            search_method="HYBRID",
                            filter_string="text_field_1:(something something dogs)",
                            hybrid_parameters={
                                "retrievalMethod": retrievalMethod,
                                "rankingMethod": rankingMethod
                            },
                            limit=10
                        )

                        self.assertEqual(len(hybrid_res["hits"]), 1)
                        self.assertEqual(hybrid_res["hits"][0]["_id"], "doc8")

    def test_hybrid_search_invalid_parameters_fails(self):
        test_cases = [
            ({
                 "alpha": 0.6,
                 "rankingMethod": "tensor"
             }, "can only be defined for 'rrf'"),
            ({
                 "rrfK": 61,
                 "rankingMethod": "normalize_linear"
             }, "can only be defined for 'rrf'"),
            ({
                 "rrfK": 60.1,
             }, "must be an integer"),
            ({
                "alpha": 1.1
            }, "between 0 and 1"),
            ({
                 "rrfK": -1
             }, "greater than or equal to 0"),
            ({
                "retrievalMethod": "disjunction",
                "rankingMethod": "lexical"
            }, "rankingMethod must be: rrf"),
            ({
                 "retrievalMethod": "tensor",
                 "rankingMethod": "rrf"
             }, "rankingMethod must be: tensor or lexical"),
            ({
                 "retrievalMethod": "lexical",
                 "rankingMethod": "rrf"
             }, "rankingMethod must be: tensor or lexical"),
            # Searchable attributes need to match retrieval method
            ({
                "retrievalMethod": "tensor",
                "rankingMethod": "tensor",
                "searchableAttributesLexical": ["text_field_1"]
             }, "can only be defined for 'lexical',"),
            ({
                "retrievalMethod": "lexical",
                "rankingMethod": "lexical",
                "searchableAttributesTensor": ["text_field_1"]
             }, "can only be defined for 'tensor',"),
            # Score modifiers need to match ranking method
            ({
                 "retrievalMethod": "tensor",
                 "rankingMethod": "tensor",
                 "scoreModifiersLexical": {
                     "multiply_score_by": [
                         {"field_name": "mult_field_1", "weight": 1.0}
                     ]
                 },
             }, "can only be defined for 'lexical',"),
            ({
                 "retrievalMethod": "lexical",
                 "rankingMethod": "lexical",
                 "scoreModifiersTensor": {
                    "multiply_score_by": [
                        {"field_name": "mult_field_1", "weight": 1.0}
                    ]
                 }
             }, "can only be defined for 'tensor',"),
            # Non-existent retrieval method
            ({"retrievalMethod": "something something"},
                "not a valid enumeration member"),
            # Non-existent ranking method
            ({"rankingMethod": "something something"},
                "not a valid enumeration member")
        ]

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                for hybrid_parameters, error_message in test_cases:
                    with self.subTest(hybrid_parameters=hybrid_parameters):
                        with self.assertRaises(MarqoWebError) as e:
                            res = self.client.index(index_name).search(
                                "dogs",
                                search_method="HYBRID",
                                hybrid_parameters=hybrid_parameters
                            )
                        self.assertIn(error_message, str(e.exception))

    def test_hybrid_search_structured_invalid_fields_fails(self):
        """
        If searching with HYBRID, searchableAttributesLexical must only have lexical fields, and
        searchableAttributesTensor must only have tensor fields.
        """
        # Non-lexical field
        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor")
        ]

        for retrievalMethod, rankingMethod in test_cases:
            with self.subTest(retrieval=retrievalMethod, ranking=rankingMethod):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(self.text_index_name).search(
                        "dogs",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod":retrievalMethod,
                            "rankingMethod":rankingMethod,
                            "searchableAttributesLexical":["text_field_1", "add_field_1"]
                        }
                    )
                self.assertIn("has no lexically searchable field add_field_1", str(e.exception))

        # Non-tensor field
        test_cases = [
            ("disjunction", "rrf"),
            ("tensor", "tensor"),
            ("tensor", "lexical")
        ]
        for retrievalMethod, rankingMethod in test_cases:
            with self.subTest(retrieval=retrievalMethod, ranking=rankingMethod):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(self.text_index_name).search(
                        "dogs",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": retrievalMethod,
                            "rankingMethod": rankingMethod,
                            "searchableAttributesTensor": ["mult_field_1", "text_field_1"]
                        }
                    )
                self.assertIn("has no tensor field mult_field_1", str(e.exception))

    def test_hybrid_parameters_with_wrong_search_method_fails(self):
        """
        Tests that providing hybrid parameters with a wrong search method fails.
        """

        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).search(
                        "dogs",
                        search_method="LEXICAL",
                        hybrid_parameters={
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                        }
                    )
                self.assertIn("can only be provided for 'HYBRID'", str(e.exception))

    def test_hybrid_search_default_parameters(self):
        """
        Test hybrid search when no hybrid parameters are provided.
        Search results should exactly match that of disjunction, rrf
        """
        for index_name in [self.text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name).add_documents(
                    self.docs_list,
                    tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                        if "unstructured" in index_name else None
                )

                default_hybrid_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="HYBRID",
                    limit=10
                )

                disjunction_res = self.client.index(index_name).search(
                    "dogs",
                    search_method="HYBRID",
                    hybrid_parameters={
                        "retrievalMethod": "disjunction",
                        "rankingMethod": "rrf",
                        "alpha": 0.5,
                        "rrfK": 60
                    },
                    limit=10
                )

                self.assertEqual(len(default_hybrid_res["hits"]), len(disjunction_res["hits"]))
                for i in range(len(default_hybrid_res["hits"])):
                    self.assertEqual(default_hybrid_res["hits"][i]["_id"], disjunction_res["hits"][i]["_id"])
                    self.assertEqual(default_hybrid_res["hits"][i]["_score"], disjunction_res["hits"][i]["_score"])

    # TODO: test_hybrid_search_with_images
    # TODO: test_hybrid_search_opposite_retrieval_and_ranking