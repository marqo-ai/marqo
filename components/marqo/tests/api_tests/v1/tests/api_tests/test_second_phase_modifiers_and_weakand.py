import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestRelevanceCutoffFeature(MarqoTestCase):
    unstructured_index_name = f"test_relevance_cutoff_feature_unstructured_{uuid.uuid4()}"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.create_indexes(
            [
                {
                    "indexName": cls.unstructured_index_name,
                    "type": "unstructured",
                    "model": "hf/all-MiniLM-L6-v2",
                }
            ]
        )

        irrelevant_docs = [
            {
                '_id': f'{doc_id}',
                'text': 'Irrelevant documents but has a score modifier field.',
                "score_modifier_value": 100.0
            }
            for doc_id in range(10)
        ]

        relevant_docs = [
            {
                "_id": 'relevant_0',
                "text": "This is a relevant documents without score modifier.",
            }
        ]

        cls.client.index(cls.unstructured_index_name).add_documents(
            irrelevant_docs + relevant_docs,
            tensor_fields=["text"]
        )

        cls.indexes_to_delete = [cls.unstructured_index_name]

    def setUp(self):
        pass  # To override MarqoTestCase setUp that clears indexes

    def test_default_behavior_still_apples_score_modifiers_in_the_first_phase(self):
        response = self.client.index(self.unstructured_index_name).search(
            "relevant documents",
            search_method="HYBRID",
            hybrid_parameters={
                "rankingMethod": "lexical",
                "retrievalMethod": "lexical",
                "scoreModifiersLexical": {
                    "add_to_score": [{"field_name": "score_modifier_value", "weight": 1.0}]
                },
            },
            limit=10
        )
        hits_ids = [hit["_id"] for hit in response['hits']]
        # The relevant document should not exist due to score modifiers boosting irrelevant docs
        self.assertNotIn('relevant_0', hits_ids)

    def test_second_phase_lexical_score_modifiers_works(self):
        response = self.client.index(self.unstructured_index_name).search(
            "relevant documents",
            search_method="HYBRID",
            hybrid_parameters={
                "rankingMethod": "lexical",
                "retrievalMethod": "lexical",
                "secondPhaseModifier": True,
                "scoreModifiersLexical": {
                    "add_to_score": [{"field_name": "score_modifier_value", "weight": 1.0}]
                },
            },
            limit=10
        )
        hits_ids = [hit["_id"] for hit in response['hits']]
        # The relevant document should exist now due to second phase lexical score modifiers
        self.assertEqual('relevant_0', hits_ids[9])

    def test_weakand_and_second_phase_lexical_score_modifiers_work(self):
        response = self.client.index(self.unstructured_index_name).search(
            "relevant documents",
            search_method="HYBRID",
            hybrid_parameters={
                "rankingMethod": "lexical",
                "retrievalMethod": "lexical",
                "secondPhaseModifier": True,
                "rerankDepthLexical": 10,
                "rerankCount": 1,
                "weakAndParameters": {
                    "stopwordLimit": 0.99,
                    "adjustTarget": 0.1,
                    "allowDropAll": True,
                    "filterThreshold": 0.0
                },
                "scoreModifiersLexical": {
                    "add_to_score": [{"field_name": "score_modifier_value", "weight": 1.0}]
                },
            },
            limit=10
        )
        hits_ids = [hit["_id"] for hit in response['hits']]
        # The relevant document should exist now due to second phase lexical score modifiers
        self.assertEqual('relevant_0', hits_ids[0])

    def test_second_phase_lexical_score_modifiers_works_in_disjunction(self):
        response = self.client.index(self.unstructured_index_name).search(
            "relevant documents",
            search_method="HYBRID",
            hybrid_parameters={
                "rankingMethod": "rrf",
                "retrievalMethod": "disjunction",
                "alpha": 0.01,
                "secondPhaseModifier": True,
                "scoreModifiersLexical": {
                    "add_to_score": [{"field_name": "score_modifier_value", "weight": 1.0}]
                },
            },
            limit=10
        )
        hits_ids = [hit["_id"] for hit in response['hits']]
        # The relevant document should exist now due to second phase lexical score modifiers
        self.assertEqual('relevant_0', hits_ids[9])

    def test_unsupported_retrieval_ranking_method_combination_raises_error(self):
        test_case = [
            ("tensor", "lexical", "Tensor - Lexical"),
            ("tensor", "tensor", "Tensor - Tensor"),
            ("lexical", "tensor", "Lexical - Tensor"),
        ]
        for retrieval_method, ranking_method, msg in test_case:
            with self.subTest(msg):
                with self.assertRaises(MarqoWebError) as context:
                    self.client.index(self.unstructured_index_name).search(
                        "relevant documents",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "rankingMethod": ranking_method,
                            "retrievalMethod": retrieval_method,
                            "secondPhaseModifier": True,
                            "scoreModifiersLexical": {
                                "add_to_score": [{"field_name": "score_modifier_value", "weight": 1.0}]
                            },
                        },
                        limit=10
                    )
                self.assertIn(
                    "'secondPhaseModifier' can only be set to True when 'retrievalMethod' is 'disjunction'",
                    str(context.exception)
                )
