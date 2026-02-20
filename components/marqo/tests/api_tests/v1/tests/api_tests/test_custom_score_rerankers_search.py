"""
API test: custom score rerankers (marqo__score_*) with hybrid RRF on unstructured index.
Proves base RRF order is doc1..doc5; with add_to_score on a ranking field, order becomes doc5..doc1.
"""
import uuid

from tests.marqo_test import MarqoTestCase

# From integ test_custom_score_reranking_feature: deterministic docs for query "tuxedo"
TUXEDO_QUERY = "tuxedo"
DOCS = [
    {
        "_id": "doc1", "lex_retrieval_field": "tuxedo tuxedo tuxedo", 
        "tensor_retrieval_field": "tuxedo",
        "lex_ranking_field": "tuxedo", 
        "tensor_ranking_field": "unrelated"
    },
    {
        "_id": "doc2", "lex_retrieval_field": "no match", 
        "tensor_retrieval_field": "suit",
        "lex_ranking_field": "tuxedo tuxedo", 
        "tensor_ranking_field": "rainbow tie"
    },
    {
        "_id": "doc3", "lex_retrieval_field": "tuxedo tuxedo", 
        "tensor_retrieval_field": "unrelated",
        "lex_ranking_field": "tuxedo tuxedo tuxedo", 
        "tensor_ranking_field": "shorts"
    },
    {"_id": "doc4", "lex_retrieval_field": "no match", "tensor_retrieval_field": "shorts",
     "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo", "tensor_ranking_field": "suit"},
    {"_id": "doc5", "lex_retrieval_field": "tuxedo", "tensor_retrieval_field": "backpack",
     "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo", "tensor_ranking_field": "tuxedo"},
]
TENSOR_FIELDS = ["tensor_retrieval_field", "tensor_ranking_field"]
BASE_RRF_ORDER = ["doc1", "doc2", "doc3", "doc4", "doc5"]
REVERSED_ORDER = ["doc5", "doc4", "doc3", "doc2", "doc1"]


class TestCustomScoreRerankersSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.index_name = "custom_score_rerank_" + str(uuid.uuid4()).replace("-", "")
        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-16-SigLIP/webli",
            }
        ])
        cls.indexes_to_delete = [cls.index_name]

    def test_custom_score_rerank_reverses_order(self):
        """Base RRF order is doc1..doc5; add_to_score with bm25 lex_ranking_field reverses to doc5..doc1."""
        self.client.index(self.index_name).add_documents(
            DOCS,
            tensor_fields=TENSOR_FIELDS,
        )
        self.assertEqual(5, self.client.index(self.index_name).get_stats()["numberOfDocuments"])

        base = self.client.index(self.index_name).search(
            q=TUXEDO_QUERY,
            search_method="HYBRID",
            hybrid_parameters={
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5001,
                "rrfK": 60,
                "searchableAttributesTensor": ["tensor_retrieval_field"],
                "searchableAttributesLexical": ["lex_retrieval_field"],
            },
            limit=5,
        )
        self.assertEqual([h["_id"] for h in base["hits"]], BASE_RRF_ORDER)

        with_modifier = self.client.index(self.index_name).search(
            q=TUXEDO_QUERY,
            search_method="HYBRID",
            hybrid_parameters={
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5001,
                "rrfK": 60,
                "searchableAttributesTensor": ["tensor_retrieval_field"],
                "searchableAttributesLexical": ["lex_retrieval_field"],
            },
            score_modifiers={
                "add_to_score": [
                    {"field_name": "marqo__score_bm25_field_lex_ranking_field", "weight": 1.0}
                ],
            },
            limit=5,
        )
        self.assertEqual([h["_id"] for h in with_modifier["hits"]], REVERSED_ORDER)
