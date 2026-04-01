import os
import unittest
from unittest import mock

from marqo.core.constants import MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import Model, UnstructuredMarqoIndex
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import ScoreModifierLists
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLexicalSearchOptionalTerms(MarqoTestCase):
    """Verifies that unquoted (optional) terms in lexical queries do not filter the recall set.
    Only quoted (required) terms define recall; optional terms contribute to scoring only.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        semi_structured_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.create_indexes([semi_structured_index])
        cls.index_name = semi_structured_index.name

        docs = [
            {"_id": "doc1", "title": "Quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "Quick silver racing through the forest"},
            {"_id": "doc3", "title": "Aluminum portable water bottle large capacity"},
            {"_id": "doc4", "title": "Vintage wooden bookshelf with drawers"},
            {"_id": "doc5", "title": "Outdoor garden table - heavy duty rust proof"},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                index_name=cls.index_name,
                docs=docs,
                tensor_fields=["title"],
            )
        )

    def setUp(self):
        self.device_patcher = mock.patch.dict(os.environ, {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu",
        })
        self.device_patcher.start()

    def tearDown(self):
        super().tearDown()
        self.device_patcher.stop()

    def _search(self, query: str):
        return tensor_search.search(
            text=query, config=self.config, index_name=self.index_name,
            search_method=SearchMethod.LEXICAL
        )

    def _result_ids(self, res) -> set:
        return {hit['_id'] for hit in res['hits']}

    def test_required_term_with_optional_punctuation(self):
        """Quoted required term with unquoted punctuation '-' should still return results."""
        res = self._search('"Quick" -')
        self.assertEqual({'doc1', 'doc2'}, self._result_ids(res))

    def test_multiple_required_terms_with_optional_punctuation(self):
        """Multiple quoted required terms with unquoted punctuation should still return results."""
        res = self._search('"Aluminum" "portable" -')
        self.assertEqual({'doc3'}, self._result_ids(res))

    def test_required_term_with_optional_word(self):
        """Optional unquoted word should not filter — all docs matching the required term are returned."""
        res = self._search('"Quick" fox')
        self.assertEqual({'doc1', 'doc2'}, self._result_ids(res))

    def test_optional_term_from_different_doc_does_not_expand_recall(self):
        """Optional term matching a different doc should not pull that doc into results."""
        res = self._search('"Aluminum" forest')
        self.assertEqual({'doc3'}, self._result_ids(res))

    def test_all_required_terms_and_together(self):
        """Multiple quoted terms should AND together — only docs matching all are returned."""
        res = self._search('"Quick" "fox"')
        self.assertEqual({'doc1'}, self._result_ids(res))

    def test_all_optional_terms_use_weak_and(self):
        """All unquoted terms should use weakAnd — docs matching any terms are returned."""
        res = self._search('Quick fox')
        result_ids = self._result_ids(res)
        self.assertIn('doc1', result_ids)

    def test_required_term_not_in_any_doc(self):
        """A required term not present in any document should return no results."""
        res = self._search('"Xylophone99"')
        self.assertEqual(set(), self._result_ids(res))

    def test_punctuation_in_doc_stripped_on_ingestion(self):
        """A standalone '-' in document text like 'table - heavy' is stripped by Vespa's
        tokenizer during ingestion, so it cannot be matched as a search term."""
        res = self._search('"garden" "table"')
        self.assertEqual({'doc5'}, self._result_ids(res))

    def test_doc_with_punctuation_found_despite_optional_punctuation_in_query(self):
        """A doc containing a standalone '-' should still be found when the query
        also has an optional '-' that gets stripped."""
        res = self._search('"garden" -')
        self.assertEqual({'doc5'}, self._result_ids(res))


class TestLexicalOptionalTermsWithHybridAndScoreModifiers(MarqoTestCase):
    """Verifies that optional/required lexical terms work correctly with custom score
    rerankers and hybrid search with opposite retrieval/ranking methods.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.semi_structured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )
        cls.create_indexes([cls.semi_structured_index_request])
        cls.index_name = cls.semi_structured_index_request.name

        docs = [
            {"_id": "doc1", "title": "Quick brown fox jumps over the lazy dog"},
            {"_id": "doc2", "title": "Quick silver racing through the forest"},
            {"_id": "doc3", "title": "Aluminum portable water bottle large capacity"},
            {"_id": "doc4", "title": "Vintage wooden bookshelf with drawers"},
            {"_id": "doc5", "title": "Outdoor garden table heavy duty rust proof"},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                index_name=cls.index_name,
                docs=docs,
                tensor_fields=["title"],
            )
        )

    def setUp(self):
        self.device_patcher = mock.patch.dict(os.environ, {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu",
        })
        self.device_patcher.start()

    def tearDown(self):
        super().tearDown()
        self.device_patcher.stop()

    def _result_ids(self, res) -> set:
        return {hit['_id'] for hit in res['hits']}

    def test_custom_score_rerank_bm25_with_required_and_optional_terms(self):
        """Custom score reranker (marqo__score_bm25_field_*) with required + optional terms.
        Creates nested rank(): rank(rank(and_terms, or_terms), extra_bm25_term)."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"Quick" fox',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title", "weight": 1.0}
                ]
            ),
            result_count=10
        )
        result_ids = self._result_ids(res)
        self.assertIn('doc1', result_ids)
        self.assertIn('doc2', result_ids)

    def test_lexical_retrieval_tensor_ranking_with_required_and_optional_terms(self):
        """Lexical retrieval + Tensor ranking with required + optional terms.
        Creates nested rank(): rank(rank(and_terms, or_terms), nearestNeighbor(...))."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"Quick" fox',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Tensor,
            ),
            result_count=10
        )
        result_ids = self._result_ids(res)
        self.assertIn('doc1', result_ids)
        self.assertIn('doc2', result_ids)
        self.assertNotIn('doc3', result_ids)

    def test_tensor_retrieval_lexical_ranking_with_required_and_optional_terms(self):
        """Tensor retrieval + Lexical ranking with required + optional terms.
        Creates nested rank(): rank(tensor_term, rank(and_terms, or_terms))."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"Quick" fox',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Tensor,
                rankingMethod=RankingMethod.Lexical,
            ),
            result_count=10
        )
        result_ids = self._result_ids(res)
        self.assertGreater(len(result_ids), 0)
        self.assertIn('doc1', result_ids)
