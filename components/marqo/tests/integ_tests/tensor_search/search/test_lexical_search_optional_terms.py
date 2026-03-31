import os
import unittest
from unittest import mock

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model, UnstructuredMarqoIndex
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLexicalSearchOptionalTerms(MarqoTestCase):
    """Integration test verifying that unquoted (optional) terms in lexical queries
    do not filter the recall set.

    Regression test for a bug where or_phrases were ANDed with and_phrases in YQL,
    making optional terms mandatory. Punctuation-only tokens like '-' are stripped
    by Vespa's tokenizer during indexing, so requiring them to match returned zero
    results.

    The fix uses Vespa's rank() operator so that required (quoted) terms define the
    recall set and optional (unquoted) terms only contribute to scoring.
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
        super().setUp()
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
