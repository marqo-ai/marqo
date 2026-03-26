import os
import unittest
from unittest import mock

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestContainsFilter(MarqoTestCase):
    """Tests for the CONTAINS filter keyword on semi-structured indexes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        semi_structured_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        structured_index = cls.structured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            fields=[
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="description", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
            ],
            tensor_fields=["title", "description"]
        )

        cls.indexes = cls.create_indexes([semi_structured_index, structured_index])
        cls.semi_structured_index = cls.indexes[0]
        cls.structured_index = cls.indexes[1]

    def setUp(self) -> None:
        super().setUp()
        self.device_patcher = mock.patch.dict(os.environ, {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu",
            "MARQO_MAX_CPU_MODEL_MEMORY": "15"
        })
        self.device_patcher.start()

        # Add test documents
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index.name,
                docs=[
                    {"_id": "1", "title": "Hello World", "description": "A simple greeting program"},
                    {"_id": "2", "title": "World Cup", "description": "Football tournament"},
                    {"_id": "3", "title": "Python Programming", "description": "Learn python basics"},
                    {"_id": "4", "title": "hello again", "description": "Another greeting"},
                    {"_id": "5", "title": "Machine Learning", "description": "AI and ML concepts"},
                    {"_id": "6", "title": "Real-time Systems", "description": "Low-latency computing"},
                ],
                tensor_fields=["title", "description"]
            )
        )

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def _search(self, filter_str, text="", search_method=SearchMethod.TENSOR):
        return tensor_search.search(
            index_name=self.semi_structured_index.name,
            config=self.config,
            text=text,
            filter=filter_str,
            search_method=search_method,
        )

    def _get_ids(self, res):
        return sorted([hit["_id"] for hit in res["hits"]])

    def test_basic_contains_match(self):
        """title CONTAINS hello should match docs 1 and 4 (case-insensitive after tokenization)."""
        res = self._search("title CONTAINS hello")
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_case_insensitivity(self):
        """title CONTAINS WORLD should match docs 1 and 2."""
        res = self._search("title CONTAINS WORLD")
        self.assertEqual(self._get_ids(res), ["1", "2"])

    def test_no_match(self):
        """title CONTAINS nonexistent should return 0 hits."""
        res = self._search("title CONTAINS nonexistent")
        self.assertEqual(len(res["hits"]), 0)

    def test_not_contains(self):
        """NOT (title CONTAINS hello) should match docs 2, 3, 5."""
        res = self._search("NOT (title CONTAINS hello)", text="world programming learning")
        ids = self._get_ids(res)
        # All results should NOT have "hello" in title
        for doc_id in ids:
            self.assertNotIn(doc_id, ["1", "4"])
        # Should include at least some of docs 2, 3, 5
        self.assertTrue(len(ids) > 0)

    def test_contains_with_and(self):
        """title CONTAINS hello AND description CONTAINS greeting should match docs 1 and 4."""
        res = self._search("title CONTAINS hello AND description CONTAINS greeting")
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_contains_with_or(self):
        """title CONTAINS python OR description CONTAINS greeting should match docs 1, 3, 4."""
        res = self._search("title CONTAINS python OR description CONTAINS greeting")
        self.assertEqual(self._get_ids(res), ["1", "3", "4"])

    def test_contains_combined_with_equality(self):
        """title CONTAINS world AND description:(Football tournament) should match doc 2."""
        res = self._search("title CONTAINS world AND description:(Football tournament)")
        self.assertEqual(self._get_ids(res), ["2"])

    def test_contains_across_fields_no_overlap(self):
        """title CONTAINS hello AND description CONTAINS ai should return 0 hits."""
        res = self._search("title CONTAINS hello AND description CONTAINS ai")
        self.assertEqual(len(res["hits"]), 0)

    def test_contains_with_tensor_search(self):
        """CONTAINS filter combined with tensor search."""
        res = self._search("title CONTAINS hello", text="greeting", search_method=SearchMethod.TENSOR)
        ids = self._get_ids(res)
        # Should only return docs with "hello" in title
        for doc_id in ids:
            self.assertIn(doc_id, ["1", "4"])

    def test_contains_with_lexical_search(self):
        """CONTAINS filter combined with lexical search."""
        res = self._search("title CONTAINS hello", text="greeting", search_method=SearchMethod.LEXICAL)
        ids = self._get_ids(res)
        # Should only return docs with "hello" in title
        for doc_id in ids:
            self.assertIn(doc_id, ["1", "4"])

    def test_not_contains_with_and(self):
        """NOT (title CONTAINS hello) AND title CONTAINS world should match doc 2 only."""
        res = self._search("NOT (title CONTAINS hello) AND title CONTAINS world", text="world cup")
        self.assertEqual(self._get_ids(res), ["2"])

    def test_nonexistent_field_raises_error(self):
        """Filtering on a nonexistent field should raise an error."""
        with self.assertRaises(InvalidArgumentError):
            self._search("nonexistent_field CONTAINS value")

    def test_lowercase_contains_keyword(self):
        """The CONTAINS keyword should be case-insensitive."""
        res = self._search("title contains hello")
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_mixed_case_contains_keyword(self):
        """The CONTAINS keyword should be case-insensitive with mixed case."""
        res = self._search("title Contains hello")
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_contains_grouped_multiword_value(self):
        """description CONTAINS (simple greeting) should match doc 1 only (phrase match)."""
        res = self._search("description CONTAINS (simple greeting)")
        ids = self._get_ids(res)
        # Doc 1 has "A simple greeting program" which contains the phrase "simple greeting"
        # Doc 4 has "Another greeting" which does NOT contain "simple greeting" as a phrase
        self.assertEqual(ids, ["1"])

    def test_contains_with_hybrid_search(self):
        """CONTAINS filter combined with hybrid search."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_index.name,
            text="greeting",
            filter="title CONTAINS hello",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
            ),
        )
        ids = self._get_ids(res)
        # Should only return docs with "hello" in title (docs 1 and 4)
        self.assertEqual(ids, ["1", "4"])

    def test_contains_with_special_characters(self):
        """Vespa tokenization splits hyphenated words, so CONTAINS matches individual parts.

        Doc 6 has title "Real-time Systems". The tokenizer splits "Real-time" into tokens
        "real" and "time" (lowercased). This test documents that CONTAINS matches on each
        token independently when special characters like hyphens are present.
        """
        with self.subTest("title CONTAINS real should match doc 6"):
            res = self._search("title CONTAINS real")
            ids = self._get_ids(res)
            self.assertIn("6", ids)

        with self.subTest("title CONTAINS time should match doc 6"):
            res = self._search("title CONTAINS time")
            ids = self._get_ids(res)
            self.assertIn("6", ids)

    def test_structured_index_contains_raises_error(self):
        """Using CONTAINS on a structured index should raise InvalidArgumentError."""
        with self.assertRaises(InvalidArgumentError) as ctx:
            tensor_search.search(
                index_name=self.structured_index.name,
                config=self.config,
                text="test",
                filter="title CONTAINS hello",
                search_method=SearchMethod.TENSOR,
            )
        self.assertIn("CONTAINS", str(ctx.exception))
