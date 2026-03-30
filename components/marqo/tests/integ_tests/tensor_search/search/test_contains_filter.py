import unittest

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

        legacy_unstructured_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            marqo_version='2.12.0'
        )

        cls.indexes = cls.create_indexes([semi_structured_index, structured_index, legacy_unstructured_index])
        cls.semi_structured_index = cls.indexes[0]
        cls.structured_index = cls.indexes[1]
        cls.legacy_unstructured_index = cls.indexes[2]

    def setUp(self) -> None:
        super().setUp()

        # Add test documents
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index.name,
                docs=[
                    {"_id": "1", "title": "Hello World", "description": "A simple greeting program", "score": 3},
                    {"_id": "2", "title": "World Cup", "description": "Football tournament"},
                    {"_id": "3", "title": "Python Programming", "description": "Learn python basics"},
                    {"_id": "4", "title": "hello again", "description": "Another farewell", "score": 15},
                    {"_id": "5", "title": "Machine Learning", "description": "AI and ML concepts", "score": 5},
                    {"_id": "6", "title": "Real-time Systems", "description": "Low-latency computing"},
                    {"_id": "7", "title": "vintage t-shirt collection", "description": "fashion items"},
                    {"_id": "8", "title": "key:value pairs", "description": 'She said "hello" loudly'},
                ],
                tensor_fields=["title", "description"]
            )
        )

    def _search(self, filter_str, text="", search_method=SearchMethod.TENSOR):
        return tensor_search.search(
            index_name=self.semi_structured_index.name,
            config=self.config,
            text=text,
            filter=filter_str,
            search_method=search_method,
            result_count=10,
        )

    def _get_ids(self, res):
        return sorted([hit["_id"] for hit in res["hits"]])

    def test_no_filter_returns_all_docs(self):
        """Searching without a filter should return all documents."""
        res = tensor_search.search(
            index_name=self.semi_structured_index.name,
            config=self.config,
            text="",
            search_method=SearchMethod.TENSOR,
            result_count=10,
        )
        self.assertEqual(self._get_ids(res), ["1", "2", "3", "4", "5", "6", "7", "8"])

    def test_basic_contains_match(self):
        """title CONTAINS hello should match docs 1 and 4 (case-insensitive after tokenization)."""
        res = self._search("title CONTAINS hello")
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_contains_does_not_match_substring(self):
        """CONTAINS does token-level matching, not substring matching."""
        res = self._search("title CONTAINS ello")
        self.assertEqual(len(res["hits"]), 0)

    def test_case_insensitivity(self):
        """title CONTAINS WORLD should match docs 1 and 2."""
        res = self._search("title CONTAINS WORLD")
        self.assertEqual(self._get_ids(res), ["1", "2"])

    def test_no_match(self):
        """title CONTAINS nonexistent should return 0 hits."""
        res = self._search("title CONTAINS nonexistent")
        self.assertEqual(len(res["hits"]), 0)

    def test_not_contains(self):
        """NOT (title CONTAINS hello) should match all docs except 1 and 4."""
        with self.subTest("with parentheses"):
            res = self._search("NOT (title CONTAINS hello)")
            self.assertEqual(self._get_ids(res), ["2", "3", "5", "6", "7", "8"])

        with self.subTest("without parentheses"):
            res = self._search("NOT title CONTAINS hello")
            self.assertEqual(self._get_ids(res), ["2", "3", "5", "6", "7", "8"])

    def test_contains_with_and(self):
        """title CONTAINS hello AND description CONTAINS simple should match only doc 1.

        Doc 1 has title "Hello World" and description "A simple greeting program".
        Doc 4 has title "hello again" but description "Another farewell" (no "simple").
        """
        res = self._search("title CONTAINS hello AND description CONTAINS simple")
        self.assertEqual(self._get_ids(res), ["1"])

    def test_contains_with_or(self):
        """title CONTAINS python OR description CONTAINS greeting should match docs 1, 3."""
        res = self._search("title CONTAINS python OR description CONTAINS greeting")
        self.assertEqual(self._get_ids(res), ["1", "3"])

    def test_contains_combined_with_equality(self):
        """title CONTAINS world AND description:(Football tournament) should match doc 2."""
        res = self._search("title CONTAINS world AND description:(Football tournament)")
        self.assertEqual(self._get_ids(res), ["2"])

    def test_contains_combined_with_range(self):
        """title CONTAINS hello AND score:[1 TO 10] should match only doc 1.

        Docs 1 and 4 both have "hello" in title. Doc 1 has score=3 (in range),
        doc 4 has score=15 (out of range). The AND narrows to doc 1 only.
        """
        res = self._search("title CONTAINS hello AND score:[1 TO 10]")
        self.assertEqual(self._get_ids(res), ["1"])

    def test_contains_across_fields_no_overlap(self):
        """title CONTAINS hello AND description CONTAINS ai should return 0 hits."""
        res = self._search("title CONTAINS hello AND description CONTAINS ai")
        self.assertEqual(len(res["hits"]), 0)

    def test_contains_with_tensor_search(self):
        """CONTAINS filter combined with tensor search."""
        res = self._search("title CONTAINS hello", text="greeting", search_method=SearchMethod.TENSOR)
        self.assertEqual(self._get_ids(res), ["1", "4"])

    def test_contains_with_lexical_search(self):
        """CONTAINS filter combined with lexical search."""
        res = self._search("title CONTAINS hello", text="greeting", search_method=SearchMethod.LEXICAL)
        ids = self._get_ids(res)
        # Should only return docs with "hello" in title
        for doc_id in ids:
            self.assertIn(doc_id, ["1", "4"])

    def test_not_contains_with_and(self):
        """NOT (title CONTAINS hello) AND title CONTAINS world should match doc 2 only."""
        res = self._search("NOT (title CONTAINS hello) AND title CONTAINS world")
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
        self.assertEqual(ids, ["1", "4"])

    def test_contains_with_special_characters_in_filter_value(self):
        """Test CONTAINS filter where the filter value itself contains special characters."""
        with self.subTest("hyphen in filter value: title CONTAINS t-shirt"):
            res = self._search("title CONTAINS t-shirt")
            ids = self._get_ids(res)
            self.assertIn("7", ids)

        with self.subTest("hyphen in filter value: title CONTAINS real-time"):
            res = self._search("title CONTAINS real-time")
            ids = self._get_ids(res)
            self.assertIn("6", ids)

        with self.subTest("grouped value with special chars: title CONTAINS (t-shirt collection)"):
            res = self._search("title CONTAINS (t-shirt collection)")
            ids = self._get_ids(res)
            self.assertIn("7", ids)

    def test_contains_with_colon_in_value(self):
        """Test CONTAINS filter where the indexed text contains a colon."""
        # Doc 8 has title "key:value pairs". Vespa tokenizes "key:value" into "key" and "value".
        with self.subTest("token match: title CONTAINS key"):
            res = self._search("title CONTAINS key")
            ids = self._get_ids(res)
            self.assertIn("8", ids)

        with self.subTest("grouped value with colon: title CONTAINS (key:)"):
            res = self._search("title CONTAINS (key:)")
            ids = self._get_ids(res)
            self.assertIn("8", ids)

    def test_contains_with_quote_in_value(self):
        """Test CONTAINS filter where the indexed text contains quotes."""
        # Doc 8 has description 'She said "hello" loudly'. The token "hello" should match.
        with self.subTest("token match: description CONTAINS hello"):
            res = self._search("description CONTAINS hello")
            ids = self._get_ids(res)
            self.assertIn("8", ids)

        with self.subTest('grouped value with quotes: description CONTAINS (\\"hello\\")'):
            res = self._search('description CONTAINS (\\"hello\\")')
            ids = self._get_ids(res)
            self.assertIn("8", ids)

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

    def test_legacy_unstructured_index_contains_raises_error(self):
        """Using CONTAINS on a legacy unstructured index should raise InvalidArgumentError."""
        # Add a doc to the legacy index first
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.legacy_unstructured_index.name,
                docs=[{"_id": "1", "title": "Hello World"}],
                tensor_fields=["title"]
            )
        )
        with self.assertRaises(InvalidArgumentError) as ctx:
            tensor_search.search(
                index_name=self.legacy_unstructured_index.name,
                config=self.config,
                text="test",
                filter="title CONTAINS hello",
                search_method=SearchMethod.TENSOR,
            )
        self.assertIn("CONTAINS", str(ctx.exception))
