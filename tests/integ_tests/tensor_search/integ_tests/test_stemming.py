import unittest
import uuid

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search import tensor_search
from marqo.core.models.add_docs_params import AddDocsParams
from tests.integ_tests.marqo_test import MarqoTestCase


class TestStemmingIntegration(MarqoTestCase):
    """
    Integration tests for text field stemming functionality.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create semi-structured index for stemming tests
        cls.stemming_index = cls.unstructured_marqo_index_request(
            name="stemming-integ-" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/e5-small-v2")
        )
        cls.indexes = cls.create_indexes([cls.stemming_index])

    def populate_index(self):
        """Populate index with stemming test documents."""
        docs = [
            {
                "_id": "1",
                "title_stem1": "nacionalmente",  # none: nacionalmente
            },
            {
                "_id": "2", 
                "title_stem2": "nacionalmente",  # best: nacionalment
            },
            {
                "_id": "3",
                "title_stem3": "nacionalmente",  # shortest: nacionalment
            },
            {
                "_id": "4",
                "title_stem4": "nacionalmente",  # multiple: nacionalmente, nacionalment
            },
        ]

        mappings = {
            "title_stem1": {"type": "text_field", "language": "de", "stemming": "none"},
            "title_stem2": {"type": "text_field", "language": "de", "stemming": "best"},
            "title_stem3": {"type": "text_field", "language": "de", "stemming": "shortest"},
            "title_stem4": {"type": "text_field", "language": "de", "stemming": "multiple"},
        }

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=["title_stem1"]
        )

        res = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params
        )

        self.assertFalse(res.errors, "Should not have errors when adding documents")

    def test_stemming_search(self):
        """
        Test docs with different stemming configs return expected search results.
        """
        cases = [
            (
                "nacionalmente", ["title_stem1"], ["1"], "Full word matches no stemming"
            ),
            (
                "nacionalmente", ["title_stem2"], ["2"], "Full word matches best stemming"
            ),
            (
                "nacionalmente", ["title_stem3"], ["3"], "Full word matches shortest stemming"
            ),
            (
                "nacionalmente", ["title_stem4"], ["4"], "Full word matches multiple stemming"
            ),
            (
                "nacionalmente", ["title_stem1", "title_stem2"], ["1", "2"], "Full word matches with none and best fields"
            ),
            (
                "nacionalment", ["title_stem1"], [], "Stemmed word does not match none stemming"
            ),
            (
                "nacionalment", ["title_stem2"], ["2"], "Stemmed word matches best stemming"
            ),
            (
                "nacionalment", ["title_stem1", "title_stem2"], ["2"], "Stemmed word matches best stemming but not none"
            ),
            (
                "nacionalment", ["title_stem3"], ["3"], "Stemmed word matches shortest stemming"
            ),
            (
                "nacionalment", ["title_stem4"], ["4"], "Stemmed word matches multiple stemming"
            ),
        ]

        self.populate_index()

        for query, fields, expected_ids, description in cases:
            # Test LEXICAL search
            with self.subTest(f"LEXICAL search for '{query}' in {fields}: {description}"):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.stemming_index.name,
                    text=query,
                    search_method="LEXICAL",
                    searchable_attributes=fields,
                    result_count=10,
                    offset=0,
                    language="de"
                )

                actual_ids = set(hit["_id"] for hit in res["hits"] if hit["_id"] in expected_ids)
                self.assertEqual(set(expected_ids), actual_ids, f"Failed for query '{query}' in fields {fields}")

            # Test HYBRID search with lexical/lexical
            with self.subTest(f"HYBRID lexical/lexical search for '{query}' in {fields}: {description}"):
                hybrid_params = HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical, 
                    rankingMethod=RankingMethod.Lexical,
                    searchableAttributesLexical=fields
                )
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.stemming_index.name,
                    text=query,
                    search_method="HYBRID",
                    result_count=10,
                    offset=0,
                    language="de",
                    hybrid_parameters=hybrid_params
                )

                actual_ids = set(hit["_id"] for hit in res["hits"] if hit["_id"] in expected_ids)
                self.assertEqual(set(expected_ids), actual_ids, f"Failed for query '{query}' in fields {fields}")

            # Test HYBRID search with RRF alpha=0
            with self.subTest(f"HYBRID RRF alpha=0 search for '{query}' in {fields}: {description}"):
                hybrid_params = HybridParameters(
                    alpha=0,
                    searchableAttributesLexical=fields
                )
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.stemming_index.name,
                    text=query,
                    search_method="HYBRID",
                    result_count=10,
                    offset=0,
                    language="de",
                    hybrid_parameters=hybrid_params
                )

                actual_ids = set(hit["_id"] for hit in res["hits"] if hit["_id"] in expected_ids)
                self.assertEqual(set(expected_ids), actual_ids, f"Failed for query '{query}' in fields {fields}")

    def test_stemming_all_fields_search(self):
        """
        Test searching all fields (no searchable attributes specified) returns some results.
        """
        self.populate_index()

        res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="nacionalmente",
            search_method="LEXICAL",
            result_count=10,
            offset=0,
            language="de"
        )

        self.assertGreater(len(res["hits"]), 0, "Should find matches for 'nacionalmente' in all fields")

    def test_stemming_invalid_value_error(self):
        """Test that invalid stemming values produce proper errors."""
        docs = [{"_id": "invalid_test", "field": "test content"}]
        mappings = {"field": {"type": "text_field", "language": "en", "stemming": "invalid_algorithm"}}

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=[]
        )

        # Should raise an error due to invalid stemming value
        with self.assertRaises(Exception) as cm:
            self.add_documents(
                config=self.config,
                add_docs_params=add_docs_params
            )

        error_message = str(cm.exception)
        self.assertIn("stemming", error_message.lower())

    def test_stemming_field_change_error(self):
        """Test that changing stemming configuration produces error."""
        # First add document with one stemming config
        docs1 = [{"_id": "change_test1", "title": "First document"}]
        mappings1 = {"title": {"type": "text_field", "language": "en", "stemming": "best"}}

        add_docs_params1 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs1,
            mappings=mappings1,
            tensor_fields=[]
        )

        response1 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params1
        )
        self.assertFalse(response1.errors)

        # Try to add document with different stemming config for same field
        docs2 = [{"_id": "change_test2", "title": "Second document"}]
        mappings2 = {"title": {"type": "text_field", "language": "en", "stemming": "shortest"}}

        add_docs_params2 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs2,
            mappings=mappings2,
            tensor_fields=[]
        )

        response2 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params2
        )

        # Should have errors
        self.assertTrue(response2.errors)
        error_message = response2.items[0].message
        self.assertIn("different stemming configuration", error_message)

    def test_stemming_no_language(self):
        """Test that no stemming occurs when stemming is set to 'none' without language in field mapping."""
        docs = [
            {
                "_id": "no_lang_1",
                "content": "running quickly"
            },
            {
                "_id": "no_lang_2", 
                "content": "runs fast"
            }
        ]

        # Add documents with no stemming and no language in mapping
        mappings = {
            "content": {"type": "text_field", "stemming": "none"}
        }

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=[]
        )

        res = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params
        )
        
        self.assertFalse(res.errors, "Should not have errors when adding documents without language")

        # Search for exact matches should work
        running_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="running",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="en"
        )
        
        runs_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="runs",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="en"
        )

        # Verify exact matches work
        running_ids = {hit["_id"] for hit in running_res["hits"]}
        runs_ids = {hit["_id"] for hit in runs_res["hits"]}
        
        self.assertEqual({"no_lang_1"}, running_ids, "Should find document with 'running'")
        self.assertEqual({"no_lang_2"}, runs_ids, "Should find document with 'runs'")


if __name__ == '__main__':
    unittest.main()
