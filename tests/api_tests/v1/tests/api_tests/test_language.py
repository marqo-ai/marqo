import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestLanguage(MarqoTestCase):
    """Test language functionality in API endpoints."""

    multilingual_index_name = "multilingual_index_" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.multilingual_index_name,
                "type": "unstructured",
                "model": "hf/e5-small-v2",
            }
        ])

        cls.indexes_to_delete = [cls.multilingual_index_name]

    def populate_index(self):
        # Add multilingual documents
        docs = [
            {
                "_id": "1",
                "title1": "Vestido Mole Perfeito",  # Portuguese
                "title2": "Collections de livres",  # French
                "title3": "White dog",  # English,
                "size": "M"
            },
            {
                "_id": "2",
                "title1": "Vestido Mole Confortável",
                "title2": "collections art francais",
                "title3": "black cat",
                "size": "M"
            },
            {
                "_id": "3",
                "title1": "Vestido Leve e Elegante",
                "title2": "mes collections Preferees",
                "title3": "blue sky",
                "size": "S"
            }
        ]

        res = self.client.index(self.multilingual_index_name).add_documents(
            docs,
            tensor_fields=["title3"],
            mappings={
                "title1": {"type": "text_field", "language": "pt"},
                "title2": {"type": "text_field", "language": "fr"},
                "title3": {"type": "text_field", "language": "en"}
            }
        )

        self.assertFalse(res['errors'], "Should not have errors when adding documents")

    def test_lang_search_methods(self):
        """Test Portuguese and French search across different methods (lexical, hybrid lexical, hybrid RRF)."""
        self.populate_index()

        # Define language/query combinations
        language_configs = [
            ("pt", "mole", "title1"),  # Portuguese
            ("fr", "collections", "title2"),  # French
        ]

        # Define search method configurations
        search_configs = [
            ("LEXICAL", {}, "lexical search"),
            ("HYBRID", {"retrievalMethod": "lexical", "rankingMethod": "lexical"}, "hybrid lexical/lexical"),
            ("HYBRID", {"alpha": 0}, "hybrid RRF with alpha=0")
        ]

        for language, query, expected_field in language_configs:
            for search_method, hybrid_params, description in search_configs:
                with self.subTest(f"{language} {description} for '{query}'"):
                    res = self.client.index(self.multilingual_index_name).search(
                        q=query,
                        search_method=search_method,
                        language=language,
                        hybrid_parameters=hybrid_params if hybrid_params else None
                    )
                    self.assertGreater(len(res["hits"]), 0,
                                       f"Should find {language} matches for '{query}' in {description}")

                    # Verify all returned documents contain the query in the expected field
                    for hit in res["hits"]:
                        self.assertIn(query, hit[expected_field].lower())

    def test_english_search_no_matches(self):
        """Test English searches for non-English words return no matches."""
        self.populate_index()

        queries = ["mole", "collections"]  # Portuguese and French words

        search_configs = [
            ("LEXICAL", {}),
            ("HYBRID", {"retrievalMethod": "lexical", "rankingMethod": "lexical"}),
            ("HYBRID", {"alpha": 0})  # rrf with alpha=0 (purely lexical results)
        ]

        for query in queries:
            for search_method, hybrid_params in search_configs:
                with self.subTest(f"English {search_method} search for '{query}'"):
                    res = self.client.index(self.multilingual_index_name).search(
                        q=query,
                        search_method=search_method,
                        language="en",
                        hybrid_parameters=hybrid_params if hybrid_params else None
                    )

                    self.assertEqual(len(res["hits"]), 0,
                                     f"Should find no English matches for '{query}' in {search_method}")

    def test_hybrid_rrf(self):
        """Test hybrid RRF search finds matches for 'canine and feline' in tensor field."""
        self.populate_index()
        res = self.client.index(self.multilingual_index_name).search(
            q="canine and feline",
            search_method="HYBRID"
        )
        self.assertGreater(len(res["hits"]), 0, "Should find matches for 'dress' in hybrid default")

        # Verify exact order of hits
        hit_ids = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(['2', '1', '3'], hit_ids)

    def test_field_language_override(self):
        """Test that field language is set in the schema, as others test could pass with good automatic detection."""
        # mole stems differently in Portuguese and English
        docs = [
            {
                "_id": "1",
                "title_pt": "mole",
            },
            {
                "_id": "2",
                "title_en": "mole",
            },
        ]

        add_res = self.client.index(self.multilingual_index_name).add_documents(
            docs,
            tensor_fields=[],
            mappings={
                "title_pt": {"type": "text_field", "language": "pt"},
                "title_en": {"type": "text_field", "language": "en"},
            }
        )

        self.assertFalse(add_res['errors'], "Should not have errors when adding documents")

        res_pt = self.client.index(self.multilingual_index_name).search(
            q="mole",
            search_method="LEXICAL",
            language="pt"
        )

        res_en = self.client.index(self.multilingual_index_name).search(
            q="mole",
            search_method="LEXICAL",
            language="en"
        )

        hits_pt = [hit["_id"] for hit in res_pt["hits"]]
        hits_en = [hit["_id"] for hit in res_en["hits"]]

        self.assertEqual(["1"], hits_pt, "Should find only the Portuguese doc")
        self.assertEqual(["2"], hits_en, "Should find only the English doc")

    def test_facets_and_relevance_cutoff(self):
        self.populate_index()

        cases = [
            ("pt", True),
            ("en", False)
        ]

        for language, matches in cases:
            with self.subTest(f"Testing facets and relevance cutoff for language: {language}"):
                res = self.client.index(self.multilingual_index_name).search(
                    q="mole",
                    search_method="HYBRID",
                    language=language,
                    hybrid_parameters={
                        'searchableAttributesLexical': ['title1'],
                        'retrievalMethod': 'lexical',
                        'rankingMethod': 'lexical',
                    }, # Use lexical retrieval so that facets don't get tensor hits
                    facets={
                        "fields": {
                            "size": {"type": "string"}
                        }
                    }, relevance_cutoff={"method": "mean_std_dev", "parameters": {"stdDevFactor": 1.2}},
                )

                if matches:
                    self.assertGreater(len(res["hits"]), 0, "Should find matches for 'mole'")
                    self.assertGreater(len(res['facets']['size']), 0, "Should have facets for 'size'")
                    self.assertGreater(res["_relevantCandidates"], 0,
                                       "Should have relevant candidates count greater than 0")
                else:
                    self.assertEqual(len(res["hits"]), 0, "Should find no matches for 'mole' in English")
                    self.assertEqual(len(res['facets']), 0, "Should have no facets for 'size' in English")
                    self.assertEqual(res["_relevantCandidates"], 0,
                                     "Should have no relevant candidates count in English")

    def test_tensor_search_with_language_error(self):
        """Test that specifying language for tensor search raises an error."""
        self.populate_index()
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.multilingual_index_name).search(
                q="dress",
                search_method="TENSOR",
                language="en"
            )
        self.assertEqual(e.exception.status_code, 422)
        self.assertIn("language", str(e.exception).lower())


if __name__ == "__main__":
    import unittest

    unittest.main()
