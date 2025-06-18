"""
Comprehensive integration tests for language feature.
Tests language functionality with different search methods and field configurations.
Follows TestSearchSemiStructured pattern with focus on unstructured index testing.
"""
import copy
import os
import uuid
from unittest import mock

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLanguage(MarqoTestCase):
    """Integration tests for language functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create one unstructured index for all language testing (following user requirement for unstructured index only)
        cls.unstructured_index = cls.unstructured_marqo_index_request(
            name='test_language_search_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        cls.indexes = cls.create_indexes([cls.unstructured_index])
        cls.index = cls.indexes[0]

    def setUp(self) -> None:
        super().setUp()
        # Clear index before each test
        self.clear_index_by_index_name(self.index.name)
        
        # Set device to CPU for all tests
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_positive_language_search_all_languages(self):
        """Positive test cases for all 5 languages with all three search methods."""
        
        # Define test data for each language
        language_test_data = {
            "en": {
                "language_code": "en-US",
                "docs": [
                    {"_id": "en1", "title": "Running in the beautiful park", "content": "The athlete runs quickly"},
                    {"_id": "en2", "title": "Swimming at the ocean", "content": "Swimming is excellent exercise"},
                    {"_id": "en3", "title": "Reading interesting books", "content": "Books provide knowledge and entertainment"}
                ],
                "search_tests": [
                    ("running", "en1"),
                    ("swimming", "en2"), 
                    ("books", "en3")
                ]
            },
            "es": {
                "language_code": "es",
                "docs": [
                    {"_id": "es1", "title": "Corriendo en el parque hermoso", "content": "El atleta corre rápidamente"},
                    {"_id": "es2", "title": "Nadando en el océano", "content": "Nadar es excelente ejercicio"},
                    {"_id": "es3", "title": "Leyendo libros interesantes", "content": "Los libros proporcionan conocimiento"}
                ],
                "search_tests": [
                    ("corriendo", "es1"),
                    ("nadando", "es2"),
                    ("libros", "es3")
                ]
            },
            "fr": {
                "language_code": "fr",
                "docs": [
                    {"_id": "fr1", "title": "Courant dans le parc magnifique", "content": "L'athlète court rapidement"},
                    {"_id": "fr2", "title": "Nageant dans l'océan", "content": "La natation est un excellent exercice"},
                    {"_id": "fr3", "title": "Lisant des livres intéressants", "content": "Les livres fournissent des connaissances"}
                ],
                "search_tests": [
                    ("courant", "fr1"),
                    ("nageant", "fr2"),
                    ("livres", "fr3")
                ]
            },
            "de": {
                "language_code": "de",
                "docs": [
                    {"_id": "de1", "title": "Laufen im schönen Park", "content": "Der Athlet läuft schnell"},
                    {"_id": "de2", "title": "Schwimmen im Ozean", "content": "Schwimmen ist ausgezeichnete Übung"},
                    {"_id": "de3", "title": "Interessante Bücher lesen", "content": "Bücher bieten Wissen und Unterhaltung"}
                ],
                "search_tests": [
                    ("laufen", "de1"),
                    ("schwimmen", "de2"),
                    ("bücher", "de3")
                ]
            },
            "pt": {
                "language_code": "pt",
                "docs": [
                    {"_id": "pt1", "title": "Correndo no parque bonito", "content": "O atleta corre rapidamente"},
                    {"_id": "pt2", "title": "Nadando no oceano", "content": "Nadar é excelente exercício"},
                    {"_id": "pt3", "title": "Lendo livros interessantes", "content": "Livros fornecem conhecimento"}
                ],
                "search_tests": [
                    ("correndo", "pt1"),
                    ("nadando", "pt2"),
                    ("livros", "pt3")
                ]
            }
        }

        for language, test_data in language_test_data.items():
            with self.subTest(language=language):
                # Clear index for each language test
                self.clear_index_by_index_name(self.index.name)
                
                # Set up mappings for this language
                mappings = {
                    "title": {"type": "text_field", "language": test_data["language_code"]},
                    "content": {"type": "text_field", "language": test_data["language_code"]}
                }

                # Add documents for this language
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.index.name,
                        docs=test_data["docs"],
                        tensor_fields=["title"] if language in ["en", "fr", "de"] else ["content"],  # Vary tensor fields
                        mappings=mappings
                    )
                )

                # Test all three search methods
                search_methods = [
                    (SearchMethod.LEXICAL, "lexical", None),
                    (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Lexical
                    )),
                    (SearchMethod.HYBRID, "hybrid_rrf", None)
                ]

                for search_method, method_name, hybrid_params in search_methods:
                    with self.subTest(search_method=method_name):
                        for search_term, expected_id in test_data["search_tests"]:
                            with self.subTest(search_term=search_term):
                                result = tensor_search.search(
                                    config=self.config,
                                    index_name=self.index.name,
                                    text=search_term,
                                    search_method=search_method,
                                    hybrid_parameters=hybrid_params
                                )
                                
                                # Assert that we get some results
                                self.assertGreater(len(result["hits"]), 0, 
                                                 f"No results for {search_term} in {language} using {method_name}")
                                
                                # For lexical search, check if expected ID is first
                                # For hybrid searches, just check if expected ID is in results
                                hit_ids = [hit["_id"] for hit in result["hits"]]
                                if method_name == "lexical":
                                    self.assertEqual(result["hits"][0]["_id"], expected_id,
                                                   f"Expected {expected_id} as first result for {search_term} in {language}")
                                else:
                                    self.assertIn(expected_id, hit_ids,
                                                f"Expected {expected_id} in results for {search_term} in {language} using {method_name}")


    def test_index_with_three_text_fields_different_languages(self):
        """Test index with three text fields: French, Portuguese, and default (none)."""
        docs = [
            {
                "_id": "multi1",
                "french_field": "Bonjour le monde français",
                "portuguese_field": "Olá mundo português", 
                "default_field": "Hello default world"
            },
            {
                "_id": "multi2",
                "french_field": "Chat noir français",
                "portuguese_field": "Gato preto português",
                "default_field": "Black cat default"
            }
        ]
        
        mappings = {
            "french_field": {"type": "text_field", "language": "fr"},
            "portuguese_field": {"type": "text_field", "language": "pt"},
            # default_field has no mapping (default language)
        }

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=["french_field"],  # Make some lexical fields tensor fields but not all
                mappings=mappings
            )
        )

        # Define test scenarios for different language content searches
        search_scenarios = [
            ("français", "French content", ["multi1", "multi2"]),
            ("português", "Portuguese content", ["multi1", "multi2"]),
            ("default", "Default content", ["multi1", "multi2"]),
            ("world", "All fields common word", ["multi1"]),  # Only in default field
            ("cat", "Cat in multiple languages", ["multi2"])  # Should find "chat" and "gato" variants
        ]

        # Test each scenario with different search methods
        search_methods = [
            (SearchMethod.LEXICAL, "lexical", None),
            (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical
            )),
            (SearchMethod.HYBRID, "hybrid_rrf", None)
        ]

        for search_term, scenario_name, expected_ids in search_scenarios:
            with self.subTest(scenario=scenario_name):
                for search_method, method_name, hybrid_params in search_methods:
                    with self.subTest(search_method=method_name):
                        result = tensor_search.search(
                            config=self.config,
                            index_name=self.index.name,
                            text=search_term,
                            search_method=search_method,
                            hybrid_parameters=hybrid_params
                        )
                        
                        # Assert that we get some results
                        self.assertGreater(len(result["hits"]), 0, 
                                         f"No results for {search_term} using {method_name}")
                        
                        # Check that expected IDs are in results
                        hit_ids = [hit["_id"] for hit in result["hits"]]
                        for expected_id in expected_ids:
                            self.assertIn(expected_id, hit_ids, 
                                        f"Expected {expected_id} in results for {search_term} using {method_name}")

    def test_search_field_with_language_in_add_docs_without_language_in_search(self):
        """Test searching a field with language specified in add_docs, without specifying language in search."""
        docs = [
            {"_id": "lang1", "content": "Este es contenido en español"},
            {"_id": "lang2", "content": "This is content in English"}
        ]
        
        mappings = {
            "content": {"type": "text_field", "language": "es"}
        }

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=["content"],  # Make some lexical fields tensor fields but not all
                mappings=mappings
            )
        )

        # Test different search scenarios and methods
        search_scenarios = [
            ("contenido", "lang1", "Spanish content word"),
            ("español", None, "Spanish language identifier")  # None means just check for results
        ]

        search_methods = [
            (SearchMethod.LEXICAL, "lexical", None),
            (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical
            )),
            (SearchMethod.HYBRID, "hybrid_rrf", None)
        ]

        for search_term, expected_first_id, scenario_name in search_scenarios:
            with self.subTest(scenario=scenario_name):
                for search_method, method_name, hybrid_params in search_methods:
                    with self.subTest(search_method=method_name):
                        result = tensor_search.search(
                            config=self.config,
                            index_name=self.index.name,
                            text=search_term,
                            search_method=search_method,
                            hybrid_parameters=hybrid_params
                        )
                        
                        # Assert that we get some results
                        self.assertGreater(len(result["hits"]), 0, 
                                         f"No results for {search_term} using {method_name}")
                        
                        # For specific expected results, check first hit for lexical search
                        if expected_first_id and method_name == "lexical":
                            self.assertEqual(result["hits"][0]["_id"], expected_first_id,
                                           f"Expected {expected_first_id} as first result for {search_term}")
                        elif expected_first_id:  # For hybrid searches, just check if ID is in results
                            hit_ids = [hit["_id"] for hit in result["hits"]]
                            self.assertIn(expected_first_id, hit_ids,
                                        f"Expected {expected_first_id} in results for {search_term} using {method_name}")

    def test_language_change_scenarios(self):
        """Test different language change scenarios."""
        
        language_change_scenarios = [
            {
                "name": "Different language second time",
                "description": "Error when field indexed with different language first vs second time",
                "first_docs": [{"_id": "change1", "title": "Título en español"}],
                "first_mappings": {"title": {"type": "text_field", "language": "es"}},
                "second_docs": [{"_id": "change2", "title": "Titre en français"}],
                "second_mappings": {"title": {"type": "text_field", "language": "fr"}},
                "should_error": True
            },
            {
                "name": "Default to specific language",
                "description": "Error when field had default language first but specific language second time",
                "first_docs": [{"_id": "default1", "title": "Default title"}],
                "first_mappings": {},  # No mappings = default language
                "second_docs": [{"_id": "specific1", "title": "Specific language title"}],
                "second_mappings": {"title": {"type": "text_field", "language": "en"}},
                "should_error": True
            },
            {
                "name": "Specific to default language",
                "description": "Allowed when field had specific language first but default second time",
                "first_docs": [{"_id": "specific1", "title": "English title"}],
                "first_mappings": {"title": {"type": "text_field", "language": "en"}},
                "second_docs": [{"_id": "default2", "title": "Default title"}],
                "second_mappings": {},  # No mappings = default language
                "should_error": False
            }
        ]

        for scenario in language_change_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Clear index for each scenario
                self.clear_index_by_index_name(self.index.name)
                
                # First add documents with initial language configuration
                with self.subTest(step="first_add"):
                    response1 = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=self.index.name,
                            docs=scenario["first_docs"],
                            tensor_fields=[],
                            mappings=scenario["first_mappings"]
                        )
                    )
                    # First add should always succeed
                    self.assertFalse(response1.errors, f"First add failed for {scenario['name']}")

                # Then try to add documents with different language configuration
                with self.subTest(step="second_add"):
                    response2 = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=self.index.name,
                            docs=scenario["second_docs"],
                            tensor_fields=[],
                            mappings=scenario["second_mappings"]
                        )
                    )
                    
                    # Check if result matches expectation
                    if scenario["should_error"]:
                        self.assertTrue(response2.errors, 
                                      f"Expected error for {scenario['name']} but got success")
                    else:
                        self.assertFalse(response2.errors, 
                                       f"Expected success for {scenario['name']} but got error")
                        
                        # If successful, verify both documents exist
                        if not response2.errors:
                            with self.subTest(step="verify_docs"):
                                doc1 = tensor_search.get_document_by_id(
                                    config=self.config,
                                    index_name=self.index.name,
                                    document_id=scenario["first_docs"][0]["_id"]
                                )
                                doc2 = tensor_search.get_document_by_id(
                                    config=self.config,
                                    index_name=self.index.name,
                                    document_id=scenario["second_docs"][0]["_id"]
                                )
                                
                                self.assertEqual(doc1["title"], scenario["first_docs"][0]["title"])
                                self.assertEqual(doc2["title"], scenario["second_docs"][0]["title"])
