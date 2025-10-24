"""
Comprehensive integration tests for language feature.
Tests language functionality with different search methods and field configurations.
Follows TestSearchSemiStructured pattern with focus on unstructured index testing.
"""
import copy
import os
import uuid
from unittest import mock

from marqo.core.exceptions import UnsupportedFeatureError, AddDocumentsError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.marqo_index import Model, FieldType, FieldFeature
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel, RelevanceCutoffMethod, \
    MeanStdParameters
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLanguage(MarqoTestCase):
    """Integration tests for language functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        index_requests = []

        cls.three_fields_index = cls.unstructured_marqo_index_request(
            name='test_three_fields_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.three_fields_index)

        cls.language_change_different_index = cls.unstructured_marqo_index_request(
            name='test_lang_change_diff_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.language_change_different_index)

        cls.language_change_default_to_specific_index = cls.unstructured_marqo_index_request(
            name='test_lang_change_def_to_spec_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.language_change_default_to_specific_index)

        cls.language_change_specific_to_default_index = cls.unstructured_marqo_index_request(
            name='test_lang_change_spec_to_def_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.language_change_specific_to_default_index)

        # Create separate indexes for each language in the all_languages test
        cls.en_us_index = cls.unstructured_marqo_index_request(
            name='test_lang_en_us_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.en_us_index)

        cls.es_index = cls.unstructured_marqo_index_request(
            name='test_lang_es_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.es_index)

        cls.fr_index = cls.unstructured_marqo_index_request(
            name='test_lang_fr_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.fr_index)

        cls.de_index = cls.unstructured_marqo_index_request(
            name='test_lang_de_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.de_index)

        cls.pt_br_index = cls.unstructured_marqo_index_request(
            name='test_lang_pt_br_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.pt_br_index)

        # Create a separate index for the language persistence test
        cls.language_persistence_index = cls.unstructured_marqo_index_request(
            name='test_lang_persistence_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2')
        )
        index_requests.append(cls.language_persistence_index)

        cls.facets_and_relevance_cutoff_index = cls.unstructured_marqo_index_request(
            name='test_facets_and_relevance_cutoff_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
        )
        index_requests.append(cls.facets_and_relevance_cutoff_index)

        # Create a structured index for testing language search failure
        cls.structured_index = cls.structured_marqo_index_request(
            name='test_structured_lang_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
            fields=[
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="content", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
            ],
            tensor_fields=["title", "content"]
        )
        index_requests.append(cls.structured_index)

        # 2.16 unstructured index -- must support language
        cls.v216_unstructured_index = cls.unstructured_marqo_index_request(
            name='test_v216_unstructured_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
            marqo_version='2.16.0',
        )
        index_requests.append(cls.v216_unstructured_index)

        # 2.15 unstructured index -- must not support language
        cls.v215_unstructured_index = cls.unstructured_marqo_index_request(
            name='test_v215_unstructured_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
            marqo_version='2.15.0',
        )
        index_requests.append(cls.v215_unstructured_index)

        # Batch create all indexes
        cls.indexes = cls.create_indexes(index_requests)

    def test_language_add_docs_and_search(self):
        """Positive test cases for 5 languages with all three search methods."""

        # Note for non-english, it's been verified that test queries don't get any hits
        # with automatic language detection

        language_test_data = {
            "en-US": {
                "docs": [
                    {"_id": "en1", "title": "Running in the beautiful park", "content": "The athlete runs quickly"},
                    {"_id": "en2", "title": "Running at the ocean", "content": "Swimming is excellent exercise"},
                    {"_id": "en3", "title": "Reading interesting books",
                     "content": "Books provide knowledge and entertainment"}
                ],
                "search_tests": [
                    ("running", ["en1", "en2"]),
                    ("books", ["en3"])
                ]
            },
            "es": {
                "docs": [
                    {"_id": "es1", "title": "Corriendo en el parque hermoso", "content": "El atleta corre rápidamente"},
                    {"_id": "es2", "title": "Nadando en el océano", "content": "Nadar es excelente ejercicio"},
                    {"_id": "es3", "title": "Leyendo libros interesantes",
                     "content": "Los libros proporcionan conocimiento"}
                ],
                "search_tests": [
                    ("Corriendo", ['es1']),
                    ("Corriendo Nadando", ['es1', 'es2']),
                ]
            },
            "fr": {
                "docs": [
                    {"_id": "fr1", "title": "Courant dans le parc magnifique", "content": "L'athlète court rapidement"},
                    {"_id": "fr2", "title": "Nageant dans l'océan", "content": "La natation est un excellent exercice"},
                    {"_id": "fr3", "title": "Lisant des livres intéressants",
                     "content": "Les livres fournissent des connaissances"}
                ],
                "search_tests": [
                    ("Courant", ['fr1']),
                    ("Courant Nageant livres", ['fr1', 'fr2', 'fr3']),
                ]
            },
            "de": {
                "docs": [
                    {"_id": "de1", "title": "Laufen im schönen Park", "content": "Der Athlet läuft schnell"},
                    {"_id": "de2", "title": "Schwimmen im Ozean", "content": "Schwimmen ist ausgezeichnete Übung"},
                    {"_id": "de3", "title": "Interessante Bücher lesen",
                     "content": "Bücher bieten Wissen und Unterhaltung"}
                ],
                "search_tests": [
                    ("Laufen", ['de1']),
                    ("Schwimmen Bücher", ['de2', 'de3']),
                    ("Schwimmen Bücher schönen", ['de1', 'de2', 'de3']),
                ]
            },
            "pt-BR": {
                "docs": [
                    {"_id": "pt1", "title": "Correndo no parque bonito", "content": "O atleta corre rapidamente"},
                    {"_id": "pt2", "title": "Nadando no oceano", "content": "Nadar é excelente exercício"},
                    {"_id": "pt3", "title": "Lendo livros interessantes", "content": "Livros fornecem conhecimento"}
                ],
                "search_tests": [
                    ("Correndo", ['pt1']),
                    ("Correndo oceano Lendo", ['pt1', 'pt2', 'pt3']),
                ]
            }
        }

        # Test each language with its own pre-created index
        language_index_map = {
            "en-US": self.en_us_index,
            "es": self.es_index,
            "fr": self.fr_index,
            "de": self.de_index,
            "pt-BR": self.pt_br_index
        }

        for language, test_data in language_test_data.items():
            with self.subTest(language=language):
                created_index = language_index_map[language]

                mappings = {
                    "title": {"type": "text_field", "language": language},
                    "content": {"type": "text_field", "language": language}
                }

                response = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=created_index.name,
                        docs=test_data["docs"],
                        tensor_fields=["title"],
                        mappings=mappings
                    )
                )

                # Check for errors in add_documents response
                self.assertFalse(response.errors,
                                 f"Failed to add documents: {response.errors if hasattr(response, 'errors') else 'Unknown error'}")

                # Test all three search methods
                search_methods = [
                    (SearchMethod.LEXICAL, "lexical", None),
                    (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Lexical
                    )),
                    (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Tensor
                    )),
                    (SearchMethod.HYBRID, "hybrid_rrf", None)
                ]

                for search_method, method_name, hybrid_params in search_methods:
                    with self.subTest(search_method=method_name):
                        for search_term, expected_ids in test_data["search_tests"]:
                            with self.subTest(search_term=search_term):
                                result = tensor_search.search(
                                    config=self.config,
                                    index_name=created_index.name,
                                    text=search_term,
                                    search_method=search_method,
                                    hybrid_parameters=hybrid_params,
                                    language=language,
                                )

                                hit_ids = [hit["_id"] for hit in result["hits"]]
                                if method_name == "lexical" or hybrid_params is not None:
                                    self.assertEqual(sorted(hit_ids), sorted(expected_ids))
                                else:  # rrf can have other results
                                    self.assertTrue(set(expected_ids).issubset(set(hit_ids)),
                                                    f"Expected {expected_ids} in results for "
                                                    f"{search_term} in {language} using {method_name}")

    def test_index_with_three_text_fields_different_languages(self):
        """Test index with three text fields: French, Portuguese, and English (automatic detection)."""
        docs = [
            {
                "_id": "multi1",
                "french_field": "Bonjour le monde français",
                "portuguese_field": "Olá mundo português",
                "english_field": "White dog"
            },
            {
                "_id": "multi2",
                "french_field": "Chat noir français",
                "portuguese_field": "Gato preto português",
                "english_field": "Black cat"
            }
        ]

        mappings = {
            "french_field": {"type": "text_field", "language": "fr"},
            "portuguese_field": {"type": "text_field", "language": "pt"},
            # english_field has no mapping (automatic detection)
        }

        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.three_fields_index.name,
                docs=docs,
                tensor_fields=["english_field"],  # Make some lexical fields tensor fields but not all
                mappings=mappings
            )
        )

        # Check for errors in add_documents response
        self.assertFalse(response.errors,
                         f"Failed to add documents: {response.errors if hasattr(response, 'errors') else 'Unknown error'}")

        # Define test scenarios for different language content searches
        search_scenarios = [
            ("français", "fr", ["multi1", "multi2"]),
            ("português", "pt", ["multi1", "multi2"]),
            ("Cat", None, ["multi2"]),
            ("Cat", "en", ["multi2"]),
        ]

        # Test each scenario with different search methods
        search_methods = [
            (SearchMethod.LEXICAL, "lexical", None),
            (SearchMethod.HYBRID, "hybrid_lexical", HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical
            )),
            (SearchMethod.HYBRID, "hybrid_rrf", HybridParameters(
                alpha=0  # lexical
            ))
        ]

        for search_term, language, expected_ids in search_scenarios:
            with self.subTest(search_term=search_term, language=language):
                for search_method, method_name, hybrid_params in search_methods:
                    with self.subTest(search_method=method_name):
                        result = tensor_search.search(
                            config=self.config,
                            index_name=self.three_fields_index.name,
                            text=search_term,
                            search_method=search_method,
                            hybrid_parameters=hybrid_params,
                            language=language
                        )

                        # Assert that we get some results
                        self.assertGreater(len(result["hits"]), 0,
                                           f"No results for {search_term} using {method_name}")

                        # Check that expected IDs are in results
                        hit_ids = [hit["_id"] for hit in result["hits"]]
                        self.assertTrue(set(expected_ids).issubset(set(hit_ids)),
                                        f"Expected {expected_ids} in results for {search_term} "
                                        f"using {method_name}")

        # Test tensor search, hybrid RRF on English
        search_methods = [SearchMethod.TENSOR, SearchMethod.HYBRID]
        for search_method in search_methods:
            with self.subTest(search_method=search_method):
                result = tensor_search.search(
                    config=self.config,
                    index_name=self.three_fields_index.name,
                    text="golden retriever",
                    search_method=search_method,
                )

                # Assert results in the exact order
                self.assertGreater(len(result["hits"]), 0)
                hits_ids = [hit["_id"] for hit in result["hits"]]
                self.assertEqual(['multi1', 'multi2'], hits_ids)

    def test_language_tensor_lexical(self):
        """Test index with three text fields: French, Portuguese, and English (automatic detection)."""
        docs = [
            {
                "_id": "1",
                "title": "Green tree",
                "description": "Grande collection",
            },
            {
                "_id": "2",
                "title": "Black dog",
                "description": "Grande maison",
            },
            {
                "_id": "3",
                "title": "Big tree",
                "description": "Grande collection connexions",
            },
        ]

        mappings = {
            "description": {"type": "text_field", "language": "fr"},
        }

        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.three_fields_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )

        self.assertFalse(response.errors,
                         f"Failed to add documents: {response.errors if hasattr(response, 'errors') else 'Unknown error'}")

        result = tensor_search.search(
            config=self.config,
            index_name=self.three_fields_index.name,
            text='dog collection connexions',
            search_method='hybrid',
            hybrid_parameters=HybridParameters(
                retrievalMethod='tensor',
                rankingMethod='lexical',
            ),
            language='fr'
        )

        # Verify the expected order when reranked lexically -- we've also verified this order is only
        # returned when the language is set to French
        self.assertGreater(len(result["hits"]), 0)
        hits_ids = [hit["_id"] for hit in result["hits"]]
        self.assertEqual(['3', '2', '1'], hits_ids)

    def test_lexical_hybrid_searchable_attributes(self):
        """Test lexical and hybrid lexical/lexical search with searchable_attributes and hybridParameters.searchableAttributesLexical."""

        # French documents from test_language_add_docs_and_search
        french_docs = [
            {"_id": "fr1", "title": "Courant dans le parc magnifique", "content": "L'athlète court rapidement"},
            {"_id": "fr2", "title": "Nageant dans l'océan", "content": "La natation est un excellent exercice"},
            {"_id": "fr3", "title": "Lisant des livres intéressants",
             "content": "Les livres fournissent des connaissances"}
        ]

        mappings = {
            "title": {"type": "text_field", "language": "fr"},
            "content": {"type": "text_field", "language": "fr"}
        }

        # Add documents to French index
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.fr_index.name,
                docs=french_docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )
        self.assertFalse(response.errors, "Failed to add French documents")

        # Test cases for searchable_attributes and hybridParameters.searchableAttributesLexical
        test_cases = [
            {
                "name": "title_only",
                "searchable_attributes": ["title"],
                "hybrid_searchable_attributes": ["title"],
                "query": "Courant",
                "expected_ids": ["fr1"],
                "description": "Search for 'Courant' only in title field"
            },
            {
                "name": "content_only",
                "searchable_attributes": ["content"],
                "hybrid_searchable_attributes": ["content"],
                "query": "rapidement",
                "expected_ids": ["fr1"],
                "description": "Search for 'rapidement' only in content field"
            }
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                # Test lexical search with searchable_attributes
                with self.subTest(search_method="lexical"):
                    result = tensor_search.search(
                        config=self.config,
                        index_name=self.fr_index.name,
                        text=case["query"],
                        search_method=SearchMethod.LEXICAL,
                        searchable_attributes=case["searchable_attributes"],
                        language="fr"
                    )

                    hit_ids = [hit["_id"] for hit in result["hits"]]
                    self.assertEqual(hit_ids, case["expected_ids"])

                # Test hybrid lexical/lexical search with hybridParameters.searchableAttributesLexical
                with self.subTest(search_method="hybrid_lexical_lexical"):
                    result = tensor_search.search(
                        config=self.config,
                        index_name=self.fr_index.name,
                        text=case["query"],
                        search_method=SearchMethod.HYBRID,
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Lexical,
                            searchableAttributesLexical=case["hybrid_searchable_attributes"]
                        ),
                        language="fr"
                    )

                    hit_ids = [hit["_id"] for hit in result["hits"]]
                    self.assertEqual(hit_ids, case["expected_ids"])

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
                "should_error": True,
                "index": self.language_change_different_index
            },
            {
                "name": "Default to specific language",
                "description": "Error when field had default language first but specific language second time",
                "first_docs": [{"_id": "default1", "title": "Default title"}],
                "first_mappings": {},  # No mappings = default language
                "second_docs": [{"_id": "specific1", "title": "Specific language title"}],
                "second_mappings": {"title": {"type": "text_field", "language": "en"}},
                "should_error": True,
                "index": self.language_change_default_to_specific_index
            },
            {
                "name": "Specific to default language",
                "description": "Allowed when field had specific language first but default second time",
                "first_docs": [{"_id": "specific1", "title": "English title"}],
                "first_mappings": {"title": {"type": "text_field", "language": "en"}},
                "second_docs": [{"_id": "default2", "title": "Default title"}],
                "second_mappings": {},  # No mappings = default language
                "should_error": False,
                "index": self.language_change_specific_to_default_index
            }
        ]

        for scenario in language_change_scenarios:
            with self.subTest(scenario=scenario["description"]):
                # First add documents with initial language configuration
                with self.subTest(step="first_add"):
                    response1 = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=scenario["index"].name,
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
                            index_name=scenario["index"].name,
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
                        with self.subTest(step="verify_docs"):
                            doc1 = tensor_search.get_document_by_id(
                                config=self.config,
                                index_name=scenario["index"].name,
                                document_id=scenario["first_docs"][0]["_id"]
                            )
                            doc2 = tensor_search.get_document_by_id(
                                config=self.config,
                                index_name=scenario["index"].name,
                                document_id=scenario["second_docs"][0]["_id"]
                            )

                            self.assertEqual(doc1["title"], scenario["first_docs"][0]["title"])
                            self.assertEqual(doc2["title"], scenario["second_docs"][0]["title"])

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

        mappings = {
            "title_pt": {"type": "text_field", "language": "pt"},
            "title_en": {"type": "text_field", "language": "en"},
        }

        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.three_fields_index.name,
                docs=docs,
                tensor_fields=[],
                mappings=mappings
            )
        )

        self.assertFalse(response.errors, "Should not have errors when adding documents")

        # Test Portuguese search
        res_pt = tensor_search.search(
            config=self.config,
            index_name=self.three_fields_index.name,
            text="mole",
            search_method=SearchMethod.LEXICAL,
            language="pt"
        )

        # Test English search
        res_en = tensor_search.search(
            config=self.config,
            index_name=self.three_fields_index.name,
            text="mole",
            search_method=SearchMethod.LEXICAL,
            language="en"
        )

        hits_pt = [hit["_id"] for hit in res_pt["hits"]]
        hits_en = [hit["_id"] for hit in res_en["hits"]]

        self.assertEqual(["1"], hits_pt, "Should find only the Portuguese doc")
        self.assertEqual(["2"], hits_en, "Should find only the English doc")

    def test_facets_and_relevance_cutoff(self):
        """Test facets and relevance cutoff with language parameter."""
        # Populate the index with documents
        docs = [
            {
                "_id": "1",
                "title": "Vestido Mole Perfeito",  # Portuguese
                "size": "M"
            },
            {
                "_id": "2",
                "title": "Vestido Mole Confortável",
                "size": "M"
            },
            {
                "_id": "3",
                "title": "Vestido Leve e Elegante",
                "size": "S"
            }
        ]

        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.facets_and_relevance_cutoff_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings={
                    "title": {"type": "text_field", "language": "pt"},
                }
            )
        )

        self.assertFalse(response.errors, "Should not have errors when adding documents")

        cases = [
            ("pt", True),
            ("en", False)
        ]

        for language, matches in cases:
            with self.subTest(f"Testing facets and relevance cutoff for language: {language}"):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.facets_and_relevance_cutoff_index.name,
                    text="mole",
                    search_method="HYBRID",
                    language=language,
                    hybrid_parameters=HybridParameters(
                        searchableAttributesLexical=['title'],
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Lexical
                    ),
                    facets=FacetsParameters(
                        fields={
                            "size": FieldFacetsConfiguration(type='string')
                        }
                    ),
                    relevance_cutoff=RelevanceCutoffModel(method=RelevanceCutoffMethod.MeanStdDev,
                                                          parameters=MeanStdParameters(stdDevFactor=-100)
                                                          )
                )

                if matches:
                    self.assertGreater(len(res["hits"]), 0, "Should find matches for 'mole'")
                    self.assertGreater(len(res['facets']['size']), 0, "Should have facets for 'size'")
                    self.assertGreater(res["_probeCandidates"], 0,
                                       "Should have relevant candidates count greater than 0")
                else:
                    self.assertEqual(len(res["hits"]), 0, "Should find no matches for 'mole' in English")
                    self.assertEqual(len(res['facets']), 0, "Should have no facets for 'size' in English")
                    self.assertEqual(res["_probeCandidates"], 0,
                                     "Should have no relevant candidates count in English")

    def test_structured_index_language_search_fails(self):
        """Test that searching with language parameter fails for structured indexes."""

        # Add documents to the structured index
        docs = [
            {"_id": "doc1", "title": "Running in the park", "content": "Exercise is good for health"},
            {"_id": "doc2", "title": "Swimming in the ocean", "content": "Water sports are fun"}
        ]

        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.structured_index.name,
                docs=docs
            )
        )

        # Verify documents were added successfully
        self.assertFalse(response.errors, "Failed to add documents to structured index")

        # Test search methods with language parameter
        search_tests = [
            (SearchMethod.LEXICAL, "lexical", "running"),
            (SearchMethod.HYBRID, "hybrid_default", "swimming")
        ]

        for search_method, method_name, search_text in search_tests:
            with self.subTest(search_method=method_name):
                with self.assertRaises(UnsupportedFeatureError) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=self.structured_index.name,
                        text=search_text,
                        search_method=search_method,
                        language="en-US"
                    )

                # Verify we get an appropriate error
                self.assertIn("language", str(cm.exception).lower())

    def test_v216_unstructured_index_supports_language_mapping(self):
        """Test that v2.16 unstructured index supports language mapping."""
        docs = [
            {"_id": "v216_doc1", "title": "Corriendo en el parque hermoso", "content": "El atleta corre rápidamente"},
            {"_id": "v216_doc2", "title": "Nadando en el océano", "content": "Nadar es excelente ejercicio"},
            {"_id": "v216_doc3", "title": "Leyendo libros interesantes",
             "content": "Los libros proporcionan conocimiento"}
        ]

        mappings = {
            "title": {"type": "text_field", "language": "es"},
            "content": {"type": "text_field", "language": "es"}
        }

        # Add documents with language mapping
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.v216_unstructured_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )

        # Verify documents were added successfully
        self.assertFalse(response.errors, "Failed to add documents with language mapping to v2.16 index")

        # Test search with language parameter
        result = tensor_search.search(
            config=self.config,
            index_name=self.v216_unstructured_index.name,
            text="Corriendo",
            search_method=SearchMethod.LEXICAL,
            language="es"
        )

        # Verify we get hits
        self.assertGreater(len(result["hits"]), 0, "Expected hits for Spanish search on v2.16 index")
        hit_ids = [hit["_id"] for hit in result["hits"]]
        self.assertIn("v216_doc1", hit_ids, "Expected v216_doc1 in search results")

    def test_v215_unstructured_index_rejects_language_mapping(self):
        """Test that v2.15 unstructured index rejects language mapping."""
        docs = [
            {"_id": "v215_doc1", "title": "Corriendo en el parque hermoso", "content": "El atleta corre rápidamente"},
            {"_id": "v215_doc2", "title": "Nadando en el océano", "content": "Nadar es excelente ejercicio"}
        ]

        mappings = {
            "title": {"type": "text_field", "language": "es"},
            "content": {"type": "text_field", "language": "es"}
        }

        # Try to add documents with language mapping - this should fail
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.v215_unstructured_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )

        # Verify that adding documents with language mapping fails
        self.assertTrue(
            response.errors,
            "Expected error when adding documents with language mapping to v2.15 index"
        )

        # Check the specific error message content in the response items
        error_items = [item for item in response.items if item.error]
        self.assertEqual(2, len(error_items), "Expected two errors items in response")

        # Verify all items have the language version error
        for error_item in error_items:
            self.assertIn(
                "Language is only supported for indexes created with Marqo version 2.16.0 or later",
                error_item.error
            )
            self.assertIn("2.15.0", error_item.error)

    def test_language_persists_when_not_specified_in_subsequent_adds(self):
        """Test that language configuration persists when not specified in subsequent document additions."""
        # First add document with explicit language=pt
        docs1 = [{"_id": "pt_doc1", "content": "mole"}]
        mappings1 = {"content": {"type": "text_field", "language": "pt"}}

        add_docs_params1 = AddDocsParams(
            index_name=self.language_persistence_index.name,
            docs=docs1,
            mappings=mappings1,
            tensor_fields=[]
        )

        response1 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params1
        )
        self.assertFalse(response1.errors, "Should not have errors when adding first document")

        # Add second document to same field without specifying mappings
        # This should use the existing field configuration (language=pt)
        docs2 = [{"_id": "pt_doc2", "content": "mole"}]

        add_docs_params2 = AddDocsParams(
            index_name=self.language_persistence_index.name,
            docs=docs2,
            tensor_fields=[]
        )

        response2 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params2
        )
        self.assertFalse(response2.errors, "Should not have errors when adding second document")

        # Test that language=pt is still in effect for both documents
        # Search for "mole" with language=en should not find any hits (because both docs are pt)
        en_res = tensor_search.search(
            config=self.config,
            index_name=self.language_persistence_index.name,
            text="mole",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="en"
        )

        # With language=en, "mole" should not match Portuguese "mole" documents
        en_ids = {hit["_id"] for hit in en_res["hits"]}
        self.assertEqual(set(), en_ids,
                         "Should not find any documents when searching with wrong language (en instead of pt)")

        # Verify that searching with correct language (pt) finds both documents
        pt_res = tensor_search.search(
            config=self.config,
            index_name=self.language_persistence_index.name,
            text="mole",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="pt"
        )

        pt_ids = {hit["_id"] for hit in pt_res["hits"]}
        self.assertEqual({"pt_doc1", "pt_doc2"}, pt_ids,
                         "Should find both documents when searching with correct language (pt)")
