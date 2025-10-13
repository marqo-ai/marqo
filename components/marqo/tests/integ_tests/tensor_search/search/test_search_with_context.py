import os
import unittest
from unittest import mock

from marqo.tensor_search.enums import EnvVars
from pydantic.v1.error_wrappers import ValidationError
from marqo.core.exceptions import InvalidFieldNameError, UnsupportedFeatureError
from marqo.api.exceptions import InvalidArgError, IllegalRequestedDocCount
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor, SearchContextDocuments, SearchContextDocumentsParameters
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.utils.vector_interpolation import Slerp, Lerp, Nlerp, AllZeroWeightsError, ZeroMagnitudeVectorError
from marqo.exceptions import InvalidArgumentError, InternalError
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from tests.integ_tests.marqo_test import MarqoTestCase
from tests.integ_tests.utils.transition import *
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.tensor_search.models.api_models import CustomVectorQuery
from marqo.core import exceptions as core_exceptions

class TestSearchWithContext(MarqoTestCase):

    structured_index_basic = "structured_index_basic"
    unstructured_index_basic = "unstructured_index_basic"
    legacy_unstructured_index_basic = "legacy_unstructured_index_basic"
    
    # The index in this test is created with 'hf/all-MiniLM-L6-v2' with 384 dimensions
    # Don't use random model for this test suite as we need to guarantee the same query generate the same embeddings
    DIMENSION = 384

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index_basic_request = cls.structured_marqo_index_request(
            name=cls.structured_index_basic,
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='text_field_2', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='non_vector_text_field', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
                FieldRequest(name='score_field', type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="tags", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter])
            ],
            tensor_fields=["text_field_1", "text_field_2", "image_field_1"]
        )
        unstructured_index_basic_request = cls.unstructured_marqo_index_request(
            name=cls.unstructured_index_basic,
            model=Model(name="hf/all-MiniLM-L6-v2"),
        )

        legacy_unstructured_index_basic_request = cls.unstructured_marqo_index_request(
            name=cls.legacy_unstructured_index_basic,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            marqo_version='2.12.0'
        )

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_index_basic_request,
            unstructured_index_basic_request,
            legacy_unstructured_index_basic_request
        ])


        # Default text indexes for the context.documents tests
        cls.structured_default_text_index = cls.indexes[0]  # Use the structured index we created
        cls.unstructured_default_text_index = cls.indexes[1]  # Use the unstructured index we created
        cls.legacy_unstructured_default_text_index = cls.indexes[2]  # Use the legacy unstructured index we created

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        super().setUp()
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def _populate_index_orchids(self, index):
        """Helper method to populate an index with orchid and related test documents.
        
        This method adds a standardized set of test documents including orchids, flowers,
        and continents that can be used for testing context document functionality.
        
        Args:
            index: The index to populate (structured or unstructured)
            
        Returns:
            List of added document dictionaries
        """
        docs = [
            {"_id": "orchid1", "text_field_1": "Anacamptis laxiflora is a species of orchid found in wet meadows with alkaline soil.", "tags": ["flower", "orchid"]},
            {"_id": "orchid2", "text_field_1": "Cephalanthera longifolia reaches on average 20-60 centimetres in height and is a type of orchid.", "tags": ["flower", "orchid"]},
            {"_id": "orchid3", "text_field_1": "Anacamptis morio subsp. longicornu is a subspecies of orchid found in the Mediterranean region.", "tags": ["flower", "orchid"]},
            {"_id": "flower1", "text_field_1": "Red rose is a popular flower known for its beauty and fragrance.", "tags": ["flower", "rose"]},
            {"_id": "continent1", "text_field_1": "Europe is a continent located entirely in the Northern Hemisphere and mostly in the Eastern Hemisphere.", "tags": ["continent"]},
            {"_id": "continent2", "text_field_1": "Asia is Earth's largest and most populous continent, located primarily in the Eastern and Northern Hemispheres.", "tags": ["continent"]},
            {"_id": "continent3", "text_field_1": "Africa is the world's second-largest and second-most populous continent, after Asia in both cases.", "tags": ["continent"]},
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs,
                tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
            )
        )
        
        return docs

    # Search with context.tensor
    def test_search(self):
        for index_name in [self.structured_index_basic, self.unstructured_index_basic,
                           self.legacy_unstructured_index_basic]:
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }
                res = tensor_search.search(config=self.config, index_name=index_name, text=query,
                                           context=SearchContext(
                                               **{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 2},
                                                             {"vector": [2, ] * self.DIMENSION, "weight": -1}]}))

    def test_search_with_incorrect_tensor_dimension(self):
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }
                with self.assertRaises(InvalidArgError) as e:
                    tensor_search.search(config=self.config, index_name=index_name, text=query, context=SearchContext(
                        **{"tensor": [{"vector": [1, ] * 3, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], }))
                self.assertIn("does not match the expected dimension", str(e.exception.message))

    def test_search_with_incorrect_query_format(self):
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                query = "A rider is riding a horse jumping over the barrier"
                with self.assertRaises(InvalidArgError) as e:
                    res = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                    SearchContext(
                        **{"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}]}))
                self.assertIn("This is not supported as the context only works when the query is a dictionary.",
                              str(e.exception.message))

    def test_search_score(self):
        """Test to ensure that the score is the same for the same query with different context vectors combinations."""
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            tensor_fields = ["text_field_1"] if index_name == self.unstructured_index_basic else None
            self.add_documents(config=self.config, add_docs_params=
                                        AddDocsParams(index_name=index_name,
                                                      docs=[{"text_field_1": "A rider", "_id": "1"}],
                                                      tensor_fields=tensor_fields
                                                      )
                               )
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }

                res_1 = tensor_search.search(config=self.config, index_name=index_name, text=query)
                res_2 = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                SearchContext(**{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 0}, {"vector": [2, ] * self.DIMENSION, "weight": 0}], }))
                res_3 = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                SearchContext(**{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": -1}, {"vector": [1, ] * self.DIMENSION, "weight": 1}], }))

                self.assertEqual(res_1["hits"][0]["_score"], res_2["hits"][0]["_score"])
                self.assertEqual(res_1["hits"][0]["_score"], res_3["hits"][0]["_score"])

    def test_context_vector_with_none_query(self):
        """Test to ensure that the context vector can be used without a query."""
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                res = tensor_search.search(text=None, config=self.config, index_name=index_name, context=SearchContext(
                    **{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 1},
                                  {"vector": [2, ] * self.DIMENSION, "weight": 2}]}))

    def test_context_vector_raise_error_if_query_and_context_are_none(self):
        """Test to ensure that a proper error is raised if both query and context is None"""
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                with self.assertRaises(ValidationError) as e:
                    res = tensor_search.search(text=None, config=self.config, index_name=index_name, context=None)
                self.assertIn("One of Query(q) or context is required for TENSOR search",
                              str(e.exception))

    # Search with context.documents
    def test_search_with_context_documents_only(self):
        """Test that search works correctly when only context documents are provided (no query, no context tensor).

        This test verifies that when we search using only document IDs as context,
        the search results match what we'd expect based on those documents.

        Checks tensorFields and excludeInputDocuments parameters.
        """
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "machine learning algorithms and artificial intelligence"},
                    {"_id": "doc2", "text_field_1": "deep neural networks for computer vision tasks"},
                    {"_id": "doc3", "text_field_1": "natural language processing and text generation"},
                    {"_id": "doc4", "text_field_1": "reinforcement learning for game playing"},
                    {"_id": "doc5", "text_field_1": "statistical models for data analysis"},
                    {"_id": "doc6", "text_field_1": "clustering algorithms for unsupervised learning"},
                    # doc7 can only be extracted with text_field_2
                    {"_id": "doc7", "text_field_2": "completely different idea"},
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1", "text_field_2"] if isinstance(index,
                                                                                     UnstructuredMarqoIndex) else None
                    )
                )

                with self.subTest(exclude_input_documents=False, tensor_fields=["text_field_1"]):
                    # Create search context with only documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            ids={"doc1": 3.0, "doc3": 5.0, "doc5": -5.0},
                            parameters=SearchContextDocumentsParameters(
                                excludeInputDocuments=False,
                                tensorFields=["text_field_1"]
                            )
                        )
                    )

                    # Perform search with only context documents
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=10
                    )

                    # Verify search results
                    self.assertIn("hits", results)

                    # The input documents should be the first results if excludeInputDocuments is False
                    result_ids = [hit["_id"] for hit in results["hits"]]

                    # Verify that the first hit is doc3, then doc1. doc5 should be at the bottom.
                    self.assertEqual(result_ids, ["doc3", "doc1", "doc4", "doc2", "doc6", "doc7", "doc5"])

                with self.subTest(exclude_input_documents=True, tensor_fields=["text_field_1"]):
                    # Create search context with only documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            ids={"doc1": 3.0, "doc3": 5.0, "doc5": -5.0},
                            parameters=SearchContextDocumentsParameters(
                                excludeInputDocuments=True,
                                tensorFields=["text_field_1"]
                            )
                        )
                    )

                    # Perform search with only context documents
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=10
                    )

                    # Verify search results
                    self.assertIn("hits", results)

                    # The input documents should not be in the results if excludeInputDocuments is True
                    result_ids = [hit["_id"] for hit in results["hits"]]

                    # Verify that doc1 and doc3 are not in the results
                    self.assertEqual(result_ids, ["doc4", "doc2", "doc6", "doc7"])

                # Using tensor fields
                with self.subTest(excludeInputDocuments=False, tensor_fields=["tensor_field_2"]):
                    # Create search context with only documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            ids={"doc7": 1},
                            parameters=SearchContextDocumentsParameters(
                                excludeInputDocuments=False,
                                tensorFields=["text_field_2"]
                            )
                        )
                    )

                    # Perform search with only context documents
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=10
                    )

                    # Verify search results
                    self.assertIn("hits", results)

                    # The input documents should not be in the results if excludeInputDocuments is True
                    result_ids = [hit["_id"] for hit in results["hits"]]

                    # doc7 will be on top, because it's the only one with text_field_2
                    self.assertEqual(result_ids[0], "doc7")
                    self.assertEqual(results["hits"][0]["_score"], 1.0)

    def test_search_with_context_documents_tensors_and_queries(self):
        """Test that search works correctly when context documents, tensors, and queries are provided.
        Use relevant data and sample searches
        """
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "red shirt with collar unisex"},
                    {"_id": "doc2", "text_field_1": "black long pants for men"},
                    {"_id": "doc3", "text_field_1": "black shorts unisex"},
                    {"_id": "doc4", "text_field_1": "black shirt for men"},
                    {"_id": "doc5", "text_field_1": "grey pants for women"},
                    {"_id": "doc6", "text_field_1": "green hat unisex"},
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Basic search (query)
                with self.subTest("Basic query"):
                    basic_results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text={"shirt": 1, "black": -0.5},
                        result_count=6
                    )

                    # Verify the 2 shirt documents are the top 2
                    self.assertIn("hits", basic_results)
                    result_ids = [hit["_id"] for hit in basic_results["hits"]]
                    self.assertEqual(result_ids[0], "doc1")
                    self.assertEqual(result_ids[1], "doc4")
                    # Last 2 docs have "black", thus pushing them to the bottom (negative weighted query)
                    self.assertEqual(result_ids[-2], "doc2")
                    self.assertEqual(result_ids[-1], "doc3")

                # Use context documents to put doc1 at the bottom, bring doc6 to the top
                with self.subTest("With context documents"):
                    results_with_context_docs = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text={"shirt": 1, "black": -0.5},
                        context=SearchContext(
                            documents=SearchContextDocuments(
                                ids={"doc1": -10.0, "doc6": 10},
                                parameters=SearchContextDocumentsParameters(
                                    tensorFields=["text_field_1"],
                                    excludeInputDocuments=False
                                )
                            )
                        ),
                        interpolation_method="nlerp",
                        result_count=6
                    )

                    # Verify search results
                    self.assertIn("hits", results_with_context_docs)
                    result_ids = [hit["_id"] for hit in results_with_context_docs["hits"]]
                    self.assertEqual(result_ids[0], "doc6")
                    # doc1 should be at the bottom
                    self.assertEqual(result_ids[-1], "doc1")
                    # TODO: Fix SLERP here, maybe don't have negative weights first.

    def test_search_with_context_documents_missing_documents(self):
        """Test that search with context documents fails when documents don't exist."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add some documents
                docs = [
                    {"_id": "doc1", "text_field_1": "Test document one"},
                    {"_id": "doc2", "text_field_1": "Test document two"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with non-existent document
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 1.0, "non_existent_doc": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Verify error is raised for missing document
                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5
                    )

    def test_search_with_context_documents_exclude_input_succeeds(self):
        """Test that search with context documents excludes input documents when requested."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "Test document one about artificial intelligence"},
                    {"_id": "doc2", "text_field_1": "Test document two about artificial intelligence"},
                    {"_id": "doc3", "text_field_1": "Test document three about artificial intelligence"},
                    {"_id": "doc4", "text_field_1": "Test document four about neural networks"},
                    {"_id": "doc5", "text_field_1": "Test document five about machine learning"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with excludeInputDocuments=True
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 1.0, "doc2": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=True
                        )
                    )
                )

                # Perform search
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=5
                )

                # Verify input documents are excluded
                result_ids = [hit["_id"] for hit in results["hits"]]
                self.assertNotIn("doc1", result_ids)
                self.assertNotIn("doc2", result_ids)

    def test_search_with_context_documents_include_input_succeeds(self):
        """Test that search with context documents includes input documents when requested."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "Test document one about artificial intelligence"},
                    {"_id": "doc2", "text_field_1": "Test document two about artificial intelligence"},
                    {"_id": "doc3", "text_field_1": "Test document three about artificial intelligence"},
                    {"_id": "doc4", "text_field_1": "Test document four about neural networks"},
                    {"_id": "doc5", "text_field_1": "Test document five about machine learning"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with excludeInputDocuments=False
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 1.0, "doc2": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Perform search
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=5
                )

                # Verify input documents are included
                result_ids = [hit["_id"] for hit in results["hits"]]
                self.assertTrue({"doc1", "doc2"}.issubset(set(result_ids)))

    def test_search_with_context_documents_filter(self):
        """Test that search with context documents respects filter parameter."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index using the helper method
                self._populate_index_orchids(index)

                # Create search context
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"orchid1": 1.0, "orchid2": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=True
                        )
                    )
                )

                # Search with filter for orchids only
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    filter='tags:(orchid)',
                    result_count=5
                )

                # Verify only orchid docs are returned (1 and 2 removed)
                result_ids = [hit["_id"] for hit in results["hits"]]
                self.assertEqual(set(result_ids), {"orchid3"})

                # Search with filter for continents
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    filter='tags:(continent)',
                    result_count=5
                )

                # Verify only continent docs are returned
                result_ids = [hit["_id"] for hit in results["hits"]]
                self.assertTrue(all(id.startswith("continent") for id in result_ids))
                self.assertEqual(len(result_ids), 3)

    def test_search_with_context_documents_score_modifiers(self):
        """Test that search with context documents works with score modifiers."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "Machine learning", "score_field": 10},
                    {"_id": "doc2", "text_field_1": "Deep learning", "score_field": 5},
                    {"_id": "doc3", "text_field_1": "Neural networks", "score_field": 1}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 1.0, "doc2": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Create score modifiers
                score_modifiers = ScoreModifierLists(
                    multiply_score_by=[ScoreModifierOperator(field_name="score_field", weight=1.0)]
                )

                # Perform search with score modifiers
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=3,
                    score_modifiers=score_modifiers
                )

                # Verify results are ordered by score_field
                result_ids = [hit["_id"] for hit in results["hits"]]
                # Check if doc1 comes before doc2 due to higher score_field value
                self.assertIn("doc1", result_ids)
                self.assertIn("doc2", result_ids)
                self.assertTrue(result_ids.index("doc1") < result_ids.index("doc2"))

    def test_search_with_context_documents_rerank_depth(self):
        """Test that search with context documents honors rerank_depth parameter."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            # Note: excluding legacy_unstructured_default_text_index as it was created with 
            # Marqo 2.12.0 which doesn't support rerank_depth (requires 2.15.0+)
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": f"doc_{i}", "text_field_1": f"Test document {i}"} for i in range(10)
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc_0": 1.0, "doc_1": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Case 1: result_count < rerank_depth → limit is respected
                with self.subTest(case="result_count_less_than_rerank_depth"):
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=3,
                        rerank_depth=5
                    )
                    self.assertEqual(len(results["hits"]), 3)

                # Case 2: offset beyond rerank_depth → results still present
                with self.subTest(case="offset_beyond_rerank_depth"):
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=1,
                        offset=3,
                        rerank_depth=2
                    )
                    self.assertEqual(len(results["hits"]), 1)

                # Case 3: result_count > rerank_depth → limit overrides rerank_depth
                with self.subTest(case="result_count_exceeds_rerank_depth"):
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        rerank_depth=3
                    )
                    self.assertEqual(len(results["hits"]), 5)

    def test_search_with_context_documents_gets_only_specific_fields_from_vespa(self):
        """Test that search with context documents only requests specific tensor field embeddings from Vespa.
        
        This test verifies the optimization where only the required embedding fields are fetched
        from Vespa when specific tensor fields are specified in context documents.
        """
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents with multiple tensor fields
                docs = [
                    {
                        "_id": "doc1", 
                        "text_field_1": "Machine learning algorithms for classification tasks",
                        "text_field_2": "Deep neural networks and computer vision"
                    },
                    {
                        "_id": "doc2", 
                        "text_field_1": "Natural language processing and text analysis",
                        "text_field_2": "Transformer models and attention mechanisms"
                    },
                    {
                        "_id": "doc3", 
                        "text_field_1": "Reinforcement learning for game playing",
                        "text_field_2": "Policy gradient methods and Q-learning"
                    },
                    {
                        "_id": "doc4", 
                        "text_field_1": "Statistical modeling and data science",
                        "text_field_2": "Bayesian inference and probabilistic models"
                    }
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1", "text_field_2"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Import the correct constants based on index type
                if isinstance(index, StructuredMarqoIndex):
                    from marqo.core.structured_vespa_index import common as structured_common
                    expected_id_field = structured_common.FIELD_ID
                    expected_embedding_field_1 = "marqo__embeddings_text_field_1"
                    expected_embedding_field_2 = "marqo__embeddings_text_field_2"
                else:
                    from marqo.core.unstructured_vespa_index import common as unstructured_common
                    expected_id_field = unstructured_common.VESPA_FIELD_ID
                    expected_embedding_field_1 = unstructured_common.VESPA_DOC_EMBEDDINGS
                    expected_embedding_field_2 = unstructured_common.VESPA_DOC_EMBEDDINGS

                # Store the original get_batch method
                original_get_batch = self.config.vespa_client.get_batch
                vespa_requests = []

                def mock_get_batch(*args, **kwargs):
                    # Capture the arguments passed to get_batch
                    vespa_requests.append({
                        'args': args,
                        'kwargs': kwargs
                    })
                    return original_get_batch(*args, **kwargs)

                # Mock the vespa client to capture requests
                with mock.patch.object(self.config.vespa_client, 'get_batch', side_effect=mock_get_batch):
                    # Test 1: Search with only text_field_1 specified
                    with self.subTest(tensor_field="text_field_1_only"):
                        vespa_requests.clear()
                        
                        search_context = SearchContext(
                            documents=SearchContextDocuments(
                                ids={"doc1": 1.0, "doc2": 2.0},
                                parameters=SearchContextDocumentsParameters(
                                    tensorFields=["text_field_1"],  # Only specify text_field_1
                                    excludeInputDocuments=False
                                )
                            )
                        )

                        # Perform search with context documents
                        results = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text=None,
                            context=search_context,
                            result_count=5
                        )

                        # Verify that get_batch was called
                        self.assertGreater(len(vespa_requests), 0, "get_batch should have been called")
                        
                        # Check the fields requested from Vespa
                        get_batch_call = vespa_requests[0]
                        requested_fields = get_batch_call['kwargs'].get('fields', [])
                        
                        # Verify that only the required fields were requested
                        if isinstance(index, StructuredMarqoIndex):
                            # For structured index, should request ID + text_field_1 embedding field
                            self.assertIn(expected_id_field, requested_fields)
                            self.assertIn(expected_embedding_field_1, requested_fields)
                            # Should NOT include text_field_2 embedding field
                            self.assertNotIn(expected_embedding_field_2, requested_fields)
                        else:
                            # For unstructured index, should request ID + embeddings field
                            self.assertIn(expected_id_field, requested_fields)
                            self.assertIn(expected_embedding_field_1, requested_fields)

                    # Test 2: Directly test get_doc_vectors_per_tensor_field_by_ids to verify it only returns specified fields
                    with self.subTest(tensor_field="direct_vector_retrieval"):
                        vespa_requests.clear()
                        
                        # Import the function for direct testing
                        from marqo.tensor_search.tensor_search import get_doc_vectors_per_tensor_field_by_ids
                        
                        # Call the function directly with specific tensor fields
                        doc_vectors = get_doc_vectors_per_tensor_field_by_ids(
                            config=self.config,
                            index_name=index.name,
                            document_ids=["doc1", "doc2"],
                            tensor_fields=["text_field_1"]  # Only request text_field_1
                        )

                        # Verify the structure of returned vectors
                        self.assertIn("doc1", doc_vectors)
                        self.assertIn("doc2", doc_vectors)
                        
                        for doc_id in ["doc1", "doc2"]:
                            doc_embeddings = doc_vectors[doc_id]
                            # Should only contain text_field_1 embeddings
                            if isinstance(index, StructuredMarqoIndex):
                                self.assertIn("text_field_1", doc_embeddings)
                                # Should NOT contain text_field_2 embeddings for structured indices
                                self.assertNotIn("text_field_2", doc_embeddings)
                            else:
                                # For legacy unstructured, the field name is "marqo__embeddings"
                                # and it contains all embeddings (can't separate them)
                                self.assertIn("marqo__embeddings", doc_embeddings)
                                # Verify embeddings are not empty
                                self.assertGreater(len(doc_embeddings["marqo__embeddings"]), 0)

                        # Check that get_batch was called with correct fields
                        self.assertGreater(len(vespa_requests), 0, "get_batch should have been called")
                        get_batch_call = vespa_requests[0]
                        requested_fields = get_batch_call['kwargs'].get('fields', [])
                        
                        if isinstance(index, StructuredMarqoIndex):
                            self.assertIn(expected_embedding_field_1, requested_fields)
                            self.assertNotIn(expected_embedding_field_2, requested_fields)

                    # Test 3: Compare with full document retrieval to verify correctness
                    with self.subTest(tensor_field="comparison_with_full_retrieval"):
                        # Get full document with all vectors
                        full_doc_response = tensor_search.get_document_by_id(
                            config=self.config,
                            index_name=index.name,
                            document_id="doc1",
                            show_vectors=True
                        )

                        # Extract tensor facets for text_field_1 from full document
                        full_doc_text_field_1_vectors = []
                        if '_tensor_facets' in full_doc_response:
                            for facet in full_doc_response['_tensor_facets']:
                                if 'text_field_1' in facet:
                                    full_doc_text_field_1_vectors.append(facet['_embedding'])

                        # Get vectors using optimized method
                        optimized_vectors = get_doc_vectors_per_tensor_field_by_ids(
                            config=self.config,
                            index_name=index.name,
                            document_ids=["doc1"],
                            tensor_fields=["text_field_1"]
                        )

                        # Compare the vectors
                        if "doc1" in optimized_vectors and "text_field_1" in optimized_vectors["doc1"]:
                            optimized_text_field_1_vectors = optimized_vectors["doc1"]["text_field_1"]
                            
                            # Verify that the vectors match
                            self.assertEqual(len(full_doc_text_field_1_vectors), len(optimized_text_field_1_vectors))
                            for i, (full_vector, optimized_vector) in enumerate(zip(full_doc_text_field_1_vectors, optimized_text_field_1_vectors)):
                                self.assertEqual(len(full_vector), len(optimized_vector))
                                # Compare vectors with some tolerance for floating point precision
                                for j, (full_val, opt_val) in enumerate(zip(full_vector, optimized_vector)):
                                    self.assertAlmostEqual(full_val, opt_val, places=6, 
                                        msg=f"Vector mismatch at doc1, vector {i}, dimension {j}")

                    # Test 4: Test with multiple tensor fields specified
                    with self.subTest(tensor_field="multiple_fields"):
                        vespa_requests.clear()

                        doc_vectors_multi = get_doc_vectors_per_tensor_field_by_ids(
                            config=self.config,
                            index_name=index.name,
                            document_ids=["doc1"],
                            tensor_fields=["text_field_1", "text_field_2"]  # Request both fields
                        )

                        # Verify both fields are present
                        self.assertIn("doc1", doc_vectors_multi)
                        doc_embeddings = doc_vectors_multi["doc1"]
                        if isinstance(index, StructuredMarqoIndex):
                            self.assertIn("text_field_1", doc_embeddings)
                            self.assertIn("text_field_2", doc_embeddings)
                        else:
                            # For legacy unstructured, all embeddings are returned as "marqo__embeddings"
                            # regardless of which specific fields were requested
                            self.assertIn("marqo__embeddings", doc_embeddings)
                            # Should contain embeddings from both fields combined
                            self.assertGreater(len(doc_embeddings["marqo__embeddings"]), 0)

                    # Test 5: Test with no tensor fields specified (should get all)
                    with self.subTest(tensor_field="all_fields"):
                        vespa_requests.clear()
                        
                        doc_vectors_all = get_doc_vectors_per_tensor_field_by_ids(
                            config=self.config,
                            index_name=index.name,
                            document_ids=["doc1"],
                            tensor_fields=None  # No specific fields - should get all
                        )

                        # Verify all available tensor fields are present
                        self.assertIn("doc1", doc_vectors_all)
                        doc_embeddings = doc_vectors_all["doc1"]
                        
                        if isinstance(index, StructuredMarqoIndex):
                            # For structured index, should have both text fields
                            self.assertIn("text_field_1", doc_embeddings)
                            self.assertIn("text_field_2", doc_embeddings)
                            
                            # Check that get_batch was called with all embedding fields
                            get_batch_call = vespa_requests[0]
                            requested_fields = get_batch_call['kwargs'].get('fields', [])
                            self.assertIn(expected_embedding_field_1, requested_fields)
                            self.assertIn(expected_embedding_field_2, requested_fields)
                        else:
                            # For unstructured index, fields are not separated in the return
                            # But we should still have embeddings
                            self.assertGreater(len(doc_embeddings), 0)

    def test_search_with_context_documents_max_search_context_docs_env_var(self):
        """Test that MARQO_MAX_SEARCH_CONTEXT_DOCS environment variable controls the limit for context documents."""

        
        index = self.structured_default_text_index
        
        # Add documents to the index
        docs = [
            {"_id": f"doc_{i}", "text_field_1": f"Test document {i}"} 
            for i in range(15)  # Create more than default limit (10)
        ]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs,
                tensor_fields=None
            )
        )

        # Test 1: Default limit (should be 10) - try with 11 documents, should fail
        with self.subTest("default_limit_exceeded"):
            search_context = SearchContext(
                documents=SearchContextDocuments(
                    ids={f"doc_{i}": 1.0 for i in range(11)},  # 11 documents > default limit of 10
                    parameters=SearchContextDocumentsParameters(
                        tensorFields=["text_field_1"],
                        excludeInputDocuments=False
                    )
                )
            )

            with self.assertRaises(IllegalRequestedDocCount) as cm:
                tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=5
                )
            
            # Verify the error message mentions the correct limit and env var
            error_message = str(cm.exception.message)
            self.assertIn("Search context documents limit exceeded", error_message)
            self.assertIn("Maximum allowed is 10", error_message)
            self.assertIn("but got 11", error_message)
            self.assertIn(EnvVars.MARQO_MAX_SEARCH_CONTEXT_DOCS, error_message)

        # Test 2: Set environment variable to 11, same search should now pass
        with self.subTest("increased_limit_passes"):
            with mock.patch.dict(os.environ, {EnvVars.MARQO_MAX_SEARCH_CONTEXT_DOCS: "11"}):
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={f"doc_{i}": 1.0 for i in range(11)},  # 11 documents = new limit of 11
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # This should now pass without raising an exception
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=5
                )
                
                # Verify search was successful
                self.assertIn("hits", results)
                self.assertGreater(len(results["hits"]), 0)

        # Test 3: Even with increased limit, exceeding it should still fail
        with self.subTest("increased_limit_still_enforced"):
            with mock.patch.dict(os.environ, {EnvVars.MARQO_MAX_SEARCH_CONTEXT_DOCS: "11"}):
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={f"doc_{i}": 1.0 for i in range(12)},  # 12 documents > new limit of 11
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                with self.assertRaises(IllegalRequestedDocCount) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5
                    )
                
                # Verify the error message reflects the new limit
                error_message = str(cm.exception.message)
                self.assertIn("Search context documents limit exceeded", error_message)
                self.assertIn("Maximum allowed is 11", error_message)
                self.assertIn("but got 12", error_message)

    def test_search_with_context_documents_fails_for_legacy_unstructured_index(self):
        """Test that search with context documents fails for legacy unstructured index."""
        docs = self._populate_index_orchids(self.legacy_unstructured_default_text_index)
        
        context_documents = SearchContextDocuments(
            ids={"orchid1": 1.0, "orchid2": 1.0}
        )
        
        with self.assertRaises(UnsupportedFeatureError) as cm:
            tensor_search.search(
                config=self.config,
                index_name=self.legacy_unstructured_default_text_index.name,
                text=None,
                context=SearchContext(documents=context_documents)
            )
        
        self.assertIn("Search context is not supported for unstructured indexes created with Marqo version", str(cm.exception))

    def test_context_tensors_cancel_out_zero_magnitude_error(self):
        """Test that context tensors that cancel each other out raise ZeroMagnitudeVectorError with NLERP."""
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                # Create two identical tensors with opposite weights that will cancel out
                # This should result in a zero magnitude vector when using NLERP
                context_tensors = [
                    SearchContextTensor(vector=[1.0] * self.DIMENSION, weight=1.0),
                    SearchContextTensor(vector=[1.0] * self.DIMENSION, weight=-1.0)
                ]
                
                with self.assertRaises(ZeroMagnitudeVectorError) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=index_name,
                        text=None,
                        context=SearchContext(tensor=context_tensors),
                        interpolation_method=InterpolationMethod.NLERP
                    )
                
                self.assertIn("zero magnitude", str(cm.exception))

    def test_context_tensors_all_zero_weights_error(self):
        """Test that context tensors with all zero weights raise AllZeroWeightsError."""
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
            with self.subTest(msg=index_name):
                # Create context tensors with all zero weights
                context_tensors = [
                    SearchContextTensor(vector=[1.0] * self.DIMENSION, weight=0.0),
                    SearchContextTensor(vector=[2.0] * self.DIMENSION, weight=0.0)
                ]
                
                with self.assertRaises(AllZeroWeightsError) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=index_name,
                        text=None,
                        context=SearchContext(tensor=context_tensors)
                    )
                
                self.assertIn("All weights are zero", str(cm.exception))

    def test_context_documents_all_zero_weights_error(self):
        """Test that context documents with all zero weights raise InvalidArgumentError."""
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(type=index.type):
                # Populate the index with test documents
                docs = self._populate_index_orchids(index)
                
                # Create context documents with all zero weights
                context_documents = SearchContextDocuments(
                    ids={"orchid1": 0.0, "orchid2": 0.0}
                )
                
                with self.assertRaises(InvalidArgumentError) as cm:
                    tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=SearchContext(documents=context_documents),
                        result_count=3
                    )
                self.assertIn("No documents with non-zero weight provided", str(cm.exception))

    def test_context_documents_uses_different_tensor_fields(self):
        """Test that context documents produce different results when using different tensor fields.
        
        This test demonstrates that specifying different tensor fields in context search
        produces different results, proving that each field is being used independently.
        """
        # This test only works with structured index since it has defined tensor fields
        index = self.structured_default_text_index
        
        # Create documents with clear differences between tensor fields
        docs = [
            {
                "_id": "doc1",
                "text_field_1": "shirts clothing apparel",  # General shirts
                "text_field_2": "red color bright vibrant",  # Red color
                "non_vector_text_field": "Document 1",
                "score_field": 1.0,
            },
            {
                "_id": "doc2", 
                "text_field_1": "pants trousers clothing",  # Different clothing type
                "text_field_2": "black color dark clothing",  # Black color
                "non_vector_text_field": "Document 2",
                "score_field": 2.0,
            },
            {
                "_id": "context_doc",
                "text_field_1": "shirts apparel fashion",  # Similar to doc1 field1
                "text_field_2": "black dark clothing",  # Similar to doc2 field2
                "non_vector_text_field": "Context document",
                "score_field": 3.0,
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs
            )
        )
        
        # Test 1: Search using only text_field_1 in context
        # Should favor doc1 since context has "shirts" similar to doc1's "shirts"
        results_field1 = tensor_search.search(
            config=self.config,
            index_name=index.name,
            text=None,
            context=SearchContext(
                documents=SearchContextDocuments(
                    ids={"context_doc": 1.0},
                    parameters=SearchContextDocumentsParameters(
                        tensorFields=["text_field_1"],
                        excludeInputDocuments=True
                    )
                )
            ),
            result_count=2
        )
        
        # Test 2: Search using only text_field_2 in context  
        # Should favor doc2 since context has "black" similar to doc2's "black"
        results_field2 = tensor_search.search(
            config=self.config,
            index_name=index.name,
            text=None,
            context=SearchContext(
                documents=SearchContextDocuments(
                    ids={"context_doc": 1.0},
                    parameters=SearchContextDocumentsParameters(
                        tensorFields=["text_field_2"],
                        excludeInputDocuments=True
                    )
                )
            ),
            result_count=2
        )
        
        # Extract results
        field1_ids = [hit["_id"] for hit in results_field1["hits"]]
        field2_ids = [hit["_id"] for hit in results_field2["hits"]]
        
        # The key assertion: Different tensor fields should produce different top results
        self.assertEqual(field1_ids[0], "doc1", 
                        f"text_field_1 should favor doc1 due to 'shirts' similarity. Got: {field1_ids}")
        self.assertEqual(field2_ids[0], "doc2",
                        f"text_field_2 should favor doc2 due to 'black' similarity. Got: {field2_ids}")

    def test_hybrid_search_with_context_documents(self):
        """
        Integration test for all valid combinations of retrieval and ranking methods for hybrid search with context documents.
        Shows the search runs without errors and correctly retrieves the expected document.
        
        Tests for invalid methods (lexical/lexical) are in API and unit tests instead of here.
        """
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add test documents
                docs = [
                    {
                        "text_field_1": "A comparison of the best pets",
                        "text_field_2": "Animals",
                        "_id": "d1"
                    },
                    {
                        "text_field_1": "The history of dogs",
                        "text_field_2": "A history of household pets",
                        "_id": "d2"
                    }
                ]
                
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1", "text_field_2"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                test_cases = [
                    ("disjunction", "rrf", {"best pets": 1}, "animals"),
                    ("tensor", "tensor", {"best pets": 1}, None),
                    ("tensor", "lexical", {"best pets": 1}, "animals"),
                    ("lexical", "tensor", {"best pets": 1}, "animals")
                ]

                for interpolation_method in [InterpolationMethod.LERP, InterpolationMethod.NLERP, InterpolationMethod.SLERP]:
                    context = SearchContext(
                        documents=SearchContextDocuments(
                            parameters=SearchContextDocumentsParameters(
                                excludeInputDocuments=False,
                                tensorFields=["text_field_1", "text_field_2"]
                            ),
                            ids={"d1": 1}
                        )
                    )

                    for retrieval_method, ranking_method, query_tensor, query_lexical in test_cases:
                        with self.subTest(
                            retrieval_method=retrieval_method, 
                            ranking_method=ranking_method,
                            interpolation_method=interpolation_method.value
                        ):
                            # Convert string values to enum values for HybridParameters
                            retrieval_enum = RetrievalMethod(retrieval_method)
                            ranking_enum = RankingMethod(ranking_method)
                            
                            hybrid_parameters = HybridParameters(
                                queryTensor=query_tensor,
                                queryLexical=query_lexical,
                                retrievalMethod=retrieval_enum,
                                rankingMethod=ranking_enum
                            )
                            
                            res = tensor_search.search(
                                config=self.config,
                                index_name=index.name,
                                text=None,
                                context=context,
                                search_method="HYBRID",
                                hybrid_parameters=hybrid_parameters,
                                interpolation_method=interpolation_method
                            )
                            
                            # Verify the expected document is returned as the top result
                            self.assertEqual(res["hits"][0]["_id"], "d1")

    def test_search_with_context_documents_allow_missing_documents_true(self):
        """Test that search with context documents succeeds when allowMissingDocuments=True and some documents don't exist."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = HybridParameters(retrievalMethod="tensor", rankingMethod="tensor") \
                        if search_method == "HYBRID" else None
                with self.subTest(f"index={index.type}, search_method={search_method}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_1": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            ids={"doc1": 2.0, "non_existent_doc": 1.0, "doc2": 1.5, "another_missing_doc": 0.5},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=False,
                                allowMissingDocuments=True
                            )
                        )
                    )

                    # Should succeed and only use existing documents (doc1 and doc2)
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    # Verify search was successful
                    self.assertIn("hits", results)
                    self.assertGreater(len(results["hits"]), 0)

                    # Verify that existing documents are still included in results
                    result_ids = [hit["_id"] for hit in results["hits"]]
                    self.assertIn("doc1", result_ids)
                    self.assertIn("doc2", result_ids)

    def test_search_with_context_documents_allow_missing_embeddings_true(self):
        """Test that search with context documents succeeds when allowMissingEmbeddings=True and some documents lack required embeddings."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = HybridParameters(retrievalMethod="tensor", rankingMethod="tensor") \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index={index.type}, search_method={search_method}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1", "text_field_2"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc1": 2.0, "doc3": 0.1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=False,
                                allowMissingEmbeddings=True
                            )
                        )
                    )

                    # Should succeed and only use existing documents (doc1 and doc2)
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    # Verify search was successful
                    self.assertIn("hits", results)
                    self.assertGreater(len(results["hits"]), 0)

                    # Verify that existing documents are still included in results
                    result_ids = set([hit["_id"] for hit in results["hits"]])
                    self.assertEqual({"doc1", "doc2", "doc3"}, result_ids)

    def test_search_with_context_documents_allow_missing_both_parameters(self):
        """Test that search works when both allowMissingDocuments=True and allowMissingEmbeddings=True with mixed scenarios."""
        # This test only works with structured index as it has defined tensor fields
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = HybridParameters(retrievalMethod="tensor", rankingMethod="tensor") \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index={index.type}, search_method={search_method}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc1": 2.0, "doc3": 0.1, "not_exists_doc": 1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=True,
                                allowMissingEmbeddings=True,
                                allowMissingDocuments=True
                            )
                        )
                    )

                    # Should succeed and only use existing documents (doc1 and doc2)
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    # Verify search was successful
                    self.assertIn("hits", results)
                    self.assertGreater(len(results["hits"]), 0)
                    result_ids = set([hit["_id"] for hit in results["hits"]])
                    self.assertEqual({"doc2"}, result_ids)

    def test_a_proper_error_is_raised_if_marqo_can_not_collect_any_vector(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = HybridParameters(retrievalMethod="tensor", rankingMethod="tensor") \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index={index.type}, search_method={search_method}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc3": 0.1, "not_exists_doc": 1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=True,
                                allowMissingEmbeddings=True,
                                allowMissingDocuments=True
                            )
                        )
                    )

                    # Should succeed and only use existing documents (doc1 and doc2)
                    with self.assertRaises(InvalidArgError) as e:
                        results = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text=None,
                            context=search_context,
                            result_count=5,
                            search_method=search_method,
                            hybrid_parameters=hybrid_parameters
                        )

                    self.assertIn("Marqo could not collect any vectors from the search query", str(e.exception))

    def test_search_with_context_documents_allow_missing_both_parameters_for_different_hybrid_parameters(self):
        """Test that search works when both allowMissingDocuments=True and allowMissingEmbeddings=True with mixed scenarios,
        with different hybrid parameters"""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            hybrid_parameters_test_cases = (
                {"queryLexical": "*", "queryTensor": None, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryLexical": "*", "queryTensor": {}, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryTensor": None, "retrievalMethod": "tensor", "rankingMethod": "tensor"},
                {"queryTensor": {}, "retrievalMethod": "tensor", "rankingMethod": "tensor"}
            )
            for hybrid_parameters in hybrid_parameters_test_cases:
                with self.subTest(f"index={index.type}, hybrid_parameters={hybrid_parameters_test_cases}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc1": 2.0, "doc3": 0.1, "not_exists_doc": 1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=True,
                                allowMissingEmbeddings=True,
                                allowMissingDocuments=True
                            )
                        )
                    )

                    # Should succeed and only use existing documents (doc1 and doc2)
                    results = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(**hybrid_parameters)
                    )

                    # Verify search was successful
                    self.assertIn("hits", results)
                    self.assertGreater(len(results["hits"]), 0)
                    result_ids = set([hit["_id"] for hit in results["hits"]])
                    self.assertEqual({"doc2"}, result_ids)

    def test_search_with_context_documents_raise_vector_collect_errors_with_disjunction(self):
        """Test that search works when both allowMissingDocuments=True and allowMissingEmbeddings=True with mixed scenarios,
        with different hybrid parameters"""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            hybrid_parameters_test_cases = (
                {"queryLexical": "*", "queryTensor": None, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryLexical": "*", "queryTensor": {}, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryTensor": None, "retrievalMethod": "tensor", "rankingMethod": "tensor"},
                {"queryTensor": {}, "retrievalMethod": "tensor", "rankingMethod": "tensor"}
            )
            for hybrid_parameters in hybrid_parameters_test_cases:
                with self.subTest(f"index={index.type}, hybrid_parameters={hybrid_parameters_test_cases}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc3": 0.1, "not_exists_doc": 1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=True,
                                allowMissingEmbeddings=True,
                                allowMissingDocuments=True
                            )
                        )
                    )

                    with self.assertRaises(InvalidArgError) as e:
                        _ = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text=None,
                            context=search_context,
                            result_count=5,
                            search_method="HYBRID",
                            hybrid_parameters=HybridParameters(**hybrid_parameters)
                        )
                    self.assertIn("Marqo could not collect any vectors from the search query", str(e.exception))
                    self.assertIn("Please check the provided query, context (if any), "
                                  "or queryTensor(for Hybrid search)", str(e.exception))

    def test_search_with_context_if_context_vectors_exist_documents_can_have_no_embeddings(self):
        """Test that if context vector is provided, even if marqo can not collect any vectors from the documents,
        the search can still proceed when allow_missing_documents=True, allowing_missing_embeddings=True"""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            hybrid_parameters_test_cases = (
                {"queryLexical": "*", "queryTensor": None, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryLexical": "*", "queryTensor": {}, "retrievalMethod": "disjunction", "rankingMethod": "rrf"},
                {"queryTensor": None, "retrievalMethod": "tensor", "rankingMethod": "tensor"},
                {"queryTensor": {}, "retrievalMethod": "tensor", "rankingMethod": "tensor"}
            )
            for hybrid_parameters in hybrid_parameters_test_cases:
                with self.subTest(f"index={index.type}, hybrid_parameters={hybrid_parameters_test_cases}"):
                    # Add some documents
                    docs = [
                        {"_id": "doc1", "text_field_1": "Machine learning and artificial intelligence"},
                        {"_id": "doc2", "text_field_1": "Deep learning neural networks"},
                        {"_id": "doc3", "text_field_2": "Natural language processing"}
                    ]

                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=docs,
                            tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    # Create search context with mix of existing and non-existent documents
                    search_context = SearchContext(
                        documents=SearchContextDocuments(
                            # doc3 does not have "text_field_1"
                            ids={"doc3": 0.1, "not_exists_doc": 1},
                            parameters=SearchContextDocumentsParameters(
                                tensorFields=["text_field_1"],
                                excludeInputDocuments=True,
                                allowMissingEmbeddings=True,
                                allowMissingDocuments=True
                            )
                        ),
                        tensor=[SearchContextTensor(vector=[0.01] * 384, weight=1)]
                    )

                    r = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5,
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(**hybrid_parameters)
                    )

                    self.assertEqual(2, len(r["hits"]))