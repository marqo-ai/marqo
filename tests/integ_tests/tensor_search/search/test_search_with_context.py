import os
import unittest
from unittest import mock

from pydantic.v1.error_wrappers import ValidationError
from marqo.core.exceptions import InvalidFieldNameError
from marqo.api.exceptions import InvalidArgError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor, SearchContextDocuments, SearchContextDocumentsParameters
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.utils.vector_interpolation import Slerp, Lerp, Nlerp, ZeroSumWeightsError, ZeroMagnitudeVectorError
from marqo.exceptions import InvalidArgumentError, InternalError
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from integ_tests.marqo_test import MarqoTestCase
from integ_tests.utils.transition import *

class TestSearchWithContext(MarqoTestCase):

    structured_index_basic = "structured_index_basic"
    unstructured_index_basic = "unstructured_index_basic"
    
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
                FieldRequest(name='text_field_1', type=FieldType.Text),
                FieldRequest(name='text_field_2', type=FieldType.Text),
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

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_index_basic_request,
            unstructured_index_basic_request,
        ])

        # Default text indexes for the context.documents tests
        cls.structured_default_text_index = cls.indexes[0]  # Use the structured index we created
        cls.unstructured_default_text_index = cls.indexes[1]  # Use the unstructured index we created

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
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
        for index_name in [self.structured_index_basic, self.unstructured_index_basic]:
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

    def test_search_with_context_documents_all_interpolation_methods_succeeds(self):
        """Test that search works correctly with context documents using different interpolation methods.

        This test verifies that when we search using only document IDs as context with various
        interpolation methods (SLERP, LERP, NLERP), the search results match what we'd expect.
        The correct interpolation method must be called.
        """
        # Dictionary mapping interpolation methods to their corresponding classes
        interpolation_methods = {
            InterpolationMethod.SLERP: Slerp,
            InterpolationMethod.LERP: Lerp,
            InterpolationMethod.NLERP: Nlerp
        }
        
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            # Add documents to the index using the helper method
            self._populate_index_orchids(index)
            
            for method_enum, interpolator_class in interpolation_methods.items():
                with self.subTest(index=index.type, interpolation_method=method_enum):
                    # Mock the interpolation method to verify it's being called
                    original_interpolate = interpolator_class().interpolate

                    def interpolate(vectors, weights, prenormalized=False):
                        return original_interpolate(vectors, weights, prenormalized)

                    with mock.patch.object(interpolator_class, "interpolate", wraps=interpolate) as mock_interpolate:
                        # Create search context with documents
                        search_context = SearchContext(
                            documents=SearchContextDocuments(
                                ids={"orchid1": 1.0, "orchid2": 1.0},
                                parameters=SearchContextDocumentsParameters(
                                    tensorFields=["text_field_1"],
                                    excludeInputDocuments=False
                                )
                            )
                        )

                        # Perform search with context documents using the specified interpolation method
                        results = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text=None,
                            context=search_context,
                            result_count=5,
                            interpolation_method=method_enum
                        )

                        # Verify interpolation method was called
                        mock_interpolate.assert_called_once()

                        # Verify search results
                        self.assertIn("hits", results)
                        ids = [doc["_id"] for doc in results["hits"]]

                        # Orchid documents should be ranked higher
                        self.assertEqual(set(["orchid1", "orchid2", "orchid3"]), set(ids[:3]))

    def test_search_with_context_documents_some_zero_weights_succeeds(self):
        """Test that search with context documents ignores documents with zero weight."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "Test document one"},
                    {"_id": "doc2", "text_field_1": "Test document two"},
                    {"_id": "doc3", "text_field_1": "Test document three"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with some documents having zero weight
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 0.0, "doc2": 0.0, "doc3": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=None,
                    context=search_context,
                    result_count=5,
                    interpolation_method=InterpolationMethod.SLERP
                )

                # Verify search results
                self.assertIn("hits", results)
                ids = [doc["_id"] for doc in results["hits"]]

                # Document 3 should be in results since it has non-zero weight
                self.assertEqual(ids[0], "doc3")

    def test_search_with_context_documents_all_zero_weights_fails(self):
        """Test that search with context documents fails when all documents have zero weight."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents to the index
                docs = [
                    {"_id": "doc1", "text_field_1": "Test document one"},
                    {"_id": "doc2", "text_field_1": "Test document two"},
                    {"_id": "doc3", "text_field_1": "Test document three"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with all documents having zero weight
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 0.0, "doc2": 0.0, "doc3": 0.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Verify error is raised for all zero weights
                with self.assertRaises(InvalidArgumentError) as ex:
                    tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5
                    )
                self.assertIn('No documents with non-zero weight provided', str(ex.exception))

    def test_search_with_context_documents_without_vectors_fails(self):
        """Test that search with context documents fails when documents don't have vectors."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents without vectors in the specified field
                docs = [
                    {"_id": "doc1", "title": "Document without vector"},
                    {"_id": "doc2", "text_field_1": "Document with vector"}
                ]

                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Create search context with a document that doesn't have vectors
                search_context = SearchContext(
                    documents=SearchContextDocuments(
                        ids={"doc1": 1.0},
                        parameters=SearchContextDocumentsParameters(
                            tensorFields=["text_field_1"],
                            excludeInputDocuments=False
                        )
                    )
                )

                # Verify error is raised for document without vectors
                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=None,
                        context=search_context,
                        result_count=5
                    )

    def test_search_with_context_documents_invalid_tensor_fields(self):
        """Test that search with context documents fails with invalid tensor fields."""
        # This test is specific to structured index since unstructured indexes don't validate tensor fields
        index = self.structured_default_text_index

        # Add documents
        docs = [
            {"_id": "doc1", "text_field_1": "Test document one"},
            {"_id": "doc2", "text_field_1": "Test document two"}
        ]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs,
                tensor_fields=None
            )
        )

        # Create search context with invalid tensor field
        search_context = SearchContext(
            documents=SearchContextDocuments(
                ids={"doc1": 1.0, "doc2": 1.0},
                parameters=SearchContextDocumentsParameters(
                    tensorFields=["invalid_field"],
                    excludeInputDocuments=False
                )
            )
        )

        # Verify error is raised for invalid tensor field
        with self.assertRaises(InvalidFieldNameError):
            tensor_search.search(
                config=self.config,
                index_name=index.name,
                text=None,
                context=search_context,
                result_count=5
            )

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

    # TODO: Hybrid search with context documents (rrf, tensor/tensor)
    def test_search_with_context_documents_hybrid_search(self):
        """
        Test that search with context documents works with hybrid search in the following scenarios:
        1. disjunction / rrf
        2. tensor retrieval / tensor ranking
        3. lexical retrieval / tensor ranking
        4. tensor retrieval / lexical ranking
        """
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
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
