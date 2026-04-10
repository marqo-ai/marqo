import os
import unittest
from unittest import mock

from marqo.core.exceptions import InvalidFieldNameError
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_index import Model, FieldFeature, FieldType, MarqoIndex, TextPreProcessing, \
    TextSplitMethod, UnstructuredMarqoIndex
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.search.recommender import Recommender
from marqo.core.utils.vector_interpolation import Slerp, Nlerp, Lerp
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search import tensor_search
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from tests.integ_tests.marqo_test import MarqoTestCase
import pytest

from marqo.tensor_search import tensor_search, index_meta_cache
from marqo.core.models.marqo_index import IndexType


class TestRecommender(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Note text_preprocessing is set to create one vector per field, so that vector count is predictable
        # This is required for some tests

        unstructured_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            text_preprocessing=TextPreProcessing(
                split_length=1000,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            )
        )

        unstructured_text_index_nonnormalized = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'), normalize_embeddings=False,
            text_preprocessing=TextPreProcessing(
                split_length=1000,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
        )

        structured_text_index = cls.structured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="description", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="content", type=FieldType.Text,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="tags", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="multimodal_field", type=FieldType.MultimodalCombination,
                             dependent_fields={"title": 0.5, "description": 0.5})
            ],
            tensor_fields=["title", "description", "content",
                           "multimodal_field"],
            text_preprocessing=TextPreProcessing(
                split_length=1000,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
        )

        cls.indexes = cls.create_indexes(
            [
                unstructured_text_index,
                unstructured_text_index_nonnormalized,
                structured_text_index
            ]
        )

        cls.unstructured_text_index = cls.indexes[0]
        cls.unstructured_text_index_nonnormalized = cls.indexes[1]
        cls.structured_text_index = cls.indexes[2]

    def setUp(self) -> None:
        super().setUp()

        self.recommender = Recommender(self.vespa_client, self.index_management, self.config.inference)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def _populate_index(self, index: MarqoIndex):
        docs = [
            {
                "_id": "1",
                "title": "Anacamptis laxiflora",
                "description": "Anacamptis laxiflora (lax-flowered orchid, loose-flowered orchid, or green-winged "
                               "meadow orchid) is a species of orchid. It has a wide distribution in Europe and Asia "
                               "as far north as in Germany, and is found in wet meadows with alkaline soil. It grows "
                               "up to 60 cm high.[1] A. laxiflora is common in Normandy and Brittany (France), "
                               "but in the United Kingdom it is represented only on the Channel Islands, "
                               "where in Jersey it is called Jersey orchid and in Guernsey it is called Loose "
                               "Flowered orchid . Notable localities in the Channel Islands include Le Noir Pré "
                               "meadow in Jersey and several fields at Les Vicheries in Guernsey, where mass blooms of "
                               "these orchids can be observed from late May to early June.",
                "tags": ["flower", "orchid"],
            },
            {
                "_id": "2",
                "title": "Cephalanthera longifolia",
                "content": "Cephalanthera longifolia reaches on average 20–60 centimetres (7.9–23.6 in) in height "
                           "in typical conditions. This orchid has erect and glabrous multiple stems. The leaves "
                           "are dark green, long and narrowly tapering (hence the common name sword-leaved "
                           "helleborine).",
                "tags": ["flower"],
            },
            {
                "_id": "3",
                "title": "Europe",
                "description": "Europe is a continent located entirely in the Northern Hemisphere and mostly in the "
                               "Eastern Hemisphere. It comprises the westernmost part of Eurasia and is bordered by "
                               "the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean Sea "
                               "to the south, and Asia to the east. Europe is commonly considered to be separated "
                               "from Asia by the watershed of the Ural Mountains, the Ural River, the Caspian Sea, "
                               "the Greater Caucasus, the Black Sea, and the waterways of the Turkish Straits.",
                "tags": ["continent"],
            },
            {
                "_id": "4",
                "title": "Asia",
                "description": "Asia is Earth's largest and most populous continent, located primarily in the Eastern "
                               "and Northern Hemispheres. It shares the continental landmass of Eurasia with the "
                               "continent of Europe and the continental landmass of Afro-Eurasia with both Europe and "
                               "Africa. Asia covers an area of 44,579,000 square kilometres (17,212,000 sq mi), about "
                               "30% of Earth's total land area and 8.7% of the Earth's total surface area. The "
                               "continent, which has long been home to the majority of the human population, was the "
                               "site of many of the first civilizations.",
                "tags": ["continent"],
            },
            {
                "_id": "5",
                "title": "Africa",
                "description": "Africa is the world's second-largest and second-most populous continent, after Asia in "
                               "both cases. At about 30.3 million km2 (11.7 million square miles) including adjacent "
                               "islands, it covers 6% of Earth's total surface area and 20% of its land area. With "
                               "1.3 billion people as of 2018, it accounts for about 16% of the world's human population.",
            },
            {
                "_id": "6",
                "title": "Anacamptis morio subsp. longicornu",
                "description": "Anacamptis morio subsp. longicornu is a subspecies of Anacamptis morio. It is found in "
                               "the Mediterranean region, including Spain, France, Italy, Greece, Cyprus, Turkey, "
            }
        ]

        if isinstance(index, UnstructuredMarqoIndex):
            tensor_fields = ["title", "description", "content", "multimodal_field"]
        else:
            tensor_fields = None

        self.add_documents(
            self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs,
                tensor_fields=tensor_fields
            )
        )

    def test_recommend_slerp_success(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                original_slerp = Slerp().interpolate

                def interpolate(vectors, weights):
                    return original_slerp(vectors, weights)

                with mock.patch.object(Slerp, "interpolate", wraps=interpolate) as mock_interpolate:
                    res = self.recommender.recommend(
                        index_name=index.name,
                        documents=["1", "2"],
                        interpolation_method=InterpolationMethod.SLERP,
                        exclude_input_documents=False,
                    )

                    # Note aside from interpolate in recommend,
                    # search step also calls LERP interpolate once by default
                    mock_interpolate.assert_called_once()

                    ids = [doc["_id"] for doc in res["hits"]]

                    self.assertEqual(set(ids), {"1", "2", "6"})


    def test_recommend_nlerp_success(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                original_nlerp = Nlerp().interpolate

                def interpolate(vectors, weights):
                    return original_nlerp(vectors, weights)

                with mock.patch.object(Nlerp, "interpolate", wraps=interpolate) as mock_interpolate:
                    res = self.recommender.recommend(
                        index_name=index.name,
                        documents=["1", "2"],
                        interpolation_method=InterpolationMethod.NLERP,
                        exclude_input_documents=False,
                    )

                    # Note aside from interpolate in recommend,
                    # search step also calls NLERP interpolate once by default
                    self.assertEqual(mock_interpolate.call_count, 2)

                    ids = [doc["_id"] for doc in res["hits"]]

                    self.assertEqual(set(ids), {"1", "2", "6"})

    def test_recommend_nlerpZeroMagnitudeVector_failure(self):
        """
        Test that the recommender fails when the interpolated vector has zero magnitude with NLERP
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                original_nlerp = Nlerp().interpolate

                def interpolate(vectors, weights):
                    return original_nlerp(
                        [
                            [1, 0, 2, 0],
                            [-1, 0, -2, 0]
                        ],
                        [1, 1]
                    )

                with mock.patch.object(Nlerp, "interpolate", wraps=interpolate):
                    with self.assertRaisesStrict(InvalidArgumentError) as ex:
                        self.recommender.recommend(
                            index_name=index.name,
                            documents=["1", "2"],
                            interpolation_method=InterpolationMethod.NLERP,
                            exclude_input_documents=False,
                        )
                    self.assertIn('zero-magnitude vector', str(ex.exception))

    def test_recommend_lerp_success(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                original_lerp = Lerp().interpolate

                def interpolate(vectors, weights):
                    return original_lerp(vectors, weights)

                with mock.patch.object(Lerp, "interpolate", wraps=interpolate) as mock_interpolate:
                    res = self.recommender.recommend(
                        index_name=index.name,
                        documents=["1", "2"],
                        interpolation_method=InterpolationMethod.LERP,
                        exclude_input_documents=False,
                    )

                    # Recommend calls LERP interpolate twice (once internally, once in search step)
                    self.assertEqual(mock_interpolate.call_count, 2)

                    ids = [doc["_id"] for doc in res["hits"]]

                    self.assertEqual(set(ids), {"1", "2", "6"})

    def test_recommend_docsWithZeroWeight_success(self):
        """
        Test that the recommender ignores documents with zero weight
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                # This will fail unless zero weight docs are ignored, due to SLERP on zero-sum weights
                self.recommender.recommend(
                    index_name=index.name,
                    documents={"1": 0, "2": 0, "3": 1},
                    tensor_fields=['title'],
                    interpolation_method=InterpolationMethod.SLERP,
                    exclude_input_documents=False,
                )

    def test_recommend_allDocsZeroWeight_failure(self):
        """
        Test that the recommender fails when all documents have zero weight
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                with self.assertRaisesStrict(InvalidArgumentError) as ex:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents={"1": 0, "2": 0, "3": 0}
                    )
                self.assertIn('No documents with non-zero weight provided', str(ex.exception))

    def test_recommend_docsWithoutVectors_success(self):
        """
        Test that the recommender fails when one or more documents do not have embeddings
        """
        docs = [
            {
                "_id": "1",
                "title": "Anacamptis laxiflora",
            },
            {
                "_id": "2",
                "title": "Cephalanthera longifolia",
                "content": "Content"
            }
        ]
        index = self.unstructured_text_index
        self.add_documents(
            self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=docs,
                tensor_fields=['content']
            )
        )

        with self.assertRaisesStrict(InvalidArgumentError):
            self.recommender.recommend(
                index_name=index.name,
                documents=["1", "2"],
            )

    def test_recommend_structuredInvalidTensorFields_failure(self):
        """
        Test that the recommender fails when the tensor fields are invalid for a structured index
        """
        index = self.structured_text_index
        self._populate_index(index)

        with self.assertRaisesStrict(InvalidFieldNameError):
            self.recommender.recommend(
                index_name=index.name,
                documents=["1", "2"],
                tensor_fields=['title', 'invalid_field']
            )

    def test_recommend_unstructuredInvalidTensorFields_failure(self):
        """
        Test that the recommender fails when the tensor fields are invalid for an unstructured index (no vectors).
        """
        index = self.unstructured_text_index
        self._populate_index(index)

        with self.assertRaisesStrict(InvalidArgumentError):
            self.recommender.recommend(
                index_name=index.name,
                documents=["1", "2"],
                tensor_fields=['invalid_field']
            )

    def test_recommend_emptyTensorFields_success(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["1", "2"],
                    exclude_input_documents=False,
                )

                ids = [doc["_id"] for doc in res["hits"]]

                self.assertEqual(set(ids), {"1", "2", "6"})

    def test_recommend_missingDocuments_failure(self):
        """
        Test that the recommender fails when some documents are missing
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                with self.assertRaisesStrict(InvalidArgumentError):
                    self.recommender.recommend(
                        index_name=index.name,
                        documents=["100", "2"],
                    )

    def test_recommend_emptyDocuments_failure(self):
        """
        Test that the recommender fails when documents is empty
        """
        with self.assertRaisesStrict(InvalidArgumentError):
            self.recommender.recommend(
                index_name=self.unstructured_text_index.name,
                documents=None
            )

        with self.assertRaisesStrict(InvalidArgumentError):
            self.recommender.recommend(
                index_name=self.unstructured_text_index.name,
                documents=[]
            )

    def test_defaultInterpolationMethodNormalized_success(self):
        """
        Test that correct default SLERP is picked correctly for normalized indexes
        """
        index = self.unstructured_text_index
        self._populate_index(index)

        original_slerp = Slerp().interpolate

        def interpolate(vectors, weights):
            return original_slerp(vectors, weights)

        with mock.patch.object(Slerp, "interpolate", wraps=interpolate) as mock_interpolate:
            self.recommender.recommend(
                index_name=index.name,
                documents=["1", "2"],
            )
            # Note aside from interpolate in recommend,
            # search step also calls LERP interpolate once by default
            mock_interpolate.assert_called_once()

    def test_defaultInterpolationMethodNonNormalized_success(self):
        """
        Test that correct default LERP is picked for non-normalized indexes
        """
        index = self.unstructured_text_index_nonnormalized
        self._populate_index(index)

        original_lerp = Lerp().interpolate

        def interpolate(vectors, weights):
            return original_lerp(vectors, weights)

        with mock.patch.object(Lerp, "interpolate", wraps=interpolate) as mock_interpolate:
            self.recommender.recommend(
                index_name=index.name,
                documents=["1", "2"],
            )

            # Recommend calls LERP interpolate twice (once internally, once in search step)
            self.assertEqual(mock_interpolate.call_count, 2)

    def test_recommend_excludeInputDocuments_success(self):
        """
        Test that the recommender excludes input documents when requested
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                res = self.recommender.recommend(
                    index_name=index.name,
                    documents={"1": 1, "2": 1, "3": 0},
                    exclude_input_documents=True,
                )

                ids = set([doc["_id"] for doc in res["hits"]])

                self.assertFalse(any(doc in ids for doc in ["1", "2", "3"]))

    def test_recommend_includeInputDocuments_success(self):
        """
        Test that the recommender includes input documents when requested
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["1", "2"],
                    exclude_input_documents=False,
                )

                ids = [doc["_id"] for doc in res["hits"]]

                self.assertTrue({"1", "2"}.issubset(set(ids)))

    def test_recommend_filterWithoutExcludeInputDocs_success(self):
        """
        Test that the recommender uses the given filter and includes input documents
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["1", "2"],
                    filter='tags:(orchid)',  # only document matching is 1
                    exclude_input_documents=False,
                )

                ids = [doc["_id"] for doc in res["hits"]]

                self.assertEqual(ids, ['1'])

    def test_recommend_filterWithExcludeInputDocs_success(self):
        """
        Test that the recommender uses the given filter and excludes input documents
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["1", "2"],
                    filter='tags:(orchid)',  # only document matching is 1
                    exclude_input_documents=True,
                )

                ids = [doc["_id"] for doc in res["hits"]]

                self.assertEqual(ids, [])

    def test_recommend_searchCallValid_success(self):
        """
        Test that the recommender calls the search method with the correct arguments

        This test is a shortcut to avoid testing arguments that are passed to the search method unchanged and do not
        affect recommender logic directly.
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                with mock.patch.object(tensor_search, "search") as mock_search:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents=["1", "2"],
                        result_count=10,
                        offset=5,
                        highlights=True,
                        ef_search=10,
                        approximate=True,
                        searchable_attributes=['title'],
                        verbose=1,
                        reranker='bm25',
                        attributes_to_retrieve=['title'],
                        score_modifiers=ScoreModifierLists(
                            multiply_score_by=[ScoreModifierOperator(field_name="title", weight=1)]
                        )
                    )

                    mock_search.assert_called_once_with(
                        mock.ANY,
                        index.name,
                        text=None,
                        context=mock.ANY,
                        result_count=10,
                        offset=5,
                        highlights=True,
                        ef_search=10,
                        approximate=True,
                        searchable_attributes=['title'],
                        verbose=1,
                        reranker='bm25',
                        filter=mock.ANY,
                        attributes_to_retrieve=['title'],
                        score_modifiers=ScoreModifierLists(
                            multiply_score_by=[ScoreModifierOperator(field_name="title", weight=1)]
                        ),
                        processing_start=mock.ANY,
                        rerank_depth=None
                    )

                # Repeat with different values to ensure it didn't pass due to default values matching
                # the test case
                with mock.patch.object(tensor_search, "search") as mock_search:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents=["1", "2"],
                        result_count=20,
                        offset=10,
                        highlights=False,
                        ef_search=100,
                        approximate=False,
                        searchable_attributes=['title'],
                        verbose=2,
                        reranker='bm25',
                        attributes_to_retrieve=['title'],
                        score_modifiers=ScoreModifierLists(
                            multiply_score_by=[ScoreModifierOperator(field_name="title", weight=1)]
                        ),
                    )

                    mock_search.assert_called_once_with(
                        mock.ANY,
                        index.name,
                        text=None,
                        context=mock.ANY,
                        result_count=20,
                        offset=10,
                        highlights=False,
                        ef_search=100,
                        approximate=False,
                        searchable_attributes=['title'],
                        verbose=2,
                        reranker='bm25',
                        filter=mock.ANY,
                        attributes_to_retrieve=['title'],
                        score_modifiers=ScoreModifierLists(
                            multiply_score_by=[ScoreModifierOperator(field_name="title", weight=1)]
                        ),
                        processing_start=mock.ANY,
                        rerank_depth=None
                    )

    @pytest.mark.skip_for_multinode
    def test_recommend_rerank_depth_with_limit_and_offset(self):
        """
        Test that recommender honors rerank_depth and behaves correctly with result_count and offset.
        """

        docs = [
            {"_id": "doc_0", "title": "Project Overview",
             "content": "Summary of the project's goals and deliverables."},
            {"_id": "doc_1", "title": "Team Roles", "content": "Descriptions of each team member's responsibilities."},
            {"_id": "doc_2", "title": "Timeline", "content": "Key milestones and deadlines for the project."},
            {"_id": "doc_3", "title": "Budget Estimate", "content": "Projected costs and resource allocation."},
            {"_id": "doc_4", "title": "Tech Stack", "content": "Overview of technologies and tools being used."},
            {"_id": "doc_5", "title": "Risk Assessment", "content": "Potential risks and mitigation strategies."},
            {"_id": "doc_6", "title": "Client Feedback", "content": "Summary of feedback received from stakeholders."},
            {"_id": "doc_7", "title": "Testing Plan", "content": "Details on testing strategies and coverage."},
            {"_id": "doc_8", "title": "Deployment Guide",
             "content": "Steps and procedures for deploying the application."},
            {"_id": "doc_9", "title": "Post-Mortem",
             "content": "Analysis of what went well and areas for improvement."},
        ]

        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                tensor_fields = ["title", "content"] if isinstance(index, UnstructuredMarqoIndex) else None

                self.add_documents(
                    self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=docs, tensor_fields=tensor_fields
                    )
                )

                # Case 1: result_count < rerank_depth — should return result_count documents
                with self.subTest(case="result_count_less_than_rerank_depth"):
                    res = self.recommender.recommend(
                        index_name=index.name, documents=["doc_0", "doc_1"], result_count=3, rerank_depth=5
                    )
                    self.assertEqual(len(res["hits"]), 3)

                # Case 2: offset > rerank_depth — offset + limit is higher, result must be present
                with self.subTest(case="offset_beyond_rerank_depth"):
                    res = self.recommender.recommend(
                        index_name=index.name, documents=["doc_0", "doc_1"], result_count=1, offset=3, rerank_depth=2
                    )
                    self.assertEqual(len(res["hits"]), 1)

                # Case 3: result_count + offset < rerank_depth — enough reranked results to fulfill offset and count
                with self.subTest(case="offset_within_rerank_depth"):
                    res = self.recommender.recommend(
                        index_name=index.name, documents=["doc_0", "doc_1"], result_count=2, offset=2, rerank_depth=5
                    )
                    self.assertEqual(len(res["hits"]), 2)

                # Case 4: rerank_depth < result_count — offset + limit is higher so rerank_depth gets overwritten
                with self.subTest(case="result_count_greater_than_rerank_depth"):
                    res = self.recommender.recommend(
                        index_name=index.name, documents=["doc_0", "doc_1"], result_count=5, rerank_depth=3
                    )
                    self.assertEqual(len(res["hits"]), 5)

                # Case 5: ef_search < rerank_depth < result_count — ef_search overrides rerank_depth/limit
                with self.subTest(case="ef_search_limits_rerank_depth"):
                    res = self.recommender.recommend(
                        index_name=index.name, documents=["doc_0", "doc_1"], result_count=10, rerank_depth=5,
                        ef_search=3, searchable_attributes=['title']
                    )
                    # We only assert < 3 as the exact number of results varies across runs,
                    # but for one searchable attribute and one shard is capped at ef_search
                    self.assertTrue(len(res["hits"]) <= 3)

    def test_get_doc_vectors_from_ids_noDocumentsProvided_fails(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                with self.assertRaises(InvalidArgumentError):
                    self.recommender.get_doc_vectors_from_ids(index.name, [])

    def test_get_doc_vectors_from_ids_allZeroWeights_fails(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                documents = {"doc1": 0.0, "doc2": 0.0}
                with self.assertRaises(InvalidArgumentError):
                    self.recommender.get_doc_vectors_from_ids(index.name, documents)

    def test_get_doc_vectors_from_ids_invalidTensorField_fails(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                with mock.patch('marqo.tensor_search.index_meta_cache.get_index') as mock_get_index:
                    mock_index = mock.Mock()
                    mock_index.type = IndexType.Structured
                    mock_index.tensor_field_map = {"valid_field": {}}
                    mock_get_index.return_value = mock_index

                    with self.assertRaises(InvalidFieldNameError):
                        self.recommender.get_doc_vectors_from_ids(
                            index.name,
                            documents=["doc1"],
                            tensor_fields=["invalid_field"]
                        )

    def test_get_doc_vectors_from_ids_documentNotFound_fails(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                with mock.patch('marqo.tensor_search.index_meta_cache.get_index') as mock_get_index, \
                        mock.patch('marqo.tensor_search.tensor_search.get_documents_by_ids') as mock_get_docs:
                    mock_get_index.return_value = index
                    mock_get_docs.return_value.dict.return_value = {
                        "results": [{"_id": "doc1", "_found": False}]
                    }

                    with self.assertRaises(InvalidArgumentError):
                        self.recommender.get_doc_vectors_from_ids(index.name, documents=["doc1"])

    def test_get_doc_vectors_from_ids_documentWithoutEmbeddings_fails(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                with mock.patch('marqo.tensor_search.index_meta_cache.get_index') as mock_get_index, \
                        mock.patch('marqo.tensor_search.tensor_search.get_documents_by_ids') as mock_get_docs:
                    mock_get_index.return_value = index
                    mock_get_docs.return_value.dict.return_value = {
                        "results": [{"_id": "doc1", "_found": True, "_tensor_facets": []}]
                    }

                    with self.assertRaises(InvalidArgumentError):
                        self.recommender.get_doc_vectors_from_ids(index.name, documents=["doc1"])

    def test_get_doc_vectors_from_ids_success(self):
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(index_type=index.name):
                with mock.patch('marqo.tensor_search.index_meta_cache.get_index') as mock_get_index, \
                        mock.patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids') as mock_get_doc_vectors:
                    mock_get_index.return_value = index
                    mock_get_doc_vectors.return_value = {
                        "doc1": {
                            "field1": [[0.1, 0.2, 0.3]]
                        }
                    }

                    result = self.recommender.get_doc_vectors_from_ids(index.name, documents=["doc1"])
                    expected = {"doc1": [[0.1, 0.2, 0.3]]}
                    self.assertEqual(result, expected)

    def test_recommend_allow_missing_documents_true_success(self):
        """Test that allowMissingDocuments=True allows missing documents and only uses existing ones"""
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                # Should succeed with allowMissingDocuments=True and only use existing documents
                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["1", "non_existent_doc", "2", "another_missing_doc"],
                    allow_missing_documents=True,
                    exclude_input_documents=False
                )

                # Verify search was successful
                self.assertIn("hits", res)
                self.assertGreater(len(res["hits"]), 0)

                # Verify that existing documents are included in results
                result_ids = [hit["_id"] for hit in res["hits"]]
                self.assertIn("1", result_ids)
                self.assertIn("2", result_ids)

    def test_recommend_allow_missing_embeddings_true_success(self):
        """Test that allowMissingEmbeddings=True allows documents with missing embeddings"""
        # Create documents where some have embeddings for specific fields and others don't
        docs = [
            {
                "_id": "doc_with_title",
                "title": "Document with title embedding",
                "description": "Also has description"
            },
            {
                "_id": "doc_with_content", 
                "content": "Document with only content embedding",
                "tags": ["test"]
            },
            {
                "_id": "doc_mixed",
                "title": "Mixed document",
                "content": "Has both title and content"
            }
        ]

        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                # For unstructured, specify which fields to vectorize
                if isinstance(index, UnstructuredMarqoIndex):
                    tensor_fields = ["title", "description"]  # Don't vectorize content
                else:
                    tensor_fields = None

                self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )

                # Should succeed with allowMissingEmbeddings=True, using only docs with required embeddings
                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=["doc_with_title", "doc_with_content", "doc_mixed"],
                    tensor_fields=["title"],
                    allow_missing_embeddings=True,
                    exclude_input_documents=False
                )

                # Verify search was successful
                self.assertIn("hits", res)
                self.assertGreater(len(res["hits"]), 0)

                # Verify that documents with required embeddings are included
                result_ids = [hit["_id"] for hit in res["hits"]]
                self.assertIn("doc_with_title", result_ids)
                self.assertIn("doc_mixed", result_ids)

    def test_recommend_allow_both_missing_parameters_true_success(self):
        """Test that both allowMissingDocuments=True and allowMissingEmbeddings=True work together"""
        # Create documents with various scenarios
        docs = [
            {
                "_id": "complete_doc",
                "title": "Complete document",
                "description": "Has both title and description"
            },
            {
                "_id": "partial_doc",
                "content": "Document with only content",
                "tags": ["partial"]
            }
        ]

        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                # For unstructured, specify which fields to vectorize
                if isinstance(index, UnstructuredMarqoIndex):
                    tensor_fields = ["title", "description"]  # Don't vectorize content
                else:
                    tensor_fields = None

                self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )

                # Should succeed with both parameters=True, handling missing docs and missing embeddings
                res = self.recommender.recommend(
                    index_name=index.name,
                    documents=[
                        "complete_doc",           # Exists, has required embeddings
                        "non_existent_doc",       # Doesn't exist (should be ignored)
                        "partial_doc",            # Exists, but missing required embeddings (should be ignored)
                        "another_missing_doc"     # Doesn't exist (should be ignored)
                    ],
                    tensor_fields=["title"],
                    allow_missing_documents=True,
                    allow_missing_embeddings=True,
                    exclude_input_documents=False
                )

                # Verify search was successful
                self.assertIn("hits", res)
                self.assertGreater(len(res["hits"]), 0)

                # Verify that only the document with required embeddings is used for context
                result_ids = [hit["_id"] for hit in res["hits"]]
                self.assertIn("complete_doc", result_ids)

    def test_recommend_all_documents_missing_with_allow_missing_documents_true_fails(self):
        """Test that when allowMissingDocuments=True but ALL documents are missing, it should fail"""
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                # Should still fail when no documents are available at all
                with self.assertRaisesStrict(InvalidArgumentError) as cm:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents=["non_existent_1", "non_existent_2", "non_existent_3"],
                        allow_missing_documents=True
                    )

                self.assertIn("Marqo could not collect any valid vector from the documents", str(cm.exception))

    def test_recommend_all_documents_missing_embeddings_with_allow_missing_embeddings_true_fails(self):
        """Test that when allowMissingEmbeddings=True but ALL documents lack embeddings, it should fail"""
        # Create documents that all lack a specific embedding field
        docs = [
            {
                "_id": "doc1",
                "content": "Document 1 with only content"
            },
            {
                "_id": "doc2", 
                "content": "Document 2 with only content"
            },
            {
                "_id": "doc3",
                "title": "Document 3",
            }
        ]

        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                # For unstructured, only vectorize content (not title)
                if isinstance(index, UnstructuredMarqoIndex):
                    tensor_fields = ["content", "title"]
                else:
                    tensor_fields = None

                self.add_documents(
                    self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )

                # Should still fail when no documents have the required embeddings
                with self.assertRaisesStrict(InvalidArgumentError) as cm:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents=["doc1", "doc2"],
                        tensor_fields=["title"],  # Request field that no documents have
                        allow_missing_embeddings=True
                    )

                self.assertIn("Marqo could not collect any valid vector from the documents", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
