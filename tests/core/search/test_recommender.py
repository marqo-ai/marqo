import os
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
from tests.marqo_test import MarqoTestCase


class TestRecommender(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Note text_preprocessing is set to create one vector per field, so that vector count is predictable
        # This is required for some tests

        unstructured_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            text_preprocessing=TextPreProcessing(
                split_length=1000,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            )
        )

        unstructured_text_index_nonnormalized = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'), normalize_embeddings=False,
            text_preprocessing=TextPreProcessing(
                split_length=1000,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
        )

        structured_text_index = cls.structured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
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

        self.recommender = Recommender(self.vespa_client, self.index_management)

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

                    mock_interpolate.assert_called_once()

                    ids = [doc["_id"] for doc in res["hits"]]

                    self.assertEqual(set(ids), {"1", "2", "6"})

    def test_recommend_slerpZeroSumWeights_failure(self):
        """
        Test that the recommender fails when the sum of consecutive weights is zero
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            with self.subTest(type=index.type):
                self._populate_index(index)

                with self.assertRaisesStrict(InvalidArgumentError) as ex:
                    self.recommender.recommend(
                        index_name=index.name,
                        documents={"1": 1, "2": -1},
                        tensor_fields=['title'],
                        interpolation_method=InterpolationMethod.SLERP,
                        exclude_input_documents=False,
                    )
                self.assertIn('SLERP cannot interpolate', str(ex.exception))

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

                    mock_interpolate.assert_called_once()

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

                    mock_interpolate.assert_called_once()

                    ids = [doc["_id"] for doc in res["hits"]]

                    self.assertEqual(set(ids), {"1", "2", "6"})

    def test_recommend_lerpZeroSumWeights_failure(self):
        """
        Test that the recommender fails when the sum of all weights is zero with LERP (and NLERP)
        """
        for index in [self.unstructured_text_index, self.structured_text_index]:
            for method in [InterpolationMethod.LERP, InterpolationMethod.NLERP]:
                with self.subTest(type=index.type, method=method):
                    self._populate_index(index)

                    with self.assertRaisesStrict(InvalidArgumentError) as ex:
                        self.recommender.recommend(
                            index_name=index.name,
                            documents={"1": 1, "2": 2, "3": -3},
                            tensor_fields=['title'],
                            interpolation_method=method,
                            exclude_input_documents=False,
                        )
                    self.assertIn('Sum of weights is zero', str(ex.exception))

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

            mock_interpolate.assert_called_once()

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
                        processing_start=mock.ANY
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
                        processing_start=mock.ANY
                    )
