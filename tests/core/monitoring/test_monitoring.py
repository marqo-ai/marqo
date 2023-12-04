from marqo.core.exceptions import IndexNotFoundError
from marqo.core.models.marqo_index import FieldType, FieldFeature, TextPreProcessing, TextSplitMethod
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.marqo_index_stats import MarqoIndexStats
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestMonitoring(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.monitoring = cls.config.monitoring

        structured_index_request = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(
                    name='desc',
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch]
                ),
                FieldRequest(
                    name='tags',
                    type=FieldType.ArrayText,
                    features=[FieldFeature.Filter, FieldFeature.LexicalSearch]
                ),
                FieldRequest(
                    name='price',
                    type=FieldType.Float,
                    features=[FieldFeature.ScoreModifier]
                ),
            ],
            tensor_fields=['title'],
            text_preprocessing=TextPreProcessing(
                split_length=20,
                split_overlap=1,
                split_method=TextSplitMethod.Word
            )
        )

        structured_index_request_multimodal = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(
                    name='desc',
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch]
                ),
                FieldRequest(
                    name='img',
                    type=FieldType.ImagePointer
                ),
                FieldRequest(
                    name='price',
                    type=FieldType.Float,
                    features=[FieldFeature.ScoreModifier]
                ),
                FieldRequest(
                    name='title_img',
                    type=FieldType.MultimodalCombination,
                    dependent_fields={'title': 1.0, 'img': 2.0}
                )
            ],
            tensor_fields=['title', 'title_img'],
        )

        cls.indexes = cls.create_indexes([
            structured_index_request,
            structured_index_request_multimodal
        ])

        cls.structured_index = cls.indexes[0]
        cls.structured_index_multimodal = cls.indexes[1]

        # TODO - Add unstructured index to the index list to test against
        # Indexes to run generic tests against
        cls.indexes_to_test = [cls.structured_index]

    def test_get_index_stats_emptyIndex_successful(self):
        """
        get_index_stats returns the correct stats for an empty index
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=0, number_of_vectors=0),
                    self.monitoring.get_index_stats(marqo_index)
                )

    def test_get_index_stats_docsWithTensorFields_successful(self):
        """
        get_index_stats returns the correct stats for an index with documents that have tensor fields
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        docs=[{"title": "2"}, {"title": "2"}, {"title": "62"}],
                        index_name=marqo_index.name,
                        device="cpu"
                    )
                )
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=3, number_of_vectors=3),
                    self.monitoring.get_index_stats(marqo_index)
                )

    def test_get_index_stats_structuredMultimodalIndex_successful(self):
        """
        get_index_stats returns the correct stats for a multimodal index
        """
        marqo_index = self.structured_index_multimodal
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=[
                    {"title": "2",
                     "img": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/"
                            "ai_hippo_realistic.png"
                     },
                    {"title": "2"},
                    {"desc": "2"}
                ],
                index_name=marqo_index.name,
                device="cpu"
            )
        )
        self.assertEqual(
            MarqoIndexStats(number_of_documents=3, number_of_vectors=4),
            self.monitoring.get_index_stats(marqo_index)
        )

    def test_get_index_stats_docsWithoutTensorFields_successful(self):
        """
        get_index_stats returns the correct stats for an index with documents that do not have tensor fields
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        docs=[{"desc": "2"}, {"desc": "2"}, {"desc": "62"}],
                        index_name=marqo_index.name,
                        device="cpu"
                    )
                )
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=3, number_of_vectors=0),
                    self.monitoring.get_index_stats(marqo_index)
                )

    def test_get_index_stats_mixedDocs_successful(self):
        """
        get_index_stats returns the correct stats for an index with documents that have and do not have tensor fields
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        docs=[{"title": "2"}, {"title": "2"}, {"desc": "62"}],
                        index_name=marqo_index.name,
                        device="cpu"
                    )
                )
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=3, number_of_vectors=2),
                    self.monitoring.get_index_stats(marqo_index)
                )

    def test_get_index_stats_sequentialIndexingAndDeletion_successful(self):
        """
        get_index_stats returns the correct stats for an index during a sequence of indexing and deletion
        """
        operations = [
            (
                'add',  # add 3 docs, 3 have a vector
                [{"_id": "1", "title": "2"}, {"_id": "2", "title": "2"}, {"_id": "3", "title": "62"}],
                MarqoIndexStats(number_of_documents=3, number_of_vectors=3)
            ),
            (
                'add',  # add 3 docs, 1 has a vector
                [{"_id": "4", "desc": "2"}, {"_id": "5", "title": "2"}, {"_id": "6", "desc": "62"}],
                MarqoIndexStats(number_of_documents=6, number_of_vectors=4)
            ),
            (
                'delete',  # delete 2 docs, 1 has a vector
                ["1", "4"],
                MarqoIndexStats(number_of_documents=4, number_of_vectors=3)
            ),
            (
                'add',  # add 3 docs, 1 has a vector
                [{"_id": "7", "desc": "2"}, {"_id": "8", "title": "2"}, {"_id": "9", "desc": "62"}],
                MarqoIndexStats(number_of_documents=7, number_of_vectors=4)
            ),
            (
                'delete',  # delete all docs
                [f"{i}" for i in range(1, 10)],
                MarqoIndexStats(number_of_documents=0, number_of_vectors=0)
            )
        ]

        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                for operation, docs, expected_stats in operations:
                    if operation == 'add':
                        tensor_search.add_documents(
                            config=self.config, add_docs_params=AddDocsParams(
                                docs=docs,
                                index_name=marqo_index.name,
                                device="cpu"
                            )
                        )
                    elif operation == 'delete':
                        tensor_search.delete_documents(
                            config=self.config,
                            index_name=marqo_index.name,
                            doc_ids=docs
                        )
                    self.assertEqual(
                        expected_stats,
                        self.monitoring.get_index_stats(marqo_index)
                    )

    def test_get_index_stats_longText_successful(self):
        """
        get_index_stats returns the correct stats for an index with a long text field that is chunked into multiple
        vectors
        """
        number_of_words = 55

        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        docs=[{"title": "test " * number_of_words}, {"title": "2"}],  # 3 + 1 vectors expected
                        index_name=marqo_index.name,
                        device="cpu"
                    )
                )
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=2, number_of_vectors=4),
                    self.monitoring.get_index_stats(marqo_index)
                )

    def test_get_index_stats_by_name_docsWithTensorFields_successful(self):
        """
        get_index_stats_by_name returns the correct stats for an index with documents that have tensor fields
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        docs=[{"title": "2"}, {"title": "2"}, {"title": "62"}],
                        index_name=marqo_index.name,
                        device="cpu"
                    )
                )
                self.assertEqual(
                    MarqoIndexStats(number_of_documents=3, number_of_vectors=3),
                    self.monitoring.get_index_stats_by_name(marqo_index.name)
                )

    def test_get_index_stats_by_name_indexDoesNotExist_fails(self):
        """
        get_index_stats_by_name fails when the index does not exist
        """
        with self.assertRaises(IndexNotFoundError):
            self.monitoring.get_index_stats_by_name('index_does_not_exist')
