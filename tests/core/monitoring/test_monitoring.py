import uuid
from unittest.mock import patch

import pytest

from marqo.api.exceptions import HardwareCompatabilityError
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.models.marqo_index import FieldType, FieldFeature, TextPreProcessing, TextSplitMethod, IndexType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.marqo_index_stats import MarqoIndexStats, VespaStats
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase, TestImageUrls


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

        structured_index_request_encoded_name = cls.structured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', ''),
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

        unstructured_index_request = cls.unstructured_marqo_index_request(
            text_preprocessing=TextPreProcessing(
                split_length=20,
                split_overlap=1,
                split_method=TextSplitMethod.Word
            )
        )

        cls.indexes = cls.create_indexes([
            structured_index_request,
            structured_index_request_encoded_name,
            structured_index_request_multimodal,
            unstructured_index_request
        ])

        cls.structured_index = cls.indexes[0]
        cls.structured_index_encoded_name = cls.indexes[1]
        cls.structured_index_multimodal = cls.indexes[2]
        cls.unstructured_index = cls.indexes[3]

        # Indexes to run generic tests against
        cls.indexes_to_test = [
            cls.structured_index,
            cls.structured_index_encoded_name,
            cls.unstructured_index
        ]

        # WARMUP monitoring calls to avoid NoneType return for first test run
        for i in range(2):
            for marqo_index in cls.indexes_to_test:
                cls.monitoring.get_index_stats(marqo_index)

    def test_get_index_stats_emptyIndex_successful(self):
        """
        get_index_stats returns the correct stats for an empty index
        """
        for marqo_index in self.indexes_to_test:
            with self.subTest(f'{marqo_index.name} - {marqo_index.type.value}'):
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=0,
                        number_of_vectors=0,
                        backend=VespaStats()
                    ),
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
                        device="cpu",
                        tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                    )
                )
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=3,
                        number_of_vectors=3,
                        backend=VespaStats()
                    ),
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
                     "img": TestImageUrls.HIPPO_REALISTIC.value
                     },
                    {"title": "2"},
                    {"desc": "2"}
                ],
                index_name=marqo_index.name,
                device="cpu"
            )
        )
        self.assertIndexStatsEqual(
            MarqoIndexStats(
                number_of_documents=3,
                number_of_vectors=4,
                backend=VespaStats()
            ),
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
                        device="cpu",
                        tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                    )
                )
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=3,
                        number_of_vectors=0,
                        backend=VespaStats()
                    ),
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
                        device="cpu",
                        tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                    )
                )
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=3,
                        number_of_vectors=2,
                        backend=VespaStats()
                    ),
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
                MarqoIndexStats(number_of_documents=3, number_of_vectors=3, backend=VespaStats())
            ),
            (
                'add',  # add 3 docs, 1 has a vector
                [{"_id": "4", "desc": "2"}, {"_id": "5", "title": "2"}, {"_id": "6", "desc": "62"}],
                MarqoIndexStats(number_of_documents=6, number_of_vectors=4, backend=VespaStats())
            ),
            (
                'delete',  # delete 2 docs, 1 has a vector
                ["1", "4"],
                MarqoIndexStats(number_of_documents=4, number_of_vectors=3, backend=VespaStats())
            ),
            (
                'add',  # add 3 docs, 1 has a vector
                [{"_id": "7", "desc": "2"}, {"_id": "8", "title": "2"}, {"_id": "9", "desc": "62"}],
                MarqoIndexStats(number_of_documents=7, number_of_vectors=4, backend=VespaStats())
            ),
            (
                'delete',  # delete all docs
                [f"{i}" for i in range(1, 10)],
                MarqoIndexStats(number_of_documents=0, number_of_vectors=0, backend=VespaStats())
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
                                device="cpu",
                                tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                            )
                        )
                    elif operation == 'delete':
                        tensor_search.delete_documents(
                            config=self.config,
                            index_name=marqo_index.name,
                            doc_ids=docs
                        )
                    self.assertIndexStatsEqual(
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
                        device="cpu",
                        tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                    )
                )
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=2,
                        number_of_vectors=4,
                        backend=VespaStats()
                    ),
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
                        device="cpu",
                        tensor_fields=['title'] if marqo_index.type == IndexType.Unstructured else None
                    )
                )
                self.assertIndexStatsEqual(
                    MarqoIndexStats(
                        number_of_documents=3,
                        number_of_vectors=3,
                        backend=VespaStats()
                    ),
                    self.monitoring.get_index_stats_by_name(marqo_index.name)
                )

    def test_get_index_stats_by_name_indexDoesNotExist_fails(self):
        """
        get_index_stats_by_name fails when the index does not exist
        """
        with self.assertRaises(IndexNotFoundError):
            self.monitoring.get_index_stats_by_name('index_does_not_exist')

    def assertIndexStatsEqual(self, expected: MarqoIndexStats, actual: MarqoIndexStats):
        """
        Assert MarqoIndexStats equality, verifying only range validity for storage and memory stats.
        """
        self.assertEqual(expected.number_of_documents, actual.number_of_documents)
        self.assertEqual(expected.number_of_vectors, actual.number_of_vectors)
        self.assertGreater(actual.backend.memory_used_percentage, 0.0)
        self.assertLessEqual(actual.backend.memory_used_percentage, 100.0)
        self.assertGreater(actual.backend.storage_used_percentage, 0.0)
        self.assertLessEqual(actual.backend.storage_used_percentage, 100.0)

    @pytest.mark.cpu_only
    def test_getCudaInfo_fail_on_cpu_instance(self):
        """
        getCudaInfo raise the correct error when called on a CPU instance
        """
        with self.assertRaises(HardwareCompatabilityError) as cm:
            self.monitoring.get_cuda_info()
        self.assertIn("CUDA is not available on this instance", cm.exception.message)

    @patch('marqo.core.monitoring.monitoring.torch.cuda.is_available', return_value=True)
    @patch('marqo.core.monitoring.monitoring.torch.cuda.device_count', return_value=1)
    @patch('marqo.core.monitoring.monitoring.torch.cuda.get_device_name', return_value='Tesla T4')
    @patch('marqo.core.monitoring.monitoring.torch.cuda.memory_allocated', return_value=3 * 1024 ** 3)
    @patch('marqo.core.monitoring.monitoring.torch.cuda.utilization', return_value=58)
    def test_getCudaInfo_success_on_gpu_instance(self, *mocks):
        """
        getCudaInfo returns the correct information when called on a GPU instance
        """
        class MockCudaProperties:
            total_memory = 12 * 1024 ** 3

        with patch('marqo.core.monitoring.monitoring.torch.cuda.get_device_properties',
                   return_value=MockCudaProperties()):
            response = self.monitoring.get_cuda_info()
            self.assertEqual(response.dict(), {
                "cuda_devices": [
                    {
                        "device_id": 0,
                        "device_name": "Tesla T4",
                        "memory_used": "3.0 GiB",
                        "total_memory": "12.0 GiB",
                        "utilization": "58.0 %",
                        "memory_used_percent": "25.0 %"
                    }
                ]
            })
