import unittest

from marqo.core.exceptions import IndexNotFoundError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from unittest.mock import patch
import os
from marqo.tensor_search.models.index_settings import IndexSettings, IndexSettingsWithName
import pprint
import uuid


class TestGetSettings(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Default indexes
        cls.unstructured_default_index = cls.config.index_management.create_index(
            IndexSettings().to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))
        )
        cls.structured_default_index = cls.config.index_management.create_index(
            IndexSettings(
                type=IndexType.Structured,
                allFields=[
                    FieldRequest(name='field1', type=FieldType.Text),
                    FieldRequest(name='field2', type=FieldType.Text),
                ],
                tensorFields=[]
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))
        )

        # Custom settings indexes
        cls.unstructured_custom_index = cls.config.index_management.create_index(
            IndexSettings(
                type=IndexType.Unstructured,
                model='ViT-B/32',
                normalizeEmbeddings=False,
                textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))
        )
        cls.structured_custom_index = cls.config.index_management.create_index(
            IndexSettings(
                type=IndexType.Structured,
                allFields=[
                    FieldRequest(name='field1', type=FieldType.Text),
                    FieldRequest(name='field2', type=FieldType.Text),
                ],
                tensorFields=[],
                model='ViT-B/32',
                normalizeEmbeddings=False,
                textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))
        )

        cls.indexes = [
            cls.unstructured_default_index,
            cls.structured_default_index,
            cls.unstructured_custom_index,
            cls.structured_custom_index
        ]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()


    def test_no_index(self):
        self.assertRaises(IndexNotFoundError, self.config.index_management.get_index, "non-existent-index")

    def test_default_settings(self):
        """default fields should be returned if index is created with default settings
        """

        with self.subTest("Unstructured index default settings"):
            expected_unstructured_default_settings = \
                {
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAnguar
                    },
                    'filterStringMaxLength': 20,
                    'imagePreprocessing': {},
                    'model': 'hf/e5-base-v2',
                    'normalizeEmbeddings': True,
                    'textPreprocessing': {'splitLength': 2,
                                        'splitMethod': TextSplitMethod.Sentence,
                                        'splitOverlap': 0},
                    'treatUrlsAndPointersAsImages': False,
                    'type': IndexType.Unstructured,
                    'vectorNumericType': VectorNumericType.Float
                }
            # Get unstructured default settings
            self.maxDiff = None
            retrieved_index = self.config.index_management.get_index(self.unstructured_default_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_unstructured_default_settings)
        
        with self.subTest("Structured index default settings"):
            expected_structured_default_settings = \
                {
                    'allFields': [
                        {
                            'features': [],
                            'name': 'field1',
                            'type': FieldType.Text
                        },
                        {
                            'features': [],
                            'name': 'field2',
                            'type': FieldType.Text
                        }
                    ],
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAnguar
                    },
                    'imagePreprocessing': {},
                    'model': 'hf/e5-base-v2',
                    'normalizeEmbeddings': True,
                    'tensorFields': [],
                    'textPreprocessing': {
                        'splitLength': 2,
                        'splitMethod': TextSplitMethod.Sentence,
                        'splitOverlap': 0
                    },
                    'type': IndexType.Structured,
                    'vectorNumericType': VectorNumericType.Float
                }
            # Get unstructured default settings
            retrieved_index = self.config.index_management.get_index(self.structured_default_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_structured_default_settings)
        

    def test_custom_settings(self):
        """adding custom settings to the index should be reflected in the returned output
        """
        
        with self.subTest("Unstructured index custom settings"):
            expected_unstructured_custom_settings = \
                {
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAnguar
                    },
                    'filterStringMaxLength': 20,
                    'imagePreprocessing': {},
                    'model': 'ViT-B/32',
                    'normalizeEmbeddings': False,
                    'textPreprocessing': {'splitLength': 3,
                                        'splitMethod': TextSplitMethod.Word,
                                        'splitOverlap': 1},
                    'treatUrlsAndPointersAsImages': False,
                    'type': IndexType.Unstructured,
                    'vectorNumericType': VectorNumericType.Float
                }
            # Get unstructured custom settings
            retrieved_index = self.config.index_management.get_index(self.unstructured_custom_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_unstructured_custom_settings)
        
        with self.subTest("Structured index custom settings"):
            expected_structured_custom_settings = \
                {
                    'allFields': [
                        {
                            'features': [],
                            'name': 'field1',
                            'type': FieldType.Text
                        },
                        {
                            'features': [],
                            'name': 'field2',
                            'type': FieldType.Text
                        }
                    ],
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAnguar
                    },
                    'imagePreprocessing': {},
                    'model': 'ViT-B/32',
                    'normalizeEmbeddings': False,
                    'tensorFields': [],
                    'textPreprocessing': {
                        'splitLength': 3,
                        'splitMethod': TextSplitMethod.Word,
                        'splitOverlap': 1
                    },
                    'type': IndexType.Structured,
                    'vectorNumericType': VectorNumericType.Float
                }
            # Get unstructured default settings
            retrieved_index = self.config.index_management.get_index(self.structured_custom_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_structured_custom_settings)
            