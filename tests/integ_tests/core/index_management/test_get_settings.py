import os
import uuid
from unittest.mock import patch

from marqo.core.exceptions import IndexNotFoundError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search.models.index_settings import IndexSettings
from integ_tests.marqo_test import MarqoTestCase


class TestGetSettings(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_default_index = IndexSettings().to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        structured_default_index = IndexSettings(
                type=IndexType.Structured,
                allFields=[
                    FieldRequest(name='field1', type=FieldType.Text),
                    FieldRequest(name='field2', type=FieldType.Text),
                ],
                tensorFields=[]
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        unstructured_custom_index = IndexSettings(
                type=IndexType.Unstructured,
                model='ViT-B/32',
                normalizeEmbeddings=False,
                textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        unstructured_marqtune_index = IndexSettings(
            model='marqtune/model-id/release-checkpoint',
            modelProperties={
                "isMarqtuneModel": True,
                "name": "ViT-B-32",
                "dimensions": 512,
                "model_location": {
                    "s3": {
                        "Bucket": "marqtune-public-bucket",
                        "Key": "marqo-test-open-clip-model/epoch_1.pt",
                    },
                    "auth_required": False
                },
                "type": "open_clip",
            },
            normalizeEmbeddings=False,
            textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
        ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        unstructured_non_marqtune_index = IndexSettings(
            type=IndexType.Unstructured,
            model='marqtune/model-id/release-checkpoint',
            modelProperties={
                "dimensions": 384,
                "model_location": {
                    "s3": {
                        "Bucket": "marqtune-public-bucket",
                        "Key": "marqo-test-hf-model/epoch_1.zip",
                    },
                    "auth_required": False
                },
                "type": "hf",
            },
            normalizeEmbeddings=False,
            textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
        ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        structured_custom_index = IndexSettings(
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


        structured_marqtune_index = IndexSettings(
                type=IndexType.Structured,
                allFields=[
                    FieldRequest(name='field1', type=FieldType.Text),
                    FieldRequest(name='field2', type=FieldType.Text),
                ],
                tensorFields=[],
            model='marqtune/model-id/release-checkpoint',
            modelProperties={
                    "isMarqtuneModel": True,
                    "dimensions": 384,
                    "model_location": {
                        "s3": {
                            "Bucket": "marqtune-public-bucket",
                            "Key": "marqo-test-hf-model/epoch_1.zip",
                        },
                        "auth_required": False
                    },
                    "trustRemoteCode": True,
                    "type": "hf",
                },
                normalizeEmbeddings=False,
                textPreprocessing=TextPreProcessing(splitLength=3, splitMethod=TextSplitMethod.Word, splitOverlap=1),
            ).to_marqo_index_request('a' + str(uuid.uuid4()).replace('-', ''))

        cls.indexes = cls.create_indexes([
            unstructured_default_index,
            structured_default_index,
            unstructured_custom_index,
            structured_custom_index,
            unstructured_marqtune_index,
            structured_marqtune_index,
            unstructured_non_marqtune_index
        ])

        cls.unstructured_default_index = cls.indexes[0]
        cls.structured_default_index = cls.indexes[1]
        cls.unstructured_custom_index = cls.indexes[2]
        cls.structured_custom_index = cls.indexes[3]
        cls.unstructured_marqtune_index = cls.indexes[4]
        cls.structured_marqtune_index = cls.indexes[5]
        cls.unstructured_non_marqtune_index = cls.indexes[6]

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
                        'spaceType': DistanceMetric.PrenormalizedAngular
                    },
                    'filterStringMaxLength': 50,
                    'imagePreprocessing': {},
                    'model': 'hf/e5-base-v2',
                    'normalizeEmbeddings': True,
                    'textPreprocessing': {'splitLength': 2,
                                        'splitMethod': TextSplitMethod.Sentence,
                                        'splitOverlap': 0},
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'treatUrlsAndPointersAsImages': False,
                    'treatUrlsAndPointersAsMedia': False,
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
                        'spaceType': DistanceMetric.PrenormalizedAngular
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
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
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
                        'spaceType': DistanceMetric.PrenormalizedAngular
                    },
                    'filterStringMaxLength': 50,
                    'imagePreprocessing': {},
                    'model': 'ViT-B/32',
                    'normalizeEmbeddings': False,
                    'textPreprocessing': {'splitLength': 3,
                                        'splitMethod': TextSplitMethod.Word,
                                        'splitOverlap': 1},
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'treatUrlsAndPointersAsImages': False,
                    'treatUrlsAndPointersAsMedia': False,
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
                        'spaceType': DistanceMetric.PrenormalizedAngular
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
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'type': IndexType.Structured,
                    'vectorNumericType': VectorNumericType.Float
                }
            # Get unstructured default settings
            retrieved_index = self.config.index_management.get_index(self.structured_custom_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_structured_custom_settings)

    def test_index_settings_with_marqtune_model(self):
        """Model name, dimensions, and model location should be hidden if model is marqtune
        """
        with self.subTest("Unstructured index with marqtune model"):
            expected_unstructured_custom_settings = \
                {
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAngular
                    },
                    'filterStringMaxLength': 50,
                    'imagePreprocessing': {},
                    'model': 'marqtune/model-id/release-checkpoint',
                    'modelProperties': {
                        "isMarqtuneModel": True,
                    },
                    'normalizeEmbeddings': False,
                    'textPreprocessing': {'splitLength': 3,
                                          'splitMethod': TextSplitMethod.Word,
                                          'splitOverlap': 1},
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'treatUrlsAndPointersAsImages': False,
                    'treatUrlsAndPointersAsMedia': False,
                    'type': IndexType.Unstructured,
                    'vectorNumericType': VectorNumericType.Float
                }

            retrieved_index = self.config.index_management.get_index(self.unstructured_marqtune_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_unstructured_custom_settings)

        with self.subTest("Structured index with marqtune model"):
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
                        'spaceType': DistanceMetric.PrenormalizedAngular
                    },
                    'imagePreprocessing': {},
                    'model': 'marqtune/model-id/release-checkpoint',
                    'modelProperties': {
                        "isMarqtuneModel": True,
                    },
                    'normalizeEmbeddings': False,
                    'tensorFields': [],
                    'textPreprocessing': {
                        'splitLength': 3,
                        'splitMethod': TextSplitMethod.Word,
                        'splitOverlap': 1
                    },
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'type': IndexType.Structured,
                    'vectorNumericType': VectorNumericType.Float
                }

            retrieved_index = self.config.index_management.get_index(self.structured_marqtune_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_structured_custom_settings)

        with self.subTest("Unstructured index with non-marqtune model"):
            expected_unstructured_custom_settings = \
                {
                    'annParameters': {
                        'parameters': {'efConstruction': 512, 'm': 16},
                        'spaceType': DistanceMetric.PrenormalizedAngular
                    },
                    'filterStringMaxLength': 50,
                    'imagePreprocessing': {},
                    'model': 'marqtune/model-id/release-checkpoint',
                    'modelProperties': {
                        'dimensions': 384,
                        'model_location': {
                            'auth_required': False,
                            's3': {
                                'Bucket': 'marqtune-public-bucket',
                                'Key': 'marqo-test-hf-model/epoch_1.zip'
                            }
                        },
                        'type': 'hf'
                    },
                    'normalizeEmbeddings': False,
                    'textPreprocessing': {'splitLength': 3,
                                          'splitMethod': TextSplitMethod.Word,
                                          'splitOverlap': 1},
                    'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
                    'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
                    'treatUrlsAndPointersAsImages': False,
                    'treatUrlsAndPointersAsMedia': False,
                    'type': IndexType.Unstructured,
                    'vectorNumericType': VectorNumericType.Float
                }

            retrieved_index = self.config.index_management.get_index(self.unstructured_non_marqtune_index.name)
            retrieved_settings = IndexSettings.from_marqo_index(retrieved_index).dict(exclude_none=True, by_alias=True)
            self.assertEqual(retrieved_settings, expected_unstructured_custom_settings)
