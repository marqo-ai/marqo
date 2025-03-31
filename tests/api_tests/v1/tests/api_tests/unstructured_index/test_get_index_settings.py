import uuid

from tests.marqo_test import MarqoTestCase


class TestGetSettings(MarqoTestCase):
    default_index_name = "default_index" + str(uuid.uuid4()).replace('-', '')
    custom_index_name = "custom_index" + str(uuid.uuid4()).replace('-', '')
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.indexes_to_delete = [cls.default_index_name, cls.custom_index_name]

    def setUp(self) -> None:
        # Override the setUp to disable the clear index call in this unittest
        pass

    def test_default_settings(self):
        """default fields should be returned if index is created with default settings
            sample structure of output:
            {'type': 'unstructured',
            'treatUrlsAndPointersAsImages': False,
            'shortStringLengthThreshold': 20,
            'model': 'hf/all_datasets_v4_MiniLM-L6',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
            'imagePreprocessing': {},
            'vectorNumericType': 'float',
            'annParameters': {'spaceType': 'angular', 'parameters': {'efConstruction': 128, 'm': 16}}
            }
        """
        self.client.create_index(index_name=self.default_index_name, type="unstructured")

        ix = self.client.index(self.default_index_name)
        index_settings = ix.get_settings()
        fields = {'treatUrlsAndPointersAsImages', 'textPreprocessing', 'model', 'normalizeEmbeddings',
                  'imagePreprocessing', "filterStringMaxLength"}

        self.assertTrue(fields.issubset(set(index_settings)))

    def test_custom_settings(self):
        """adding custom settings to the index should be reflected in the returned output
        """
        model_properties = {'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                            'dimensions': 384,
                            'tokens': 128,
                            'type': 'hf'}

        index_settings = {
            "type": "unstructured",
            'treatUrlsAndPointersAsImages': False,
            'model': 'test-model',
            'modelProperties': model_properties,
            'normalizeEmbeddings': True,
        }

        res = self.client.create_index(index_name=self.custom_index_name, settings_dict=index_settings)

        ix = self.client.index(self.custom_index_name)
        index_settings = ix.get_settings()
        fields = {'treatUrlsAndPointersAsImages', 'textPreprocessing', 'model', 'normalizeEmbeddings',
                  'imagePreprocessing', "filterStringMaxLength"}

        self.assertTrue(fields.issubset(set(index_settings)))