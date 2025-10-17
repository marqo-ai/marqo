import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase


class TestStructuredGetSettings(MarqoTestCase):
    default_index_name = "default_index" + str(uuid.uuid4()).replace('-', '')
    custom_index_name = "custom_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.indexes_to_delete = [cls.default_index_name, cls.custom_index_name]
        cls.client = Client(**cls.client_settings)

    def setUp(self) -> None:
        # Override the setUp to disable the clear index call in this unittest
        pass

    def test_default_settings(self):
        """default fields should be returned if index is created with default settings, in camel case
            sample structure of output:
        {'type': 'structured',
        'allFields': [{'name': 'title', 'type': 'text', 'features': []}],
        'tensorFields': ['title'],
        'model': 'hf/all_datasets_v4_MiniLM-L6',
        'normalizeEmbeddings': True,
        'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
        'imagePreprocessing': {},
        'vectorNumericType': 'float',
        'annParameters': {'spaceType': 'angular', 'parameters': {'efConstruction': 128, 'm': 16}}
        }
        """
        self.client.create_index(index_name=self.default_index_name,
                                 type="structured",
                                 all_fields=[{"name": "title", "type": "text"}],
                                 tensor_fields=["title"])

        ix = self.client.index(self.default_index_name)
        index_settings = ix.get_settings()
        fields = {"type", "allFields", "tensorFields", "model",
                  "normalizeEmbeddings", "textPreprocessing", "imagePreprocessing",
                  "vectorNumericType", "annParameters"}
        self.assertTrue(fields.issubset(set(index_settings)))

    def test_custom_settings(self):
        """adding custom settings to the index should be reflected in the returned output
        Sample output:
        {'type': 'structured',
        'allFields': [{'name': 'title', 'type': 'text', 'features': []}, {'name': 'content', 'type': 'text', 'features': []}],
        'tensorFields': ['title', 'content'],
        'model': 'test-model',
        'modelProperties': {'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'dimensions': 384, 'tokens': 128, 'type': 'hf'},
        'normalizeEmbeddings': True,
        'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
        'imagePreprocessing': {},
        'vectorNumericType': 'float',
        'annParameters': {'spaceType': 'angular', 'parameters': {'efConstruction': 128, 'm': 16}}
        }
        """
        model_properties = {'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                            'dimensions': 384,
                            'tokens': 128,
                            'type': 'hf'}

        index_settings = {
            "type": "structured",
            'model': 'test-model',
            'modelProperties': model_properties,
            'normalizeEmbeddings': True,
            "allFields": [
                {"name": "title", "type": "text"},
                {"name": "content", "type": "text"},
            ],
            "tensorFields": ["title", "content"],
        }

        res = self.client.create_index(index_name=self.custom_index_name, settings_dict=index_settings)

        ix = self.client.index(self.custom_index_name)
        index_settings = ix.get_settings()
        fields = {"type", "allFields", "tensorFields", "model", "modelProperties",
                  "normalizeEmbeddings", "textPreprocessing", "imagePreprocessing",
                  "vectorNumericType", "annParameters"}
        self.assertTrue(fields.issubset(set(index_settings)))