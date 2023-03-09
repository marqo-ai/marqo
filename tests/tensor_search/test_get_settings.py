from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search, backend
from marqo.errors import IndexNotFoundError


class TestGetSettings(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {'Content-type': 'application/json'}
        self.index_name = 'my-test-index-1'
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except IndexNotFoundError as s:
            pass

    def test_no_index(self):
        self.assertRaises(IndexNotFoundError, backend.get_index_info, config=self.config, index_name=self.index_name)

    def test_default_settings(self):
        """default fields should be returned if index is created with default settings
            sample structure of output: {'index_defaults': {'treat_urls_and_pointers_as_images': False,
                                          'text_preprocessing': {'split_method': 'sentence', 'split_length': 2,
                                                                 'split_overlap': 0},
                                          'model': 'hf/all_datasets_v4_MiniLM-L6', 'normalize_embeddings': True,
                                          'image_preprocessing': {'patch_method': None}}, 'number_of_shards': 5}
        """
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name)

        index_info = backend.get_index_info(config=self.config, index_name=self.index_name)
        index_settings = index_info.index_settings
        fields = {'treat_urls_and_pointers_as_images', 'text_preprocessing', 'model', 'normalize_embeddings',
                  'image_preprocessing'}

        self.assertIn('index_defaults', index_settings)
        self.assertIn('number_of_shards', index_settings)
        self.assertTrue(fields.issubset(set(index_settings['index_defaults'])))

    def test_custom_settings(self):
        """adding custom settings to the index should be reflected in the returned output
        """
        model_properties = {'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                            'dimensions': 384,
                            'tokens': 128,
                            'type': 'sbert'}

        index_settings = {
            'index_defaults': {
                'treat_urls_and_pointers_as_images': False,
                'model': 'test-model',
                'model_properties': model_properties,
                'normalize_embeddings': True,
            }
        }

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        index_info = backend.get_index_info(config=self.config, index_name=self.index_name)
        index_settings = index_info.index_settings
        fields = {'treat_urls_and_pointers_as_images', 'text_preprocessing', 'model', 'normalize_embeddings',
                  'image_preprocessing', 'model_properties'}

        self.assertIn('index_defaults', index_settings)
        self.assertIn('number_of_shards', index_settings)
        self.assertTrue(fields.issubset(set(index_settings['index_defaults'])))
