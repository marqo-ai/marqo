from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search, backend
from marqo.errors import IndexNotFoundError


class TestGetSettings(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except IndexNotFoundError as s:
            pass

    def test_no_index(self):
        self.assertRaises(IndexNotFoundError, backend.get_index_info, config=self.config, index_name=self.index_name)

    def test_default_settings(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name)

        test_index_info = list(backend.get_index_info(config=self.config, index_name=self.index_name))
        index_info = ['hf/all_datasets_v4_MiniLM-L6',
                      {'__chunks': {'type': 'nested', 'properties': {'__field_content': {'type': 'text'},
                                                                     '__field_name': {
                                                                         'type': 'keyword'}}}},
                      {'index_defaults': {'treat_urls_and_pointers_as_images': False,
                                          'text_preprocessing': {'split_method': 'sentence', 'split_length': 2,
                                                                 'split_overlap': 0},
                                          'model': 'hf/all_datasets_v4_MiniLM-L6', 'normalize_embeddings': True,
                                          'image_preprocessing': {'patch_method': None}}, 'number_of_shards': 5}]

        self.assertEqual(index_info, test_index_info)

    def test_custom_settings(self):
        model_properties = {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                            "dimensions": 384,
                            "tokens": 128,
                            "type": "sbert"}

        index_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "test-model",
                "model_properties": model_properties,
                "normalize_embeddings": True,
            }
        }

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        test_index_info = list(backend.get_index_info(config=self.config, index_name=self.index_name))
        index_info = ['test-model', {'__chunks': {'type': 'nested', 'properties': {'__field_content': {'type': 'text'},
                                                                                  '__field_name': {
                                                                                      'type': 'keyword'}}}}, {
                          'index_defaults': {'treat_urls_and_pointers_as_images': False,
                                             'text_preprocessing': {'split_method': 'sentence', 'split_length': 2,
                                                                    'split_overlap': 0}, 'model_properties': {
                                  'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'tokens': 128,
                                  'type': 'sbert', 'dimensions': 384}, 'model': 'test-model',
                                             'normalize_embeddings': True,
                                             'image_preprocessing': {'patch_method': None}}, 'number_of_shards': 5}]

        self.assertEqual(index_info, test_index_info)

    def test_settings_with_documents(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name)
        tensor_search.add_documents(config=self.config, index_name=self.index_name, docs=[
            {
                "Title": "test-title",
                "Description": "test-desc",
                "_id": "1"
            }], auto_refresh=True)

        test_index_info = list(backend.get_index_info(config=self.config, index_name=self.index_name))
        index_info = ['hf/all_datasets_v4_MiniLM-L6', {'Description': {'type': 'text'}, 'Title': {'type': 'text'},
                                                       '__chunks': {'type': 'nested', 'properties': {
                                                           'Description': {'type': 'keyword', 'ignore_above': 32766},
                                                           'Title': {'type': 'keyword', 'ignore_above': 32766},
                                                           '__field_content': {'type': 'text'},
                                                           '__field_name': {'type': 'keyword'},
                                                           '__vector_Description': {'type': 'knn_vector',
                                                                                    'dimension': 384,
                                                                                    'method': {'engine': 'lucene',
                                                                                               'space_type': 'cosinesimil',
                                                                                               'name': 'hnsw',
                                                                                               'parameters': {
                                                                                                   'ef_construction': 128,
                                                                                                   'm': 16}}},
                                                           '__vector_Title': {'type': 'knn_vector', 'dimension': 384,
                                                                              'method': {'engine': 'lucene',
                                                                                         'space_type': 'cosinesimil',
                                                                                         'name': 'hnsw', 'parameters': {
                                                                                      'ef_construction': 128,
                                                                                      'm': 16}}}}}}, {
                          'index_defaults': {'treat_urls_and_pointers_as_images': False,
                                             'text_preprocessing': {'split_method': 'sentence', 'split_length': 2,
                                                                    'split_overlap': 0},
                                             'model': 'hf/all_datasets_v4_MiniLM-L6', 'normalize_embeddings': True,
                                             'image_preprocessing': {'patch_method': None}}, 'number_of_shards': 5}]

        self.assertEqual(index_info, test_index_info)

        # deleting docs should not change index settings
        tensor_search.delete_documents(config=self.config, index_name=self.index_name, doc_ids=["1"],
                                       auto_refresh=True)
        self.assertEqual(index_info, test_index_info)
