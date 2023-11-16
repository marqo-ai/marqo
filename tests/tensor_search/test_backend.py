import copy
import json
from tests.utils.transition import add_docs_caller
import requests
from marqo.tensor_search import enums, backend, utils
from marqo.tensor_search import tensor_search
from marqo.tensor_search.configs import get_default_ann_parameters
from marqo.errors import MarqoApiError, IndexNotFoundError
from tests.marqo_test import MarqoTestCase
from unittest import mock


class TestBackend(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(self.config, self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_chunk_properties_arent_deleted(self):
        """TODO - make sure adding new properties doesn't discard old ones"""

    def test_get_index_info(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1
        )
        index_info = backend.get_index_info(
            config=self.config, index_name=self.index_name_1)
        assert index_info.model_name
        assert "__field_name" in index_info.properties[enums.TensorField.chunks]["properties"]
        assert isinstance(index_info.properties, dict)

    def test_get_index_info_no_index(self):
        r1 = requests.get(
            url=f"{self.authorized_url}/{self.index_name_1}",
            verify=False
        )
        assert r1.status_code == 404
        try:
            index_info = backend.get_index_info(
                config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_get_index_info_no_search_model(self):
        """
        Indexes created with Marqo v.1.4.0 and earlier do not have search_model or search_model_properties.
        Thus, backend response will not contain those fields in metadata.
        For backwards compatibility, this method should populate `IndexInfo.search_model_name` with `None`.
        """

        mock__get = mock.MagicMock()
        
        # Sample backend response from OpenSearch (with no search_model)
        mock__get.return_value = {
            self.index_name_1: {
                'mappings': {
                    '_meta': {
                        'index_settings': {
                            'index_defaults': {
                                'ann_parameters': {
                                    'engine': 'lucene',
                                    'name': 'hnsw',
                                    'parameters': {
                                        'ef_construction': 128,
                                        'm': 16
                                    },
                                    'space_type': 'cosinesimil'
                                },
                                'image_preprocessing': {'patch_method': None},
                                'model': 'ViT-L/14',
                                'normalize_embeddings': True,
                                'text_preprocessing': {'split_length': 2,
                                                        'split_method': 'sentence',
                                                        'split_overlap': 0},
                                'treat_urls_and_pointers_as_images': True
                            },
                            'number_of_replicas': 0,
                            'number_of_shards': 3
                        },
                        'media_type': 'text',
                        'model': 'ViT-L/14'
                    },
                    'dynamic_templates': [{'strings': {'mapping': {'type': 'text'}, 'match_mapping_type': 'string'}}],
                    'properties': {'Title': {'type': 'text'},
                    '__chunks': {
                        'properties': {
                            'Title': {
                                'ignore_above': 32766,
                                'type': 'keyword'
                            },
                            '__field_content': {'type': 'text'},
                            '__field_name': {'type': 'keyword'},
                            '__vector_marqo_knn_field': {
                                'dimension': 768,
                                'method': {
                                    'engine': 'lucene',
                                    'name': 'hnsw',
                                    'parameters': {
                                        'ef_construction': 128,
                                        'm': 16
                                    },
                                    'space_type': 'cosinesimil'
                                },
                                'type': 'knn_vector'
                            },
                            'captioned_image': {
                                'properties': {
                                    'caption': {
                                        'ignore_above': 32766,
                                        'type': 'keyword'
                                    },
                                    'image': {'ignore_above': 32766, 'type': 'keyword'}
                                }
                            }
                        },
                        'type': 'nested'
                    },
                    'captioned_image': {
                        'properties': {'caption': {'type': 'text'},
                                        'image': {'type': 'text'}
                                    }}}}}}

        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            return backend.get_index_info(config=self.config, index_name=self.index_name_1)
        
        index_info = run()
        assert index_info.search_model_name is None
        

    def test_get_cluster_indices(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        cluster_indices = backend.get_cluster_indices(config=self.config)
        assert '.opendistro_security' not in cluster_indices
        assert isinstance(cluster_indices, set)
        assert self.index_name_1 in cluster_indices

    def test_get_cluster_indices_mocked(self):
        mock__get = mock.MagicMock()
        mock__get.return_value = {
            '.opendistro_security': {'aliases': {}},
            'my-test-index-99': {'aliases': {}},
            'security-auditlog-2023.06.15': {'aliases': {}},
            'security-auditlog-2023.06.16': {'aliases': {}},
            'security-auditlog-2023.06.20': {'aliases': {}},
            'test_index': {'aliases': {}},
            '.kibana': {'aliases': {}},
            '.kibana_1': {'aliases': {}},
        }

        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            return backend.get_cluster_indices(config=self.config)
        cluster_indices = run()
        assert cluster_indices == {'my-test-index-99', 'test_index'}
        assert isinstance(cluster_indices, set)

    def test_add_customer_field_properties_defaults_lucene(self):
        mock_config = copy.deepcopy(self.config)
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config, index_name=self.index_name_1)
        
        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        retrieved_settings = settings[self.index_name_1]["mappings"]["_meta"][enums.IndexSettingsField.index_settings]

        # check meta has lucene engine
        assert retrieved_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.ann_parameters][enums.IndexSettingsField.ann_engine] \
            == "lucene"

        # check mappings has lucene engine
        params = settings[self.index_name_1]['mappings']['properties']['__chunks']['properties'][enums.TensorField.marqo_knn_field]['method']
        assert params['engine'] == 'lucene'
        
    def test_add_customer_field_properties_default_ann_parameters(self):
        mock_config = copy.deepcopy(self.config)
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config, index_name=self.index_name_1)
        
        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        retrieved_settings = settings[self.index_name_1]["mappings"]["_meta"][enums.IndexSettingsField.index_settings]

        # check ann parameters in meta are correct
        assert retrieved_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.ann_parameters] \
            == get_default_ann_parameters()

        # check ann parameters in mappings are correct
        params = settings[self.index_name_1]['mappings']['properties']['__chunks']['properties'][enums.TensorField.marqo_knn_field]['method']
        assert params \
            == get_default_ann_parameters()

    def test_add_customer_field_properties_index_ann_parameters(self):
        mock_config = copy.deepcopy(self.config)
        mock__put = mock.MagicMock()

        custom_settings = {
            enums.IndexSettingsField.index_defaults: {
                enums.IndexSettingsField.ann_parameters: {
                    enums.IndexSettingsField.ann_method_parameters: {
                        enums.IndexSettingsField.hnsw_ef_construction: 1,
                        enums.IndexSettingsField.hnsw_m: 2
                    }
                }
            }   
        }
        tensor_search.create_vector_index(
            config=mock_config,
            index_name=self.index_name_1,
            index_settings=custom_settings
        )

        settings = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}/_mapping",
            verify=False
        ).json()
        retrieved_settings = settings[self.index_name_1]["mappings"]["_meta"][enums.IndexSettingsField.index_settings]

        # check ann parameters in meta are correct
        assert custom_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.ann_parameters][enums.IndexSettingsField.ann_method_parameters] \
            == retrieved_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.ann_parameters][enums.IndexSettingsField.ann_method_parameters]

        # check ann parameters in mappings are correct
        params = settings[self.index_name_1]['mappings']['properties']['__chunks']['properties'][enums.TensorField.marqo_knn_field]['method']
        assert params['parameters'] \
            == custom_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.ann_parameters][enums.IndexSettingsField.ann_method_parameters]

    def test__remove_system_indices(self):
        index_names = ['.kibana', 'my-index', '.opendistro_security', 'some-other-index', '.kibana-100']
        assert backend._remove_system_indices(index_names) == {'my-index', 'some-other-index'}

    def test__remove_system_indices_empty(self):
        index_names = []
        assert backend._remove_system_indices(index_names) == set()

    def test__remove_system_indices_only_system_indices(self):
        index_names = ['.kibana', '.opendistro_security', '.kibana-100']
        assert backend._remove_system_indices(index_names) == set()

    def test__remove_system_indices_non_list_input(self):
        index_names = ('.kibana', 'my-index', '.opendistro_security', 'some-other-index', '.kibana-100')
        assert backend._remove_system_indices(index_names) == {'my-index', 'some-other-index'}

    def test__remove_system_indices_case_sensitivity(self):
        index_names = ['.Kibana', 'My-Index', '.Opendistro_Security', 'Some-Other-Index', '.Kibana-100']
        assert backend._remove_system_indices(index_names) == set(index_names)
