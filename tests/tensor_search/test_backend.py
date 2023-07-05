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

    def test_get_cluster_indices(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        cluster_indices = backend.get_cluster_indices(config=self.config)
        assert '.opendistro_security' not in cluster_indices
        assert isinstance(cluster_indices, set)
        assert self.index_name_1 in cluster_indices

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

