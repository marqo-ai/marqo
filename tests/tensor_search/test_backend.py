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
        @mock.patch("marqo._httprequests.HttpRequests.put", mock__put)
        def run():
            add_docs_caller(config=mock_config, docs=[{"f1": "doc"}, {"f2":"C"}],
                            index_name=self.index_name_1, auto_refresh=True)
            return True
        assert run()
        args, kwargs0 = mock__put.call_args_list[0]
        sent_dict = json.loads(kwargs0["body"])
        assert "lucene" == sent_dict["properties"][enums.TensorField.chunks
            ]["properties"][utils.generate_vector_name(field_name="f1")]["method"]["engine"]
    
    def test_add_customer_field_properties_default_ann_parameters(self):
        mock_config = copy.deepcopy(self.config)
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config, index_name=self.index_name_1)
        @mock.patch("marqo._httprequests.HttpRequests.put", mock__put)
        def run():
            add_docs_caller(config=mock_config, docs=[{"f1": "doc"}, {"f2":"C"}],
                                        index_name=self.index_name_1, auto_refresh=True)
            return True
        assert run()
        args, kwargs0 = mock__put.call_args_list[0]
        sent_dict = json.loads(kwargs0["body"])
        assert sent_dict["properties"][enums.TensorField.chunks]["properties"][utils.generate_vector_name(field_name="f1")]["method"] == get_default_ann_parameters()

    def test_add_customer_field_properties_index_ann_parameters(self):
        mock_config = copy.deepcopy(self.config)
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config,
            index_name=self.index_name_1,
            index_settings={
                enums.IndexSettingsField.index_defaults: {
                    enums.IndexSettingsField.ann_parameters: {
                        enums.IndexSettingsField.ann_method_parameters: {
                            enums.IndexSettingsField.hnsw_ef_construction: 1,
                            enums.IndexSettingsField.hnsw_m: 2
                        }
                    }
                }   
            }
        )
        @mock.patch("marqo._httprequests.HttpRequests.put", mock__put)
        def run():
            add_docs_caller(config=mock_config, docs=[{"f1": "doc"}, {"f2":"C"}],
                                        index_name=self.index_name_1, auto_refresh=True)
            return True
        assert run()
        args, kwargs0 = mock__put.call_args_list[0]
        sent_dict = json.loads(kwargs0["body"])
        assert sent_dict["properties"][enums.TensorField.chunks]["properties"][utils.generate_vector_name(field_name="f1")]["method"]['engine'] == "lucene"
        assert sent_dict["properties"][enums.TensorField.chunks]["properties"][utils.generate_vector_name(field_name="f1")]["method"]["parameters"] == {
                            enums.IndexSettingsField.hnsw_ef_construction: 1,
                            enums.IndexSettingsField.hnsw_m: 2
                        }

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
