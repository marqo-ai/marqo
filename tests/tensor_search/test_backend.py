import copy
import json
import pprint

import requests
from marqo.tensor_search import enums, backend, utils
from marqo.tensor_search import tensor_search
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

    def test_add_customer_field_properties_nmslib_for_s2search(self):
        mock_config = copy.deepcopy(self.config)
        mock_config.cluster_is_s2search = True
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config, index_name=self.index_name_1)
        @mock.patch("marqo._httprequests.HttpRequests.put", mock__put)
        def run():
            tensor_search.add_documents(config=mock_config, docs=[{"f1": "doc"}, {"f2":"C"}],
                                        index_name=self.index_name_1, auto_refresh=True)
            return True
        assert run()
        args, kwargs0 = mock__put.call_args_list[0]
        sent_dict = json.loads(kwargs0["body"])
        assert "nmslib" == sent_dict["properties"][enums.TensorField.chunks
            ]["properties"][utils.generate_vector_name(field_name="f1")]["method"]["engine"]

    def test_add_customer_field_properties_defaults_lucene(self):
        mock_config = copy.deepcopy(self.config)
        mock_config.cluster_is_s2search = False
        mock__put = mock.MagicMock()

        tensor_search.create_vector_index(
            config=mock_config, index_name=self.index_name_1)
        @mock.patch("marqo._httprequests.HttpRequests.put", mock__put)
        def run():
            tensor_search.add_documents(config=mock_config, docs=[{"f1": "doc"}, {"f2":"C"}],
                                        index_name=self.index_name_1, auto_refresh=True)
            return True
        assert run()
        args, kwargs0 = mock__put.call_args_list[0]
        sent_dict = json.loads(kwargs0["body"])
        assert "lucene" == sent_dict["properties"][enums.TensorField.chunks
            ]["properties"][utils.generate_vector_name(field_name="f1")]["method"]["engine"]