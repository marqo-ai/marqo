import requests
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from unittest import mock
from marqo.tensor_search import tensor_search
from marqo.errors import (
    IndexNotFoundError, TooManyRequestsError, DiskWatermarkBreachError
)

class Test_HttpRequests(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_too_many_reqs_error(self):
        # Generic 429, we assume the cause is TooManyRequestsError
        # TODO: OpenSearch seems to raise 429 for any requestRejected: https://opensearch.org/docs/2.7/data-prepper/pipelines/configuration/sources/http-source/
        # Therefore, we should use response.error.type instead of response.status_code
        mock_post = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 429
        mock_response.json = lambda: {'error': {'type': 'too_many_requests'}} # TODO: confirm if this is really what opensearch returns
        mock_post.return_value = mock_response
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        @mock.patch('requests.post', mock_post)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, requests.get, requests.put})
        def run():
            try:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        docs=[{"some ": "doc"}], auto_refresh=True, device="cpu"
                    )
                )
                raise AssertionError
            except TooManyRequestsError:
                pass
            return True
        assert run()
    
    def test_opensearch_cluster_block_exception(self):
        mock_post = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 429
        mock_response.json = lambda: \
        {'error': 
            {'reason': 'index [TEST-INDEX] blocked by: '
                        '[TOO_MANY_REQUESTS/12/disk usage exceeded flood-stage '
                        'watermark, index has read-only-allow-delete block];',
            'root_cause': [{'reason': 'index [TEST-INDEX] blocked by: '
                                        '[TOO_MANY_REQUESTS/12/disk usage '
                                        'exceeded flood-stage watermark, index '
                                        'has read-only-allow-delete block];',
                            'type': 'cluster_block_exception'}],
            'type': 'cluster_block_exception'},
            'status': 429
        }
        mock_post.return_value = mock_response
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        @mock.patch('requests.post', mock_post)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, requests.get, requests.put})
        def run():
            try:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        docs=[{"some ": "doc"}], auto_refresh=True, device="cpu"
                    )
                )
                raise AssertionError
            except DiskWatermarkBreachError as e:
                print(e.message)
            return True
        assert run()
        


