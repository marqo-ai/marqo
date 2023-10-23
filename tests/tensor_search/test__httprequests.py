import requests
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from unittest import mock
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import EnvVars, SearchMethod
from marqo.errors import (
    IndexNotFoundError, TooManyRequestsError,
    DiskWatermarkBreachError, MarqoWebError, BackendCommunicationError
)
from http import HTTPStatus
import os

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
                # Disk breach is a 400 error
                assert e.code == "disk_watermark_breach_error"
                assert e.status_code == HTTPStatus.BAD_REQUEST
                assert "Marqo storage is full" in e.message

            return True
        assert run()
        
    def test_opensearch_search_retry(self):
        mock_post = mock.MagicMock()
        mock_get = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 500
        error_message = """HTTPSConnectionPool(host='internal-abcdefghijk-123456789.us-east-1.elb.amazonaws.com', port=9200):
Max retries exceeded with url: /my-test-index-1/_mapping (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)'))
"""
        mock_get.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_post.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_environ = {
            EnvVars.MARQO_OPENSEARCH_MAX_SEARCH_RETRY_ATTEMPTS: str(3),
            EnvVars.MARQO_OPENSEARCH_MAX_ADD_DOCS_RETRY_ATTEMPTS: str(3),
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.post', mock_post)
        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        docs=[{"some ": "doc"}], auto_refresh=True, device="cpu"
                    )
                )
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message

            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.LEXICAL)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message

            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.TENSOR)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
            return True
        assert run()