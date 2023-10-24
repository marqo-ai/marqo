import time
import requests
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from unittest import mock
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import EnvVars, SearchMethod
from marqo._httprequests import HttpRequests
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
        self.bulk_retry_index_name_1 = "self.bulk_retry_index_name_1"
        self.bulk_retry_index_name_2 = "self.bulk_retry_index_name_2"

        for index_name in [self.index_name_1, self.bulk_retry_index_name_1, self.bulk_retry_index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=index_name)
            except IndexNotFoundError as s:
                pass
        self.httprequest_object = HttpRequests(self.config)

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
    
    def test_backoff_sleep_first_attempt(self):
        """Test backoff sleep for the first attempt with cap 5"""
        result = self.httprequest_object.calculate_backoff_sleep(0, 5)
        self.assertEqual(result, 0.01)  # Expected sleep time: 10ms

    def test_backoff_sleep_within_cap(self):
        """Test backoff sleep within the specified cap (cap 5)"""
        result = self.httprequest_object.calculate_backoff_sleep(3, 5)
        self.assertEqual(result, 0.08)  # Expected sleep time: 80ms (2^3 * 10ms)

    def test_backoff_sleep_reaches_cap(self):
        """Test backoff sleep reaches the specified cap (cap 5)"""
        result = self.httprequest_object.calculate_backoff_sleep(10, 5)
        self.assertEqual(result, 5.0)  # Expected sleep time equals the cap

    def test_backoff_sleep_with_zero_cap(self):
        """Test backoff sleep with a cap of 0 (should always return 0)"""
        result = self.httprequest_object.calculate_backoff_sleep(5, 0)
        self.assertEqual(result, 0.0)  # Expected sleep time is always 0 with cap 0

    def test_httprequest_success_request(self):
        mock_get = mock.MagicMock()
        mock_post = mock.MagicMock()
        mock_put = mock.MagicMock()
        mock_delete = mock.MagicMock()

        mock_allowed_operations = {mock_post, mock_get, mock_put, mock_delete}


        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: {'success': {'hits': ['test']}}
        mock_environ = {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('requests.post', mock_post)
        @mock.patch('requests.put', mock_put)
        @mock.patch('requests.delete', mock_delete)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', mock_allowed_operations)
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            for method in mock_allowed_operations:
                method.return_value = mock_response
                try:
                    res = self.httprequest_object.send_request(
                        http_method=method,
                        path="some_path",
                        body="some_body"
                    )
                except Exception as e:
                    return False
                assert method.call_count == 1
                assert res == {'success': {'hits': ['test']}}
            return True
        assert run()

    def test_httprequest_raw_send_request(self):
        mock_get = mock.MagicMock()
        mock_post = mock.MagicMock()
        mock_put = mock.MagicMock()
        mock_delete = mock.MagicMock()

        mock_allowed_operations = {mock_post, mock_get, mock_put, mock_delete}


        mock_response = requests.Response()
        mock_response.status_code = 500
        error_message = """HTTPSConnectionPool(host='internal-abcdefghijk-123456789.us-east-1.elb.amazonaws.com', port=9200):
Max retries exceeded with url: /my-test-index-1/_mapping (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)'))
"""
        mock_get.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_post.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_put.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_delete.side_effect = requests.exceptions.ConnectionError(error_message)
        mock_environ = {
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('requests.post', mock_post)
        @mock.patch('requests.put', mock_put)
        @mock.patch('requests.delete', mock_delete)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', mock_allowed_operations)
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            for method in mock_allowed_operations:
                try:
                    res = self.httprequest_object.send_request(
                        http_method=method,
                        path="some_path",
                        body="some_body",
                        max_retry_attempts=None,
                        max_retry_backoff_seconds=None
                    )
                    raise AssertionError
                except BackendCommunicationError as e:
                    assert e.code == "backend_communication_error"
                    assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                    assert "Max retries exceeded with url" in e.message
                    assert method.call_count == 1
            return True
        assert run()

    def test_httprequest_send_request_variable_default_retry_and_backoff(self):
        retry_backoff_list = [
            {'retry_attempts': 5, 'backoff_seconds': 3}, 
            {'retry_attempts': 3, 'backoff_seconds': 2},
            {'retry_attempts': 10, 'backoff_seconds': 5},
            {'retry_attempts': 7, 'backoff_seconds': 1}
        ]
        for mock_retry_pair in retry_backoff_list:
            mock_get = mock.MagicMock()
            mock_post = mock.MagicMock()
            mock_put = mock.MagicMock()
            mock_delete = mock.MagicMock()

            mock_allowed_operations = {mock_post, mock_get, mock_put, mock_delete}


            mock_response = requests.Response()
            mock_response.status_code = 500
            error_message = """HTTPSConnectionPool(host='internal-abcdefghijk-123456789.us-east-1.elb.amazonaws.com', port=9200):
    Max retries exceeded with url: /my-test-index-1/_mapping (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)'))
    """
            mock_get.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_post.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_put.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_delete.side_effect = requests.exceptions.ConnectionError(error_message)

            mock_environ = {
                "DEFAULT_MARQO_MAX_BACKEND_RETRY_ATTEMPTS": str(mock_retry_pair['retry_attempts']),
                "DEFAULT_MARQO_MAX_BACKEND_RETRY_BACKOFF": str(mock_retry_pair['backoff_seconds']),
                "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
            }
            @mock.patch('requests.get', mock_get)
            @mock.patch('requests.post', mock_post)
            @mock.patch('requests.put', mock_put)
            @mock.patch('requests.delete', mock_delete)
            @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', mock_allowed_operations)
            @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
            def run():
                for method in mock_allowed_operations:
                    try:
                        start_time = time.time()
                        res = self.httprequest_object.send_request(
                            http_method=method,
                            path="some_path",
                            body="some_body",
                        )
                        raise AssertionError
                    except BackendCommunicationError as e:
                        assert e.code == "backend_communication_error"
                        assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                        assert "Max retries exceeded with url" in e.message
                        assert method.call_count == mock_retry_pair['retry_attempts'] + 1 # +1 since the first call is not a retry
                        expected_total_wait_time = 0
                        for i in range(mock_retry_pair['retry_attempts']):
                            expected_total_wait_time += self.httprequest_object.calculate_backoff_sleep(i, mock_retry_pair['backoff_seconds'])
                        end_time = time.time() - start_time
                        assert expected_total_wait_time <= end_time # time should be <= once max retries reach
                return True
            assert run()

    def test_httprequest_send_request_variable_retry_and_backoff(self):
        retry_backoff_list = [
            {'retry_attempts': 5, 'backoff_seconds': 3}, 
            {'retry_attempts': 3, 'backoff_seconds': 2},
            {'retry_attempts': 10, 'backoff_seconds': 5},
            {'retry_attempts': 7, 'backoff_seconds': 1}
        ]
        for mock_retry_pair in retry_backoff_list:
            mock_get = mock.MagicMock()
            mock_post = mock.MagicMock()
            mock_put = mock.MagicMock()
            mock_delete = mock.MagicMock()

            mock_allowed_operations = {mock_post, mock_get, mock_put, mock_delete}


            mock_response = requests.Response()
            mock_response.status_code = 500
            error_message = """HTTPSConnectionPool(host='internal-abcdefghijk-123456789.us-east-1.elb.amazonaws.com', port=9200):
    Max retries exceeded with url: /my-test-index-1/_mapping (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)'))
    """
            mock_get.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_post.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_put.side_effect = requests.exceptions.ConnectionError(error_message)
            mock_delete.side_effect = requests.exceptions.ConnectionError(error_message)

            mock_environ = {
                "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
            }
            @mock.patch('requests.get', mock_get)
            @mock.patch('requests.post', mock_post)
            @mock.patch('requests.put', mock_put)
            @mock.patch('requests.delete', mock_delete)
            @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', mock_allowed_operations)
            @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
            def run():
                for method in mock_allowed_operations:
                    try:
                        start_time = time.time()
                        res = self.httprequest_object.send_request(
                            http_method=method,
                            path="some_path",
                            body="some_body",
                            max_retry_attempts=mock_retry_pair['retry_attempts'],
                            max_retry_backoff_seconds=mock_retry_pair['backoff_seconds']
                        )
                        raise AssertionError
                    except BackendCommunicationError as e:
                        assert e.code == "backend_communication_error"
                        assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                        assert "Max retries exceeded with url" in e.message
                        assert method.call_count == mock_retry_pair['retry_attempts'] + 1 # +1 since the first call is not a retry
                        expected_total_wait_time = 0
                        for i in range(mock_retry_pair['retry_attempts']):
                            expected_total_wait_time += self.httprequest_object.calculate_backoff_sleep(i, mock_retry_pair['backoff_seconds'])
                        end_time = time.time() - start_time
                        assert expected_total_wait_time <= end_time # time should be <= once max retries reach
                return True
                return True
            assert run()

    def test_opensearch_search_lexical_no_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.LEXICAL)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 1 #  should be called once since max_retry_attempts defaults to 0

            return True
        assert run()

    def test_opensearch_search_lexical_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            EnvVars.MARQO_MAX_BACKEND_SEARCH_RETRY_ATTEMPTS: str(3),
            EnvVars.MARQO_MAX_BACKEND_ADD_DOCS_RETRY_ATTEMPTS: str(3),
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.LEXICAL)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 4 # 4 since the first call is not a retry

            return True
        assert run()

    def test_opensearch_search_tensor_no_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.TENSOR)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 1 #  should be called once since max_retry_attempts defaults to 0

            return True
        assert run()

    def test_opensearch_search_tensor_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            EnvVars.MARQO_MAX_BACKEND_SEARCH_RETRY_ATTEMPTS: str(3),
            EnvVars.MARQO_MAX_BACKEND_ADD_DOCS_RETRY_ATTEMPTS: str(3),
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="cool match",
                search_method=SearchMethod.TENSOR)
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 4 # 4 since the first call is not a retry

            return True
        assert run()
    
    def test_opensearch_bulk_search_no_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.bulk_retry_index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.bulk_retry_index_name_2)

        res = tensor_search.search(
        config=self.config, index_name=self.bulk_retry_index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

        res = tensor_search.search(
        config=self.config, index_name=self.bulk_retry_index_name_2, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.bulk_search(
                    query=BulkSearchQuery(
                        queries=[
                            BulkSearchQueryEntity(index=self.bulk_retry_index_name_1, q="a test query", limit=2, searchMethod="LEXICAL"),
                            BulkSearchQueryEntity(index=self.bulk_retry_index_name_2, q="a test query", limit=2, searchMethod="TENSOR")
                        ]
                    ),
                    marqo_config=self.config
                )
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 1 #  should be called once since max_retry_attempts defaults to 0

            return True
        assert run()

    def test_opensearch_bulk_search_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.bulk_retry_index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.bulk_retry_index_name_2)

        res = tensor_search.search(
        config=self.config, index_name=self.bulk_retry_index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

        res = tensor_search.search(
        config=self.config, index_name=self.bulk_retry_index_name_2, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            EnvVars.MARQO_MAX_BACKEND_SEARCH_RETRY_ATTEMPTS: str(3),
            EnvVars.MARQO_MAX_BACKEND_ADD_DOCS_RETRY_ATTEMPTS: str(3),
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.get', mock_get)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {requests.post, mock_get, requests.put})
        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            try:
                res = tensor_search.bulk_search(
                    query=BulkSearchQuery(
                        queries=[
                            BulkSearchQueryEntity(index=self.bulk_retry_index_name_1, q="a test query", limit=2, searchMethod="LEXICAL"),
                            BulkSearchQueryEntity(index=self.bulk_retry_index_name_2, q="a test query", limit=2, searchMethod="TENSOR")
                        ]
                    ),
                    marqo_config=self.config
                )
                raise AssertionError
            except BackendCommunicationError as e:
                assert e.code == "backend_communication_error"
                assert e.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
                assert "Max retries exceeded with url" in e.message
                assert mock_get.call_count == 4 # 4 since the first call is not a retry

            return True
        assert run()

    def test_opensearch_add_docs_no_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.post', mock_post)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, requests.get, requests.put})
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
                assert mock_post.call_count == 1 #  should be called once since max_retry_attempts defaults to 0
            return True
        assert run()

    def test_opensearch_add_docs_retry(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        res = tensor_search.search(
        config=self.config, index_name=self.index_name_1, text="cool match",
        search_method=SearchMethod.LEXICAL,
        device='cpu') # populate index_meta_cache

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
            EnvVars.MARQO_MAX_BACKEND_SEARCH_RETRY_ATTEMPTS: str(3),
            EnvVars.MARQO_MAX_BACKEND_ADD_DOCS_RETRY_ATTEMPTS: str(3),
            "MARQO_BEST_AVAILABLE_DEVICE": "cpu"
        }

        @mock.patch('requests.post', mock_post)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, requests.get, requests.put})
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
                assert mock_post.call_count == 4 # 4 since the first call is not a retry
            return True
        assert run()