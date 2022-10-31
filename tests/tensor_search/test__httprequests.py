import requests
from tests.marqo_test import MarqoTestCase
from marqo import _httprequests
from unittest import mock
from marqo.tensor_search import tensor_search
from marqo.errors import (
    IndexNotFoundError, TooManyRequestsError
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
        mock_post = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 429
        mock_response.json = lambda: '{"a":"b"}'
        mock_post.return_value = mock_response
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        @mock.patch('requests.post', mock_post)
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_post, requests.get, requests.put})
        def run():
            try:
                res = tensor_search.add_documents(
                    config=self.config, index_name=self.index_name_1,
                    docs=[{"some ": "doc"}], auto_refresh=True
                )
                raise AssertionError
            except TooManyRequestsError:
                pass
            return True
        assert run()

