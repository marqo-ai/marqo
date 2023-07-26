from unittest import mock
import unittest

from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from marqo.errors import IndexNotFoundError
from marqo.tensor_search.utils import calculate_health_status


class TestHealthCheck(MarqoTestCase):
    def setUp(self) -> None:
        self.index_name = "health-check-index"

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(index_name=self.index_name, config=self.config)
        except IndexNotFoundError as e:
            pass

    def test_health_check(self):
        health_check_status = tensor_search.check_health(self.config)
        assert 'backend' in health_check_status
        assert 'status' in health_check_status['backend']
        assert 'status' in health_check_status

    def test_health_check_red_backend(self):
        mock__get = mock.MagicMock()
        statuses_to_check = ['red', 'yellow', 'green']

        for status in statuses_to_check:
            mock__get.return_value = {
                'status': status
            }
            @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
            def run():
                health_check_status = tensor_search.check_health(self.config)
                assert health_check_status['status'] == status
                assert health_check_status['backend']['status'] == status
                return True
            assert run()

    def test_health_check_unknown_backend_response(self):
        mock__get = mock.MagicMock()
        mock__get.return_value = dict()
        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            health_check_status = tensor_search.check_health(self.config)
            assert health_check_status['status'] == 'red'
            assert health_check_status['backend']['status'] == 'red'
            return True
        assert run()

    def test_index_health_check(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
        assert 'backend' in health_check_status
        assert 'status' in health_check_status['backend']
        assert 'status' in health_check_status

    def test_index_health_check_red_backend(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        mock__get = mock.MagicMock()
        statuses_to_check = ['red', 'yellow', 'green']

        for status in statuses_to_check:
            mock__get.return_value = {
                'status': status
            }
            @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
            def run():
                health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
                assert health_check_status['status'] == status
                assert health_check_status['backend']['status'] == status
                return True
            assert run()

    def test_index_health_check_path(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        with mock.patch("marqo._httprequests.HttpRequests.get") as mock_get:
            tensor_search.check_index_health(index_name=self.index_name, config=self.config)
            args, kwargs = mock_get.call_args
            self.assertIn(f"_cluster/health/{self.index_name}", kwargs['path'])

    def test_index_health_check_unknown_backend_response(self):
        mock__get = mock.MagicMock()
        mock__get.return_value = dict()

        # Ensure the index does not exist
        with self.assertRaises(IndexNotFoundError):
            tensor_search.delete_index(index_name=self.index_name, config=self.config)
        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
            assert health_check_status['status'] == 'red'
            assert health_check_status['backend']['status'] == 'red'
            return True
        assert run()


class TestCalculateStatus(unittest.TestCase):
    def test_status_green(self):
        response = {'status': 'green'}
        result = calculate_health_status(response)
        self.assertEqual(result, ('green', 'green'))

    def test_status_yellow(self):
        response = {'status': 'yellow'}
        result = calculate_health_status(response)
        self.assertEqual(result, ('yellow', 'yellow'))

    def test_status_red(self):
        response = {'status': 'red'}
        result = calculate_health_status(response)
        self.assertEqual(result, ('red', 'red'))

    def test_no_status_in_response(self):
        response = {'something_else': 'random'}
        result = calculate_health_status(response)
        self.assertEqual(result, ('red', 'red'))

    def test_no_response(self):
        result = calculate_health_status(None)
        self.assertEqual(result, ('red', 'red'))

