import json
import pprint
from unittest import mock

import requests
from numpy.ma import copy

from marqo.tensor_search.enums import IndexSettingsField
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.tensor_search import tensor_search
from marqo.tensor_search import configs
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField


class TestHealthCheck(MarqoTestCase):

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
