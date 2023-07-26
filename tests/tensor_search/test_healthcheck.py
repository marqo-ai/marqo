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
from marqo import errors
from marqo.tensor_search import health


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
    
    def test_convert_watermark_to_bytes(self):
        test_cases = [
            # byte watermarks (total_in_bytes is ignored)
            ("0b", "99999", 0),
            ("21b", "99999", 21),
            ("21B", "99999", 21),
            ("2.1B", "99999", 2.1),
            ("2.1 b", "99999", 2.1),
            ("2.1garbage1b", "99999", errors.InternalError),
            # kb/gb/mb/tb watermarks (total_in_bytes is ignored)
            ("0kb", "99999", 0),
            ("21kb", "99999", 21 * 1024),
            ("2.1MB", "99999", 2.1 * 1024 ** 2),
            ("21GB", "99999", 21 * 1024 ** 3),
            ("2.1 TB", "99999", 2.1 * 1024 ** 4),
            ("2.1garbagePB", "99999", errors.InternalError),
            ("2.1XB", "99999", errors.InternalError),

            # percentage watermarks
            ("0%", "1000", 1000),
            ("80%", "1000", 200),
            ("100%", "1000", 0),
            ("40.%", "1000", 600),
            ("0.5%", "1000", 995),
            ("1garbage2%", "1000", errors.InternalError),
            ("-1%", "1000", errors.InternalError),
            ("101%", "1000", errors.InternalError),
 
            # ratio watermarks
            ("0", "1000", 1000),
            (".80", "1000", 200),
            ("1.00", "1000", 0),
            ("0.4", "1000", 600),
            (".005", "1000", 995),
            ("0.1garbage2", "1000", errors.InternalError),
            ("-.01", "1000", errors.InternalError),
            ("1.01", "1000", errors.InternalError),

            # edge cases
            ("", "99999", errors.InternalError),
            (" ", "99999", errors.InternalError),
            (None, "99999", errors.InternalError),
        ]

        for watermark, total_in_bytes, expected in test_cases:
            try:
                result = health.convert_watermark_to_bytes(watermark, total_in_bytes)
                self.assertAlmostEqual(result, expected)
            except expected:
                pass
    

    def test_check_opensearch_disk_watermark_breach
