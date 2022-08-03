import unittest
from unittest import mock
from marqo import config, client
from marqo import enums
from marqo.errors import MarqoApiError
from tests.marqo_test import MarqoTestCase


class TestConfig(MarqoTestCase):

    def setUp(self) -> None:
        self.default_client = client.Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.default_client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.default_client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

