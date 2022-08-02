import unittest
from unittest import mock
from marqo import config, client
from marqo import enums
from marqo.errors import MarqoApiError

class TestConfig(unittest.TestCase):

    def setUp(self) -> None:
        self.default_client = client.Client(url='https://localhost:9200', main_user="admin", main_password="admin")
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

