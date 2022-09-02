import unittest
from marqo.tensor_search.utils import construct_authorized_url


class MarqoTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        local_opensearch_settings = {
            "url": 'https://localhost:9200',
            "main_user": "admin",
            "main_password": "admin"
        }
        s2search_settings = {}
        cls.client_settings = local_opensearch_settings
        cls.authorized_url = construct_authorized_url(
            url_base=cls.client_settings["url"],
            username=cls.client_settings["main_user"],
            password=cls.client_settings["main_password"]
        )
