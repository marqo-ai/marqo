from tests.marqo_test import MarqoTestCase
from marqo.config import Config

class VespaTestCase(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configure_request_metrics()

    @classmethod
    def tearDownClass(cls):
        pass