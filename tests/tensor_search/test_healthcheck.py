import json
import pprint
import requests
from marqo.tensor_search.enums import IndexSettingsField
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError
from marqo.tensor_search import tensor_search
from marqo.tensor_search import configs
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField


class TestHealthCheck(MarqoTestCase):

    def test_health_check(self):
        pprint.pprint(
            tensor_search.check_health(self.config)
        )
