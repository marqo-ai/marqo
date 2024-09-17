from typing import Callable

from locust import HttpUser
from locust.clients import HttpSession
import marqo
from marqo._httprequests import HttpRequests, HTTP_OPERATIONS, ALLOWED_OPERATIONS
from marqo.config import Config
from marqo.index import Index
import os

class MarqoLocustHttpUser(HttpUser):
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        session: HttpSession = self.client
        marqo_client = marqo.Client(url=self.host, api_key=os.getenv('MARQO_CLOUD_API_KEY'))
        marqo_client.http = MarqoLocustHttpRequests(session, marqo_client.config)

        def _get_locust_enhanced_index(index_name: str):
            index = Index(marqo_client.config, index_name=index_name)
            index.http = MarqoLocustHttpRequests(session, marqo_client.config)
            return index

        marqo_client.index = _get_locust_enhanced_index
        self.client = marqo_client


class MarqoLocustHttpRequests(HttpRequests):

    def __init__(self, session: HttpSession, config: Config) -> None:
        super().__init__(config)
        self.operation_mapping = {
            'delete': session.delete,
            'get': session.get,
            'post': session.post,
            'put': session.put,
            'patch': session.patch,
        }

    def _operation(self, method: HTTP_OPERATIONS) -> Callable:
        if method not in ALLOWED_OPERATIONS:
            raise ValueError("{} not an allowed operation {}".format(method, ALLOWED_OPERATIONS))

        return self.operation_mapping[method]