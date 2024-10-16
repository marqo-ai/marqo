from typing import Callable

from locust import HttpUser
from locust.clients import HttpSession
import marqo
from marqo._httprequests import HttpRequests, HTTP_OPERATIONS, ALLOWED_OPERATIONS
from marqo.config import Config
from marqo.index import Index
import os
from urllib.parse import urlparse
import json as jsonlib

class MarqoLocustHttpUser(HttpUser):
    abstract = True

    def __init__(self, *args, **kwargs):
        return_telemetry = kwargs.pop("return_telemetry", False)
        add_doc_batch_mode = kwargs.pop("add_doc_batch_mode", "per_document")
        super().__init__(*args, **kwargs)
        session: HttpSession = self.client
        marqo_client = marqo.Client(url=self.host, api_key=os.getenv('MARQO_CLOUD_API_KEY'),
                                    return_telemetry=return_telemetry)
        marqo_client.http = MarqoLocustHttpRequests(session, marqo_client.config, add_doc_batch_mode)

        def _get_locust_enhanced_index(index_name: str):
            index = Index(marqo_client.config, index_name=index_name)
            index.http = MarqoLocustHttpRequests(session, marqo_client.config, add_doc_batch_mode)
            return index

        marqo_client.index = _get_locust_enhanced_index
        self.client = marqo_client


class MarqoLocustHttpRequests(HttpRequests):

    def __init__(self, session: HttpSession, config: Config, add_doc_batch_mode: str) -> None:
        super().__init__(config)

        def post_with_add_doc_batch_mode(url, data=None, json=None, **kwargs):
            path = urlparse(url).path
            if path.endswith('/documents'):  # add_documents
                request_body = jsonlib.loads(data)
                request_body['batchVectorisationMode'] = add_doc_batch_mode
                return session.post(url, data=jsonlib.dumps(request_body), json=json, **kwargs)

            return session.post(url, data=data, json=json, **kwargs)

        self.operation_mapping = {
            'delete': session.delete,
            'get': session.get,
            'post': post_with_add_doc_batch_mode,
            'put': session.put,
            'patch': session.patch,
        }

    def _operation(self, method: HTTP_OPERATIONS) -> Callable:
        if method not in ALLOWED_OPERATIONS:
            raise ValueError("{} not an allowed operation {}".format(method, ALLOWED_OPERATIONS))

        return self.operation_mapping[method]