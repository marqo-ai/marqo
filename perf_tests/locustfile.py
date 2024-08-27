from __future__ import annotations

from locust import events, task, between, run_single_user
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = "locust-test"


class IndexingUser(MarqoLocustHttpUser):
    fixed_count = 1
    wait_time = between(1, 2)

    @task
    def add_docs(self):
        self.client.index(INDEX_NAME).add_documents(documents=[
            {'title': 'hello'},
            {'title': 'hello, world'},
            {'title': 'hello, my friend'},
            {'title': 'hello, doggy'},
            {'title': 'hello, kitty'},
        ], tensor_fields=['title'])


class SearchUser(MarqoLocustHttpUser):
    wait_time = between(1, 2)

    @task
    def search(self):
        self.client.index(INDEX_NAME).search('hello')


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    marqo_client = marqo.Client(url="http://localhost:8882")
    marqo_client.create_index(INDEX_NAME, model='hf/e5-base-v2')
    marqo_client.index(INDEX_NAME).add_documents(documents=[
        {'title': 'hello'},
        {'title': 'hello, world'},
        {'title': 'hello, my friend'},
        {'title': 'hello, doggy'},
        {'title': 'hello, kitty'},
    ], tensor_fields=['title'])


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    marqo_client = marqo.Client(url="http://localhost:8882")
    marqo_client.delete_index(INDEX_NAME)


# @events.request.add_listener
# def on_request(name, response, exception, **kwargs):
#     """
#     Event handler that get triggered on every request
#     """
#     # print out processing time for each request
#     print(name,  response.json()['processingTimeMs'])


if __name__ == "__main__":
    run_single_user(IndexingUser)
