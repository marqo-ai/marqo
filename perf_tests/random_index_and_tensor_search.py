from __future__ import annotations

import random
import os

from locust import events, task, between, run_single_user
from locust.env import Environment
from wonderwords import RandomSentence, RandomWord
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')


class AddDocToUnstructuredIndexUser(MarqoLocustHttpUser):
    fixed_count = 1
    wait_time = between(1, 2)
    s = RandomSentence()

    @task
    def add_docs(self):
        # Generate random documents batch (5-10 docs) with random length description of 1-5 sentences
        random_docs = [{
            'title': self.s.sentence(),
            'description': ' '.join([self.s.sentence() for j in range(random.randint(1, 5))])
        } for i in range(random.randint(5, 10))]

        self.client.index(INDEX_NAME).add_documents(documents=random_docs, tensor_fields=['title', 'description'])


class SearchUser(MarqoLocustHttpUser):
    wait_time = between(1, 2)
    w = RandomWord()

    @task
    def search(self):
        # Random search query to retrieve first 20 results
        self.client.index(INDEX_NAME).search(
            q=' '.join(self.w.random_words(amount=random.randint(1, 5))),
            search_method='TENSOR',
            limit=20,
            show_highlights=False,
            offset=0,
        )


@events.init.add_listener
def on_test_start(environment: Environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        # Create index if run local
        marqo_client = marqo.Client(url=host)
        marqo_client.create_index(INDEX_NAME, model=os.getenv('MARQO_INDEX_MODEL_NAME', 'hf/e5-base-v2'))


@events.quitting.add_listener
def on_test_stop(environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        marqo_client = marqo.Client(url=host)
        marqo_client.delete_index(INDEX_NAME)


# @events.request.add_listener
# def on_request(name, response, exception, **kwargs):
#     """
#     Event handler that get triggered on every request
#     """
#     # print out processing time for each request
#     print(name,  response.json()['processingTimeMs'])


if __name__ == "__main__":
    run_single_user(AddDocToUnstructuredIndexUser)
