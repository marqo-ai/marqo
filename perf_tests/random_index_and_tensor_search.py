from __future__ import annotations

import random

from locust import events, task, between, run_single_user
from wonderwords import RandomSentence, RandomWord
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

# TODO make this configurable
INDEX_NAME = "locust-test"


class IndexingUser(MarqoLocustHttpUser):
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


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    # TODO make the marqo client configurable, only create index for local tests
    marqo_client = marqo.Client(url="http://localhost:8882")
    # TODO make the model configurable as well
    marqo_client.create_index(INDEX_NAME, model='hf/e5-base-v2')


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
    run_single_user(SearchUser)
