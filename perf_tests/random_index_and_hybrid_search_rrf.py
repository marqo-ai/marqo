from __future__ import annotations

import random
import os

from locust import events, task, between, run_single_user
from locust.env import Environment
from wonderwords import RandomSentence, RandomWord
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

"""
Performance test on Hybrid RRF search.
Does NOT use global score modifiers in search, but adds it to documents.
"""

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')


class AddDocToStructuredIndexUser(MarqoLocustHttpUser):
    fixed_count = 1
    wait_time = between(1, 2)

    @task
    def add_docs(self):
        # Generate random documents batch (5-10 docs) with random length description of 1-5 sentences
        s = "this is a random sentence."
        random_docs = [{
            'title': s,
            'description': ' '.join([s for j in range(i)]),
            'mult_field': i,
            'add_field': i
        } for i in range(10)]

        self.client.index(INDEX_NAME).add_documents(documents=random_docs)


class SearchUser(MarqoLocustHttpUser):
    wait_time = between(1, 2)
    w = RandomWord()

    @task
    def search(self):
        # Random search query to retrieve first 20 results
        self.client.index(INDEX_NAME).search(
            q=' '.join(self.w.random_words(amount=5)),
            search_method='HYBRID',
            hybrid_parameters={
                'retrievalMethod': 'disjunction',
                'rankingMethod': 'rrf'
            },
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
        marqo_client.create_index(
            INDEX_NAME,

            settings_dict={
                "type": "structured",
                "model": os.getenv('MARQO_INDEX_MODEL_NAME', 'hf/e5-base-v2'),
                "allFields": [
                    {"name": "title", "type": "text", "features": ["lexical_search"]},
                    {"name": "description", "type": "text", "features": ["lexical_search"]},
                    {"name": "mult_field", "type": "int", "features": ["score_modifier"]},
                    {"name": "add_field", "type": "int", "features": ["score_modifier"]}
                ],
                "tensorFields": ['title', 'description']
            }
        )


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
    run_single_user(AddDocToStructuredIndexUser)
