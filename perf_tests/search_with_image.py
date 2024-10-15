from __future__ import annotations

import json
import os
import random

from locust import events, task, between, run_single_user

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')
telemetry_file = open('telemetry.jsonl', 'w')


class SearchUnstructuredIndexUser(MarqoLocustHttpUser):

    def __init__(self, *args, **kwargs):
        super().__init__(return_telemetry=True, *args, **kwargs)

    wait_time = between(1, 2)

    all_text_queries = [
        'travel with plane',
        'travel',
        'horse',
        'ride',
        'green light',
        'take a bus',
        'some unrelated stuff',
    ],

    image_queries = [
        'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg',
        'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg',
        'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg',
        'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg',
        'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg',
    ]

    @task
    def search(self):
        self.client.index(INDEX_NAME).search(q=random.choice(self.all_text_queries))


@events.quitting.add_listener
def on_test_stop(environment, **kwargs):
    telemetry_file.close()


@events.request.add_listener
def on_request(name, response, exception, **kwargs):
    """
    Event handler that get triggered on every request
    """
    telemetry_file.write(json.dumps(response.json()['telemetry']['timesMs']) + '\n')


if __name__ == "__main__":
    run_single_user(SearchUnstructuredIndexUser)
