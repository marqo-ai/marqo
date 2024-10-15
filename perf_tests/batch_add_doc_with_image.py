from __future__ import annotations

import json
import os

from locust import events, task, between, run_single_user
from wonderwords import RandomSentence

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')
telemetry_file = open('telemetry.jsonl', 'w')

class AddDocToUnstructuredIndexUser(MarqoLocustHttpUser):

    def __init__(self, *args, **kwargs):
        super().__init__(return_telemetry=True, *args, **kwargs)

    wait_time = between(1, 2)
    s = RandomSentence()

    @task
    def add_docs(self):
        self.client.index(INDEX_NAME).add_documents(
            documents=[doc for docs in [[
                {
                    "_id": f"example_doc_{i}1",
                    "title": "Man riding a horse",
                    "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"
                },
                {
                    "_id": f"example_doc_{i}2",
                    "title": "Flying Plane",
                    "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
                },
                {
                    "_id": f"example_doc_{i}3",
                    "title": "Traffic light",
                    "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg"
                },
                {
                    "_id": f"example_doc_{i}4",
                    "title": "Red Bus",
                    "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg"
                },
            ] for i in range(int(32 / 4))] for doc in docs],
            tensor_fields=["title"],
            # mappings={"my_multi_modal_field": {"type": "multimodal_combination",
            #                                    "weights": {"title": 0.5, "image": 0.8}}}
        )


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
    run_single_user(AddDocToUnstructuredIndexUser)
