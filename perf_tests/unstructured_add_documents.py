from __future__ import annotations

import random
import os

from locust import events, task, between, run_single_user
from locust.env import Environment
from wonderwords import RandomSentence
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-add-docs-test')


class AddDocumentsUser(MarqoLocustHttpUser):
    wait_time = between(1, 2)
    s = RandomSentence()

    @task(2)
    def add_single_doc(self):
        # Add a single document with existing tensors
        doc_id = f"doc_{random.randint(0, 99)}"
        doc = {
            '_id': doc_id,  # Reuse existing document ID
            'title': self.s.sentence(),
            'description': ' '.join([self.s.sentence() for _ in range(random.randint(1, 3))])
        }
        
        self.client.index(INDEX_NAME).add_documents(
            documents=[doc],
            use_existing_tensors=True
        )


@events.init.add_listener
def on_test_start(environment: Environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        # Create index if running locally
        marqo_client = marqo.Client(url=host)
        settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": os.getenv('MARQO_INDEX_MODEL_NAME', 'hf/e5-base-v2'),
                "normalize_embeddings": True
            }
        }
        marqo_client.create_index(INDEX_NAME, settings_dict=settings)

        # Add initial documents
        s = RandomSentence()
        initial_docs = []
        for i in range(100):
            doc = {
                '_id': f"doc_{i}",
                'title': s.sentence(),
                'description': ' '.join([s.sentence() for _ in range(random.randint(1, 3))])
            }
            initial_docs.append(doc)
        
        marqo_client.index(INDEX_NAME).add_documents(
            documents=initial_docs,
            tensor_fields=['title', 'description']
        )


@events.quitting.add_listener
def on_test_stop(environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        marqo_client = marqo.Client(url=host)
        marqo_client.delete_index(INDEX_NAME)


if __name__ == "__main__":
    run_single_user(AddDocumentsUser)
