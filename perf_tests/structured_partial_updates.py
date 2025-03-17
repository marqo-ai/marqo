from __future__ import annotations

import random
import os

from locust import events, task, between, run_single_user
from locust.env import Environment
from wonderwords import RandomWord
import marqo

from common.marqo_locust_http_user import MarqoLocustHttpUser

INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-structured-test')


class StructuredUpdateUser(MarqoLocustHttpUser):
    wait_time = between(1, 2)
    w = RandomWord()

    @task
    def update_structured_fields(self):
        # Update non-tensor fields in a structured index
        doc_id = f"doc_{random.randint(0, 99)}"  # Random doc from 100 docs
        update_doc = {
            '_id': doc_id,
            'rating': random.randint(1, 5),
            'tags': random.sample(['new', 'sale', 'clearance', 'limited', 'featured', 'bestseller', 'seasonal'], random.randint(1, 3)),
            'in_stock': random.choice([True, False]),
            'price': round(random.uniform(10.0, 1000.0), 2),
            'inventory_counts': {
                'warehouse_1': random.randint(0, 100),
                'warehouse_2': random.randint(0, 50),
                'warehouse_3': random.randint(0, 75)
            },
            'price_history': {
                'jan': round(random.uniform(10.0, 1000.0), 2),
                'feb': round(random.uniform(10.0, 1000.0), 2),
                'mar': round(random.uniform(10.0, 1000.0), 2)
            }
        }
        
        self.client.index(INDEX_NAME).update_documents(
            documents=[update_doc]
        )


@events.init.add_listener
def on_test_start(environment: Environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        # Create structured index if running locally
        marqo_client = marqo.Client(url=host)
        marqo_client.create_index(index_name=INDEX_NAME,
                                 type="structured",
                                 all_fields=[{"name": "category", "type": "text"}, {"name": "tags", "type": "array<text>"}, {"name": "in_stock", "type": "bool"}, {"name": "price", "type": "float"}, {"name": "inventory_counts", "type": "map<text, int>"}, {"name": "price_history", "type": "map<text, float>"}],
                                 tensor_fields=["category"])

        # Add initial documents
        initial_docs = []
        w = RandomWord()
        for i in range(100):
            doc = {
                '_id': f"doc_{i}",
                'category': random.choice(['electronics', 'books', 'clothing', 'food']),
                'tags': random.sample(['new', 'sale', 'clearance', 'limited', 'featured', 'bestseller', 'seasonal'], random.randint(1, 3)),
                'in_stock': random.choice([True, False]),
                'price': round(random.uniform(10.0, 1000.0), 2),
                'inventory_counts': {
                    'warehouse_1': random.randint(0, 100),
                    'warehouse_2': random.randint(0, 50),
                    'warehouse_3': random.randint(0, 75)
                },
                'price_history': {
                    'jan': round(random.uniform(10.0, 1000.0), 2),
                    'feb': round(random.uniform(10.0, 1000.0), 2),
                    'mar': round(random.uniform(10.0, 1000.0), 2)
                }
            }
            initial_docs.append(doc)
        
        marqo_client.index(INDEX_NAME).add_documents(documents=initial_docs)


@events.quitting.add_listener
def on_test_stop(environment, **kwargs):
    host = environment.host
    local_run = host == 'http://localhost:8882'
    if local_run:
        marqo_client = marqo.Client(url=host)
        marqo_client.delete_index(INDEX_NAME)


if __name__ == "__main__":
    run_single_user(StructuredUpdateUser)
