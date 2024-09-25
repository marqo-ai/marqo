from __future__ import annotations

import random
import os

from locust import events, task, between, run_single_user
from locust.env import Environment
from wonderwords import RandomSentence, RandomWord
import requests
import marqo
import pandas as pd

from common.marqo_locust_http_user import MarqoLocustHttpUser
from functools import partial


INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')

# Create docs
NUMBER_OF_DOCS = 100
vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
vocab = requests.get(vocab_source).text.splitlines()
print("Extracted vocab from mit.edu")

df_urls = pd.read_csv("image_urls.csv")

# 835 images, because 165 docs are bad IDs
bad_ids = [7, 17, 29, 31, 34, 36, 38, 44, 50, 55, 59, 67, 69, 73, 75, 77, 80, 82, 84, 93, 103, 112, 117, 119,
           124, 132, 135,
           141, 147, 151, 154, 163, 169, 177, 187, 194, 200, 204, 206, 212, 217, 239, 243, 245, 248, 252, 258,
           260, 263, 268,
           286, 295, 302, 309, 314, 326, 328, 333, 335, 338, 349, 353, 355, 361, 368, 372, 376, 383, 395, 399,
           409, 417, 420,
           437, 442, 446, 448, 450, 458, 461, 468, 473, 475, 483, 490, 500, 512, 520, 522, 526, 528, 533, 535,
           555, 557, 559,
           562, 566, 569, 571, 573, 577, 580, 583, 586, 588, 593, 597, 604, 637, 648, 654, 656, 693, 711, 717,
           728, 737, 741,
           743, 746, 747, 750, 752, 761, 780, 793, 797, 803, 805, 817, 819, 822, 827, 829, 841, 843, 849, 851,
           853, 860, 870,
           875, 877, 883, 887, 894, 900, 903, 908, 911, 915, 925, 933, 950, 953, 956, 964, 966, 968, 972, 975,
           984, 988, 999]

docs_list = []
csv_idx = 0  # Will go from 0 to 999

for doc_idx in range(NUMBER_OF_DOCS):
    doc = {
        "_id": f"doc{doc_idx}",
        "title": f"title: {' '.join(random.choices(population=vocab, k=8))}",
        "tags": f"tags: {', '.join(random.choices(population=vocab, k=8))}",
    }

    # Add 5 images to every doc
    images_added = 0
    while images_added < 5:
        # Skip any bad links
        if csv_idx in bad_ids:
            pass
        else:
            doc[f"image_{images_added + 1}"] = df_urls.loc[csv_idx]["url"]
            images_added += 1
        csv_idx += 1

    docs_list.append(doc)

print(f"Ended with csv_idx: {csv_idx}")
print(f"Docs list created with {len(docs_list)} docs.")

print(docs_list)
class AddDocsToStructuredIndexUser(MarqoLocustHttpUser):
    fixed_count = 2
    wait_time = between(1, 2)
    @task
    def add_docs(self):
        self.client.index(INDEX_NAME).add_documents(
            docs_list,
            device="cuda",
            client_batch_size=128,
        )


class SearchUser(MarqoLocustHttpUser):
    fixed_count = 3
    wait_time = between(1, 2)
    w = RandomWord()

    @task
    def search(self):
        # Random search query to retrieve first 20 results
        self.client.index(INDEX_NAME).search(
            q="https://image.lexica.art/full_jpg/00004467-fdef-41bc-bc73-20c68444a024",
            search_method='TENSOR',
            limit=30,
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
        # TODO: Make structured, but we won't run this locally for now
        marqo_client.create_index(INDEX_NAME, model=os.getenv('MARQO_INDEX_MODEL_NAME',
                                                              'open_clip/ViT-L-14-336/openai'))


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
    run_single_user(AddDocsToStructuredIndexUser)
