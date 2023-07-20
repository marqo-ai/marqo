"""
This example uses the MultiEURLEX dataset.

Log from running:
Took 45 minutes on ml.g4dn.2xlarge
"""
# change this to 'cpu' if the machine you are running Marqo on doesn't have a
# Nvidia GPU
DEVICE = "cuda"

# import marqo:
from marqo import Client

# import the huggingface datasets package:
from datasets import load_dataset

# import other python packages
import datetime
import json
import pprint
import logging

# this will be the name of the index:
INDEX_NAME = "my-multilingual-index"

# this helps us see information about the HTTP requests
logging.basicConfig(level=logging.DEBUG)

# Create a new Marqo client:
mq = Client("http://localhost:8882")


def build_index():
    # Load the datasets. For this example we're just using the English and
    # Deutsch validation splits:
    dataset_en = load_dataset('multi_eurlex', 'en', split="validation")
    dataset_de = load_dataset('multi_eurlex', 'de', split="validation")

    # record the start time:
    t0 = datetime.datetime.now()

    # Create the index. The model we're using is multilingual:
    mq.create_index(index_name=INDEX_NAME, model='stsb-xlm-r-multilingual')

    # Let's break up large documents to make it easier to search:
    MAX_TEXT_LENGTH = 100000

    for ds, lang in [(dataset_en, "en"), (dataset_de, "de")]:
        num_docs_in_dataset = len(ds)

        for ii, doc in enumerate(ds):
            dumped = json.dumps(doc)
            # we'll set the doc ID to be the document's hash
            doc_id = str(hash(dumped))

            text_length = len(doc['text'])
            split_size = MAX_TEXT_LENGTH//2
            # break up the text of large documents:
            if text_length > MAX_TEXT_LENGTH:
                text_splits = [doc['text'][i: i + split_size] for i in range(0, text_length, split_size)]
            else:
                text_splits = [doc['text']]

            for i, sub_doc in enumerate(text_splits):
                # if a document is broken up, add the text's index to the end of the document:
                qualified_id = f"{doc_id}.{i}" if len(text_splits) > 1 else doc_id
                # create a dict to be posted
                to_post = dict(
                    [(k, v) if k != "labels" else (k, str(v)) for k, v in doc. items() if k != 'text']
                    + [("_id", qualified_id), ("language", lang), ('text', sub_doc)]
                )
                print(f"doc number {ii} out of {num_docs_in_dataset} docs in dataset {lang}. "
                      f"_id: {qualified_id}, celex_id: {doc['celex_id']}, "
                      f"json to send size: {len(json.dumps(to_post))}")
                # Index the document. The device is set to 'cuda' to take
                # advantage of the machine's GPU. If you don't have a GPU,
                # change this argument to 'cpu'.
                # We set auto_refresh to False which is optimal for indexing
                # a lot of documents.
                mq.index(index_name=INDEX_NAME).add_documents(
                    documents=[to_post], device=DEVICE, auto_refresh=False,
                    tensor_fields=["language", "text", "labels"]
                )
    t1 = datetime.datetime.now()
    print(f"finished indexing. Started at {t0}. Finished at {t1}. Took {t1 - t0}")


def search(q):
    # Set searchable_attributes to 'text', which ensures that Marqo just
    # searches the 'text' field
    result = mq.index(INDEX_NAME).search(q=q, searchable_attributes=['text'])
    # Just print out the highlights, which makes the output easier to read
    for res in result["hits"]:
        pprint.pprint(res["_highlights"])


# After you finishing indexing, comment out the following line to prevent going through
# the whole indexing process again.
build_index()

# Replace 'my_search_query' with whatever text you want to search. In English or Deutsch!
my_search_query = "Laws about the fishing industry"
search(my_search_query)
