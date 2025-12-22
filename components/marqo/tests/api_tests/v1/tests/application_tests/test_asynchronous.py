import random
import sys
import threading
import time
import uuid

import requests

from tests import marqo_test

sys.setswitchinterval(0.005)


class TestAsync (marqo_test.MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.standard_structured_index_name = "structured_standard" + str(uuid.uuid4()).replace('-', '')
        cls.standard_unstructured_index_name = "unstructured_standard" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.standard_unstructured_index_name,
                "type": "unstructured"
            },
            {
                "indexName": cls.standard_structured_index_name,
                "type": "structured",
                "allFields": [{"name": "text_field_1", "type": "text"},
                              {"name": "text_field_2", "type": "text"}],
                "tensorFields": ["text_field_1", "text_field_2"]
            }
        ])

        cls.indexes_to_delete = [cls.standard_unstructured_index_name, cls.standard_structured_index_name]

    def test_async(self):
        for index_name in [self.standard_unstructured_index_name, self.standard_structured_index_name]:
            with self.subTest(f"test async for {index_name}"):
                num_docs = 500

                vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"

                vocab = requests.get(vocab_source).text.splitlines()

                d1 = {
                    "text_field_1": "Just Your Average Doc",
                    "text_field_2": "this is a solid doc",
                    "_id": "56"
                }
                tensor_fields = ["text_field_1", "text_field_2"] if \
                    index_name == self.standard_unstructured_index_name else None
                self.client.index(index_name).add_documents([d1], tensor_fields=tensor_fields)
                assert self.client.index(index_name).get_stats()['numberOfDocuments'] == 1

                def significant_ingestion():
                    docs = [{"text_field_1": " ".join(random.choices(population=vocab, k=10)),
                                  "text_field_2": " ".join(random.choices(population=vocab, k=25)),
                                  } for _ in range(num_docs)]
                    self.client.index(index_name).add_documents(documents=docs, client_batch_size=1,
                                                                tensor_fields=tensor_fields)

                cache_update_thread = threading.Thread(
                    target=significant_ingestion)
                cache_update_thread.start()
                time.sleep(3)
                assert cache_update_thread.is_alive()
                assert self.client.index(index_name).get_stats()['numberOfDocuments'] > 1
                assert cache_update_thread.is_alive()
                assert self.client.index(index_name).get_stats()['numberOfDocuments'] < 251

                while cache_update_thread.is_alive():
                    time.sleep(1)

                assert self.client.index(index_name).get_stats()['numberOfDocuments'] == 501


