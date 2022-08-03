import requests

from marqo.client import Client
from marqo.errors import MarqoApiError
import unittest
import pprint
from tests.marqo_test import MarqoTestCase


class TestKaizanDemo(MarqoTestCase):
    """Test for Kaizan demo.

    Assumptions:
        - Local OpenSearch (not S2Search)
    """
    def setUp(self) -> None:
        client_0 = Client(**self.client_settings)
        for ix_name in ["cool-index-1", "my-first-index"]:
            try:
                client_0.delete_index(ix_name)
            except MarqoApiError as s:
                pass

    def test_demo(self):
        client = Client(**self.client_settings)
        client.index("cool-index-1").add_documents([
            {
                "Title": "The Legend of the River",
                "Description": "Once upon a time there was a cat who wore a hat. "
                               "What happens next is ridiculous. "
             },
            {
                "Title": "Top Places to Visit in Melbourne ",
                "Key Points": """ - Collingwood 
                - Toorak 
                - Brunswick
                - Cremorne 
                
                Some of these places are great living areas, and some are great places to eat.
                S2Search is based in Melbourne. Melbourne has beautiful waterways running through it.
                """
            },
        ])
        print("\nSearching the phrase 'River' across all fields")
        pprint.pprint(client.index("cool-index-1").search("River"))
        # then we search specific searchable attributes. We can see how powerful semantic search is
        print("\nThen we search specific 'River over' searchable attributes. We can see how powerful semantic search is")
        pprint.pprint(client.index("cool-index-1").search("River", searchable_attributes=["Key Points"]))

    def test_readme_example(self):

        import marqo

        mq = marqo.Client(url='https://localhost:9200', main_user="admin", main_password="admin")

        mq.index("my-first-index").add_documents([
            {
                "Title": "The Travels of Marco Polo",
                "Description": "A 13th-century travelogue describing Polo's travels"},
            {
                "Title": "Extravehicular Mobility Unit (EMU)",
                "Description": "The EMU is a spacesuit that provides environmental protection, "
                               "mobility, life support, and communications for astronauts",
                "_id": "article_591"
            }
        ])

        results = mq.index("my-first-index").search(
            q="What is the best outfit to wear on the moon?"
        )

        pprint.pprint(results)

        assert results["hits"][0]["_id"] == "article_591"

        r2 = mq.index("my-first-index").get_document(document_id="article_591")
        assert {
                "Title": "Extravehicular Mobility Unit (EMU)",
                "Description": "The EMU is a spacesuit that provides environmental protection, "
                               "mobility, life support, and communications for astronauts",
                "_id": "article_591"
            } == r2

        r3 = mq.index("my-first-index").get_stats()

        assert r3["numberOfDocuments"] == 2

        r4 = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
        assert r4["hits"][0]["Title"] == "The Travels of Marco Polo"

        r5 = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
        assert len(r5["hits"]) == 2

        r6 = mq.index("my-first-index").delete_documents(ids=["article_591"])
        assert r6['deleted'] == 1

        rneg1 = mq.index("my-first-index").delete()
        assert rneg1["acknowledged"] is True

