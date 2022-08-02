"""you need a local OpenSearch client"""
import copy
import pprint

from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError
import unittest


class TestAddDocuments(unittest.TestCase):

    def setUp(self) -> None:
        self.client = Client(url='https://localhost:9200', main_user="admin", main_password="admin")
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    # Create index tests

    def test_create_index(self):
        self.client.create_index(index_name=self.index_name_1)

    def test_create_index_double(self):
        self.client.create_index(index_name=self.index_name_1)
        try:
            self.client.create_index(index_name=self.index_name_1)
        except MarqoApiError as e:
            assert "already exists" in str(e)

    # Delete index tests:

    def test_delete_index(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.delete_index(self.index_name_1)
        self.client.create_index(index_name=self.index_name_1)

    # Get index tests:

    def test_get_index(self):
        self.client.create_index(index_name=self.index_name_1)
        index = self.client.get_index(self.index_name_1)
        assert index.index_name == self.index_name_1

    def test_get_index_non_existent(self):
        try:
            index = self.client.get_index("some-non-existent-index")
        except MarqoApiError as e:
            assert "no such index" in str(e)

    # Add documents tests:

    def test_add_documents_with_ids(self):
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
                "doc title": "Cool Document 1",
                "field 1": "some extra info",
                "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
            }
        d2 = {
                "doc title": "Just Your Average Doc",
                "field X": "this is a solid doc",
                "_id": "123456"
        }
        res = self.client.index(self.index_name_1).add_documents([
            d1, d2
        ])
        retrieved_d1 = self.client.index(self.index_name_1).get_document(document_id="e197e580-0393-4f4e-90e9-8cdf4b17e339")
        assert retrieved_d1 == d1
        retrieved_d2 = self.client.index(self.index_name_1).get_document(document_id="123456")
        assert retrieved_d2 == d2

    def test_add_documents(self):
        """indexes the documents and retrieves the documents with the generated IDs"""
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
                "doc title": "Cool Document 1",
                "field 1": "some extra info"
            }
        d2 = {
                "doc title": "Just Your Average Doc",
                "field X": "this is a solid doc"
            }
        res = self.client.index(self.index_name_1).add_documents([d1, d2])
        ids = [item["_id"] for item in res["items"]]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        retrieved_d0 = self.client.index(self.index_name_1).get_document(ids[0])
        retrieved_d1 = self.client.index(self.index_name_1).get_document(ids[1])
        del retrieved_d0["_id"]
        del retrieved_d1["_id"]
        assert retrieved_d0 == d1 or retrieved_d0 == d2
        assert retrieved_d1 == d1 or retrieved_d1 == d2

    def test_add_documents_with_ids_twice(self):
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
            "doc title": "Just Your Average Doc",
            "field X": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.index_name_1).add_documents([d1])
        assert d1 == self.client.index(self.index_name_1).get_document("56")
        d2 = {
            "_id": "56",
            "completely": "different doc.",
            "field X": "this is a solid doc"
        }
        self.client.index(self.index_name_1).add_documents([d2])
        assert d2 == self.client.index(self.index_name_1).get_document("56")

    def test_add_batched_documents(self):
        ix = self.client.index(index_name=self.index_name_1)
        doc_ids = [str(num) for num in range(0, 100)]
        docs = [
            {"Title": f"The Title of doc {doc_id}",
             "Generic text": "some text goes here...",
             "_id": doc_id}
            for doc_id in doc_ids]

        ix.add_documents(docs, batch_size=20)
        ix.refresh()
        # TODO we should do a count in here...
        # takes too long to search for all
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_long_fields(self):
        """TODO
        """
    def test_update_docs_updates_chunks(self):
        """TODO"""

    # delete documents tests:

    def test_delete_docs(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.index(self.index_name_1).add_documents([
            {"abc": "wow camel", "_id": "123"},
            {"abc": "camels are cool", "_id": "foo"}
        ])
        res0 = self.client.index(self.index_name_1).search("wow camel")
        print("res0res0")
        pprint.pprint(res0)
        assert res0['hits'][0]["_id"] == "123"
        assert len(res0['hits']) == 2
        self.client.index(self.index_name_1).delete_documents(["123"])
        res1 = self.client.index(self.index_name_1).search("wow camel")
        assert res1['hits'][0]["_id"] == "foo"
        assert len(res1['hits']) == 1

    def test_delete_docs_empty_ids(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.index(self.index_name_1).add_documents([{"abc": "efg", "_id": "123"}])
        try:
            self.client.index(self.index_name_1).delete_documents([])
            raise AssertionError
        except MarqoError as e:
            assert "can't be empty" in str(e)
        res = self.client.index(self.index_name_1).get_document("123")
        print(res)
        assert "abc" in res

    # get documents tests :

    def test_get_document(self):
        """FIXME (do edge cases)"""

    # user experience tests:

    def test_add_documents_implicitly_create_index(self):
        try:
            self.client.index(self.index_name_1).search("some str")
            raise AssertionError
        except MarqoApiError as s:
            assert "no such index" in str(s)
        self.client.index(self.index_name_1).add_documents([{"abd": "efg"}])
        # it works:
        self.client.index(self.index_name_1).search("some str")


def strip_marqo_fields(doc, strip_id=False):
    """Strips Marqo fields from a returned doc to get the original doc"""
    copied = copy.deepcopy(doc)

    strip_fields = ["_highlights", "_score"]
    if strip_id:
        strip_fields += ["_id"]

    for to_strip in strip_fields:
        try:
            del copied[to_strip]
        except KeyError:
            pass
    return copied