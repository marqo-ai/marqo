import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestUnstructuredDeleteDocuments(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.delete_indexes(["api_test_unstructured_index", "api_test_unstructured_image_index"])
        except Exception:
            pass

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')
        cls.image_index_name = "api_test_unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)
            
    def test_delete_docs(self):
        self.client.index(self.text_index_name).add_documents([
            {"abc": "wow camel", "_id": "123"},
            {"abc": "camels are cool", "_id": "foo"}
        ], tensor_fields=["abc"])

        res0 = self.client.index(self.text_index_name).search("wow camel")

        assert res0['hits'][0]["_id"] == "123"
        assert len(res0['hits']) == 2
        res2 = self.client.index(self.text_index_name).delete_documents(["123"])
        print(res2)
        res1 = self.client.index(self.text_index_name).search("wow camel")
        print(res1)
        assert res1['hits'][0]["_id"] == "foo"
        assert len(res1['hits']) == 1

    def test_delete_docs_empty_ids(self):
        self.client.index(self.text_index_name).add_documents([{"abc": "efg", "_id": "123"}], tensor_fields=[])
        try:
            self.client.index(self.text_index_name).delete_documents([])
            raise AssertionError
        except MarqoWebError as e:
            assert "can't be empty" in str(e) or "value_error.missing" in str (e)
        res = self.client.index(self.text_index_name).get_document("123")
        assert "abc" in res

    def test_delete_docs_response(self):
        """
        Ensure that delete docs response has the correct format
        items list, index_name, status, type, details, duration, startedAt, finishedAt
        """
        self.client.index(self.text_index_name).add_documents([
            {"_id": "doc1", "abc": "wow camel"},
            {"_id": "doc2", "abc": "camels are cool"},
            {"_id": "doc3", "abc": "wow camels again"}
        ], tensor_fields=[])

        res = self.client.index(self.text_index_name).delete_documents(["doc1", "doc2", "doc3"])

        assert "duration" in res
        assert "startedAt" in res
        assert "finishedAt" in res

        assert res["index_name"] == self.text_index_name
        assert res["type"] == "documentDeletion"
        assert res["status"] == "succeeded"
        assert res["details"] == {
            "receivedDocumentIds": 3,
            "deletedDocuments": 3
        }
        assert len(res["items"]) == 3

        for item in res["items"]:
            if item["_id"] in {"doc1", "doc2", "doc3"}:
                assert item["status"] == 200
                assert item["result"] == "deleted"

