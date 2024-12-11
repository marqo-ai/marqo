import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestStructuredDeleteDocuments(MarqoTestCase):
    text_index_name = "api_test_structured_index" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "api_test_structured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "image_content", "type": "image_pointer"},
                ],
                "tensorFields": ["title", "image_content"],
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_delete_docs(self):
        self.client.index(self.text_index_name).add_documents([
            {"title": "wow camel", "_id": "123"},
            {"title": "camels are cool", "_id": "foo"}
        ])

        res0 = self.client.index(self.text_index_name).search("wow camel")

        assert res0['hits'][0]["_id"] == "123"
        assert len(res0['hits']) == 2
        self.client.index(self.text_index_name).delete_documents(["123"])
        res1 = self.client.index(self.text_index_name).search("wow camel")
        assert res1['hits'][0]["_id"] == "foo"
        assert len(res1['hits']) == 1

    def test_delete_docs_empty_ids(self):
        self.client.index(self.text_index_name).add_documents([{"title": "efg", "_id": "123"}])
        try:
            self.client.index(self.text_index_name).delete_documents([])
            raise AssertionError
        except MarqoWebError as e:
            assert "can't be empty" in str(e) or "value_error.missing" in str(e)
        res = self.client.index(self.text_index_name).get_document("123")
        assert "title" in res

    def test_delete_docs_response(self):
        """
        Ensure that delete docs response has the correct format
        items list, index_name, status, type, details, duration, startedAt, finishedAt
        """
        self.client.index(self.text_index_name).add_documents([
            {"_id": "doc1", "title": "wow camel"},
            {"_id": "doc2", "title": "camels are cool"},
            {"_id": "doc3", "title": "wow camels again"}
        ])

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
            assert "_id" in item
            if item["_id"] in {"doc1", "doc2", "doc3"}:
                assert item["status"] == 200
                assert item["result"] == "deleted"
