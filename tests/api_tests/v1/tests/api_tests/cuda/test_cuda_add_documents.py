import uuid
from unittest import mock

import pytest
from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


@pytest.mark.cuda_test
class TestCudaStructuredAddDocuments(MarqoTestCase):
    text_index_name = "add_doc_api_test_structured_index" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "add_doc_api_test_structured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
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
        ]
        )

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_add_documents_with_ids(self):
        d1 = {
            "title": "Cool Document 1",
            "content": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "title": "Just Your Average Doc",
            "content": "this is a solid doc",
            "_id": "123456"
        }

        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], device = "cuda")

        retrieved_d1 = self.client.index(self.text_index_name).get_document(
            document_id="e197e580-0393-4f4e-90e9-8cdf4b17e339")
        assert retrieved_d1 == d1
        retrieved_d2 = self.client.index(self.text_index_name).get_document(document_id="123456")
        assert retrieved_d2 == d2

    def test_add_documents_without_ids(self):
        """indexes the documents and retrieves the documents with the generated IDs"""
        d1 = {
            "title": "Cool Document 1",
            "content": "some extra info"
        }
        d2 = {
            "title": "Just Your Average Doc",
            "content": "this is a solid doc"
        }
        res = self.client.index(self.text_index_name).add_documents([d1, d2], device = "cuda")
        ids = [item["_id"] for item in res["items"]]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        retrieved_d0 = self.client.index(self.text_index_name).get_document(ids[0])
        retrieved_d1 = self.client.index(self.text_index_name).get_document(ids[1])
        del retrieved_d0["_id"]
        del retrieved_d1["_id"]
        assert retrieved_d0 == d1 or retrieved_d0 == d2
        assert retrieved_d1 == d1 or retrieved_d1 == d2

    def test_add_batched_documents(self):
        ix = self.client.index(index_name=self.text_index_name)
        doc_ids = [str(num) for num in range(0, 100)]
        docs = [
            {"title": f"The Title of doc {doc_id}",
             "content": "some text goes here...",
             "_id": doc_id}
            for doc_id in doc_ids]

        ix.add_documents(docs, client_batch_size=10, device = "cuda")
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_with_ids_twice(self):
        d1 = {
            "title": "Just Your Average Doc",
            "content": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.text_index_name).add_documents([d1], device = "cuda")
        assert d1 == self.client.index(self.text_index_name).get_document("56")
        d2 = {
            "_id": "56",
            "title": "different doc.",
            "content": "this is a solid doc"
        }
        self.client.index(self.text_index_name).add_documents([d2], device = "cuda")
        assert d2 == self.client.index(self.text_index_name).get_document("56")

    def test_add_documents_missing_index_fails(self):
        with self.assertRaises(MarqoWebError) as ex:
            self.client.index("a void index").add_documents([{"title": "efg"}], device = "cuda")
        assert "index_not_found" in str(ex.exception.message)

    def test_add_documents_empty(self):
        """
        Test that adding an empty list of documents fails with bad_request
        """
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.text_index_name).add_documents(documents=[], device = "cuda")
        assert "bad_request" in str(e.exception.message)

    def test_add_docs_image_download_headers(self):
        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            image_download_headers = {"Authentication": "my-secret-key"}
            self.client.index(index_name=self.image_index_name).add_documents(
                documents=[{"title": "data"}], image_download_headers=image_download_headers,
                tensor_fields=[], device = "cuda")
            args, kwargs = mock__post.call_args
            assert "imageDownloadHeaders" in kwargs['body']
            assert kwargs['body']['imageDownloadHeaders'] == image_download_headers

            return True

        assert run()




@pytest.mark.cuda_test

class TestCudaUnstructuredAddDocuments(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

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
                "model": "open_clip/ViT-B-32/openai",
                "treatUrlsAndPointersAsImages": True,
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_add_documents_with_ids(self):
        d1 = {
            "doc_title": "Cool Document 1",
            "field_1": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "doc_title": "Just Your Average Doc",
            "field_X": "this is a solid doc",
            "_id": "123456"
        }

        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], tensor_fields=["doc_title"], device="cuda")

        retrieved_d1 = self.client.index(self.text_index_name).get_document(
            document_id="e197e580-0393-4f4e-90e9-8cdf4b17e339")
        assert retrieved_d1 == d1
        retrieved_d2 = self.client.index(self.text_index_name).get_document(document_id="123456")
        assert retrieved_d2 == d2

    def test_add_documents_without_ids(self):
        """indexes the documents and retrieves the documents with the generated IDs"""
        d1 = {
            "doc_title": "Cool Document 1",
            "field_1": "some extra info"
        }
        d2 = {
            "doc_title": "Just Your Average Doc",
            "field_X": "this is a solid doc"
        }
        res = self.client.index(self.text_index_name).add_documents([d1, d2], tensor_fields=["doc_title"]
                                                                    , device="cuda")
        ids = [item["_id"] for item in res["items"]]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        retrieved_d0 = self.client.index(self.text_index_name).get_document(ids[0])
        retrieved_d1 = self.client.index(self.text_index_name).get_document(ids[1])
        del retrieved_d0["_id"]
        del retrieved_d1["_id"]
        assert retrieved_d0 == d1 or retrieved_d0 == d2
        assert retrieved_d1 == d1 or retrieved_d1 == d2

    def test_add_batched_documents(self):
        ix = self.client.index(index_name=self.text_index_name)
        doc_ids = [str(num) for num in range(0, 100)]
        docs = [
            {"Title": f"The Title of doc {doc_id}",
             "Generic_text": "some text goes here...",
             "_id": doc_id}
            for doc_id in doc_ids]

        ix.add_documents(docs, client_batch_size=10, tensor_fields=["Title", "Generic_text"], device="cuda")
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_with_ids_twice(self):
        d1 = {
            "doc_title": "Just Your Average Doc",
            "field_X": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.text_index_name).add_documents([d1], tensor_fields=["doc_title"], device="cuda")
        assert d1 == self.client.index(self.text_index_name).get_document("56")
        d2 = {
            "_id": "56",
            "completely": "different doc.",
            "field_X": "this is a solid doc"
        }
        self.client.index(self.text_index_name).add_documents([d2], tensor_fields=["doc_title"], device="cuda")
        assert d2 == self.client.index(self.text_index_name).get_document("56")

    def test_add_documents_missing_index_fails(self):
        with self.assertRaises(MarqoWebError) as ex:
            self.client.index("a void index").add_documents([{"abd": "efg"}], tensor_fields=[], device="cuda")
        assert "index_not_found" in str(ex.exception.message)

    def test_add_documents_empty(self):
        """
        Test that adding an empty list of documents fails with bad_request
        """
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.text_index_name).add_documents(documents=[], tensor_fields=[], device="cuda")
        assert "bad_request" in str(e.exception.message)

    def test_add_docs_image_download_headers(self):
        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            image_download_headers = {"Authentication": "my-secret-key"}
            self.client.index(index_name=self.image_index_name).add_documents(
                documents=[{"some": "data"}], image_download_headers=image_download_headers,
                tensor_fields=[], device="cuda")
            args, kwargs = mock__post.call_args
            assert "imageDownloadHeaders" in kwargs['body']
            assert kwargs['body']['imageDownloadHeaders'] == image_download_headers

            return True

        assert run()

    def test_add_document_multimodal(self):
        """Test that adding a document with a multimodal field works"""
        image_content = "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg"

        documents = [
            {
                "title": "test-1",
                "image_content": image_content,
                "non_tensor": "test"
            },
            {
                "title": "test-2",
                "image_content": image_content,
                "content": "test"
            },
        ]

        # Mappings, tensor fields, number_of_documents, number_of_vectors, msg
        test_cases = [
            ({"my_multimodal_field": {"type": "multimodal_combination",
                                      "weights": {"title": 0.5, "image_content": 0.8}}},
             ["my_multimodal_field"], 2, 2, "single multimodal field"),

            ({"my_multimodal_field": {"type": "multimodal_combination",
                                      "weights": {"title": 0.5, "image_content": 0.8}}},
             ["my_multimodal_field", "title", "content"], 2, 5, "multimodal field with other tensor fields"),

            ({"my_multimodal_field": {"type": "multimodal_combination",
                                      "weights": {"content": 0.5, "void_content": 0.8}}},
             ["my_multimodal_field", "title"], 2, 3, "multimodal field with other tensor fields"),

            ({"my_multimodal_field": {"type": "multimodal_combination",
                                      "weights": {"voind_content_2": 0.5, "void_content_1": 0.8}}},
             ["my_multimodal_field"], 2, 0, "multimodal field with other tensor fields"),

            ({"my_multimodal_field_1": {"type": "multimodal_combination",
                                        "weights": {"title": 0.5, "image_content": 0.8}},
              "my_multimodal_field_2": {"type": "multimodal_combination",
                                        "weights": {"void": 0.5, "content": 0.8}}
              },
             ["my_multimodal_field_1", "my_multimodal_field_2"], 2, 3, "multiple multimodal fields"),
        ]

        for mappings, tensor_fields, number_of_documents, number_of_vectors, msg in test_cases:
            with self.subTest(msg):
                self.clear_indexes([self.image_index_name])
                self.client.index(self.image_index_name).add_documents(
                    documents=documents,
                    device = "cuda",
                    mappings=mappings,
                    tensor_fields=tensor_fields
                )

                res = self.client.index(self.image_index_name).get_stats()
                self.assertEqual(number_of_documents, res["numberOfDocuments"])
                self.assertEqual(number_of_vectors, res["numberOfVectors"])

    def test_add_documents_call_tensor_fields(self):
        """Test that calling add_documents without tensor_fields fails"""
        test_cases = [
            ({"tensor_fields": None}, "None as tensor fields"),
            ({}, "No tensor fields"),
        ]
        for tensor_fields, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(self.text_index_name).add_documents(documents=[{"some": "data"}], device="cuda",
                                                                          **tensor_fields)
                assert "bad_request" in str(e.exception.message)