import copy
import uuid
from unittest import mock

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestUnstructuredAddDocuments(MarqoTestCase):
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')
        cls.image_index_name = "api_test_unstructured_image_index" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_languagebind_index_name = "api_test_unstructured_languagebind_index" + str(uuid.uuid4()).replace('-', '')
        cls.text_index_with_normalize_embeddings_true = "api_test_unstructured_index_with_normalize_embeddings_true" + str(
            uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "normalizeEmbeddings": False,
            },
            {
                "indexName": cls.image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai",
                "treatUrlsAndPointersAsImages": True,
            },
            {
                "indexName": cls.unstructured_languagebind_index_name,
                "type": "unstructured",
                "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                "treatUrlsAndPointersAsMedia": True,
                "treatUrlsAndPointersAsImages": True
            },
            {
                "indexName": cls.text_index_with_normalize_embeddings_true,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
                "normalizeEmbeddings": True,
            }
            ])
        
        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name, cls.unstructured_languagebind_index_name, cls.text_index_with_normalize_embeddings_true]
        
        
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
        ], tensor_fields=["doc_title"])

        retrieved_d1 = self.client.index(self.text_index_name).get_document(document_id="e197e580-0393-4f4e-90e9-8cdf4b17e339")
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
        res = self.client.index(self.text_index_name).add_documents([d1, d2], tensor_fields=["doc_title"])
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

        ix.add_documents(docs, client_batch_size=10, tensor_fields=["Title", "Generic_text"])
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_with_ids_twice(self):
        d1 = {
            "doc_title": "Just Your Average Doc",
            "field_X": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.text_index_name).add_documents([d1], tensor_fields=["doc_title"])
        assert d1 == self.client.index(self.text_index_name).get_document("56")
        d2 = {
            "_id": "56",
            "completely": "different doc.",
            "field_X": "this is a solid doc"
        }
        self.client.index(self.text_index_name).add_documents([d2], tensor_fields=["doc_title"])
        assert d2 == self.client.index(self.text_index_name).get_document("56")

    def test_add_documents_missing_index_fails(self):
        with self.assertRaises(MarqoWebError) as ex:
            self.client.index("a void index").add_documents([{"abd": "efg"}], tensor_fields=[])
        assert "index_not_found" in str(ex.exception.message)

    def test_add_documents_with_device(self):
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.image_index_name).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], device="cuda:45", tensor_fields=[])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "device=cuda45" in kwargs["path"]

    def test_add_documents_no_device(self):
        """No device should be in path if no device is set
        """
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.image_index_name).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], tensor_fields=[])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "device" not in kwargs["path"]

    def test_add_documents_empty(self):
        """
        Test that adding an empty list of documents fails with bad_request
        """
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.text_index_name).add_documents(documents=[], tensor_fields=[])
        assert "bad_request" in str(e.exception.message)

    def test_add_docs_image_download_headers(self):
        mock__post = mock.MagicMock()
        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            image_download_headers = {"Authentication": "my-secret-key"}
            self.client.index(index_name=self.image_index_name).add_documents(
                documents=[{"some": "data"}], image_download_headers=image_download_headers,
                tensor_fields=[])
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
            ({"my_multimodal_field": {"type": "multimodal_combination", "weights": {"title": 0.5, "image_content": 0.8}}},
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
                    device="cpu",
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
                    self.client.index(self.text_index_name).add_documents(documents=[{"some": "data"}], **tensor_fields)
                assert "bad_request" in str(e.exception.message)

    def test_add_docs_with_large_integers_and_floats(self):
        test_documents = [
            ({"long_field_1": 1}, False),  # small positive integer
            ({"long_field_1": -1}, False),  # small negative integer
            ({"long_field_1": 1002321423542}, False),  # large positive integer that can't be handled by int
            ({"long_field_1": -9232172132345}, False),  # large positive integer that can't be handled by int
            ({'long_field_1': int("1" * 50)}, True),  # overlarge positive integer, should raise error in long field
            # overlarge negative integer, should raise error in long field
            ({'long_field_1': -1 * int("1" * 50)}, True),
            ({"double_field_1": 1e10}, False),  # large positive integer mathematical expression
            ({"double_field_1": -1e12}, False),  # large negative integer mathematical expression
            ({"double_field_1": 1e10 + 0.123249357987123}, False),  # large positive float
            ({"double_field_1": - 1e10 + 0.123249357987123}, False),  # large negative float
        ]
        for test_document, error in test_documents:
            with self.subTest(f"doc = {test_document}"):
                res = self.client.index(self.text_index_name).add_documents(
                        [test_document], tensor_fields=[]
                    )
                self.assertEqual(res['errors'], error)

    def test_custom_vector_doc_in_normalized_embedding_true(self):

        DEFAULT_DIMENSIONS = 384
        custom_vector = [1.0 for _ in range(DEFAULT_DIMENSIONS)]
        expected_custom_vector_after_normalization = [0.05103103816509247 for _ in range(DEFAULT_DIMENSIONS)]

        add_docs_res_normalized = self.client.index(index_name=self.text_index_with_normalize_embeddings_true).add_documents(
            documents=[
                {
                    "custom_vector_field_1": {
                        "content": "custom vector text",
                        "vector": custom_vector,
                    },
                    "content": "normal text",
                    "_id": "doc1",
                },
                {
                    "content": "second doc",
                    "_id": "doc2"
                }
            ],
            tensor_fields=["custom_vector_field_1"],
            mappings = {
                "custom_vector_field_1": {
                    "type": "custom_vector"
                }
            }
        )
        doc_res_normalized = self.client.index(self.text_index_with_normalize_embeddings_true).get_document(
            document_id="doc1",
            expose_facets=True
        )

        assert doc_res_normalized["custom_vector_field_1"] == "custom vector text"
        assert doc_res_normalized['_tensor_facets'][0]["custom_vector_field_1"] == "custom vector text"
        assert doc_res_normalized['_tensor_facets'][0]['_embedding'] == expected_custom_vector_after_normalization


    def test_custom_zero_vector_doc_in_normalized_embedding_true(self):

        DEFAULT_DIMENSIONS = 384
        custom_vector = [0 for _ in range(DEFAULT_DIMENSIONS)]

        add_docs_res_normalized = self.client.index(index_name=self.text_index_with_normalize_embeddings_true).add_documents(
            documents=[
                {
                    "custom_vector_field_1": {
                        "content": "custom vector text",
                        "vector": custom_vector,
                    },
                    "content": "normal text",
                    "_id": "doc1",
                },
                {
                    "content": "second doc",
                    "_id": "doc2"
                }
            ],
            tensor_fields=["custom_vector_field_1"],
            mappings = {
                "custom_vector_field_1": {
                    "type": "custom_vector"
                }
            }
        )
        self.assertEqual(add_docs_res_normalized["errors"], True)
        self.assertEqual(add_docs_res_normalized["items"][0]["status"], 400)
        self.assertIn("Field custom_vector_field_1 has zero magnitude vector, cannot normalize.", add_docs_res_normalized["items"][0]["message"])
        self.assertEqual(add_docs_res_normalized["items"][0]["code"], "invalid_argument")
        self.assertEqual(add_docs_res_normalized["items"][0]["_id"], "doc1")

        self.assertEqual(add_docs_res_normalized["items"][1]["status"], 200)
        self.assertEqual(add_docs_res_normalized["items"][1]["_id"], "doc2")