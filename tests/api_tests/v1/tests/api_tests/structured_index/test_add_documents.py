import copy
import unittest
import uuid
from unittest import mock

import pytest
from marqo.client import Client
from marqo.errors import MarqoWebError
from tests.marqo_test import MarqoTestCase


class TestStructuredAddDocuments(MarqoTestCase):
    text_index_name = "add_doc_api_test_structured_index" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "add_doc_api_test_structured_image_index" + str(uuid.uuid4()).replace('-', '')
    structured_languagebind_index_name = "add_doc_api_test_structured_languagebind_index" + str(uuid.uuid4()).replace('-', '')
    text_index_with_normalize_embeddings_true = "add_doc_api_test_structured_index_with_normalize_embeddings_true" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "normalizeEmbeddings": False,
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "int_field_1", "type": "int"},
                    {"name": "float_field_1", "type": "float"},
                    {"name": "long_field_1", "type": "long"},
                    {"name": "double_field_1", "type": "double"},
                    {"name": "array_int_field_1", "type": "array<int>"},
                    {"name": "array_float_field_1", "type": "array<float>"},
                    {"name": "array_long_field_1", "type": "array<long>"},
                    {"name": "array_double_field_1", "type": "array<double>"},
                    {"name": "custom_vector_field_1", "type": "custom_vector", "features": ["lexical_search", "filter"]},
                ],
                "tensorFields": ["title", "content", "custom_vector_field_1"],
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
            },
            {
                "indexName": cls.structured_languagebind_index_name,
                "type": "structured",
                "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                "allFields": [
                    {"name": "text_field_1", "type": "text"},
                    {"name": "text_field_2", "type": "text"},
                    {"name": "text_field_3", "type": "text"},
                    {"name": "video_field_1", "type": "video_pointer"},
                    {"name": "video_field_2", "type": "video_pointer"},
                    {"name": "video_field_3", "type": "video_pointer"},
                    {"name": "audio_field_1", "type": "audio_pointer"},
                    {"name": "audio_field_2", "type": "audio_pointer"},
                    {"name": "image_field_1", "type": "image_pointer"},
                    {"name": "image_field_2", "type": "image_pointer"},
                    {
                        "name": "multimodal_field", 
                        "type": "multimodal_combination",
                        "dependentFields": {
                            "text_field_1": 0.1,
                            "text_field_2": 0.1,
                            "image_field_1": 0.5,
                            "video_field_1": 0.1,
                            "video_field_2": 0.1,
                            "audio_field_1": 0.1
                        }
                    },
                ],
                "tensorFields": ["multimodal_field", "text_field_3", "video_field_3", "audio_field_2", "image_field_2"]
            },
            {
                "indexName": cls.text_index_with_normalize_embeddings_true,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "normalizeEmbeddings": True,
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "int_field_1", "type": "int"},
                    {"name": "float_field_1", "type": "float"},
                    {"name": "long_field_1", "type": "long"},
                    {"name": "double_field_1", "type": "double"},
                    {"name": "array_int_field_1", "type": "array<int>"},
                    {"name": "array_float_field_1", "type": "array<float>"},
                    {"name": "array_long_field_1", "type": "array<long>"},
                    {"name": "array_double_field_1", "type": "array<double>"},
                    {"name": "custom_vector_field_1", "type": "custom_vector",
                     "features": ["lexical_search", "filter"]},
                ],
                "tensorFields": ["title", "content", "custom_vector_field_1"],
            },
        ]
        )

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name, cls.structured_languagebind_index_name, cls.text_index_with_normalize_embeddings_true]

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
        ])

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
        res = self.client.index(self.text_index_name).add_documents([d1, d2])
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

        ix.add_documents(docs, client_batch_size=10)
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_with_ids_twice(self):
        d1 = {
            "title": "Just Your Average Doc",
            "content": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.text_index_name).add_documents([d1])
        assert d1 == self.client.index(self.text_index_name).get_document("56")
        d2 = {
            "_id": "56",
            "title": "different doc.",
            "content": "this is a solid doc"
        }
        self.client.index(self.text_index_name).add_documents([d2])
        assert d2 == self.client.index(self.text_index_name).get_document("56")

    def test_add_documents_missing_index_fails(self):
        with self.assertRaises(MarqoWebError) as ex:
            self.client.index("a void index").add_documents([{"title": "efg"}])
        assert "index_not_found" in str(ex.exception.message)

    def test_add_documents_with_device(self):
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.image_index_name).add_documents(documents=[
                {"title": "blah"}, {"title", "some data"}
            ], device="cuda:45")
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
                {"title": "blah"}, {"title", "some data"}
            ])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "device" not in kwargs["path"]

    def test_add_documents_empty(self):
        """
        Test that adding an empty list of documents fails with bad_request
        """
        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.text_index_name).add_documents(documents=[])
        assert "bad_request" in str(e.exception.message)

    def test_add_docs_image_download_headers(self):
        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            image_download_headers = {"Authentication": "my-secret-key"}
            self.client.index(index_name=self.image_index_name).add_documents(
                documents=[{"title": "data"}], image_download_headers=image_download_headers,
                tensor_fields=[])
            args, kwargs = mock__post.call_args
            assert "imageDownloadHeaders" in kwargs['body']
            assert kwargs['body']['imageDownloadHeaders'] == image_download_headers

            return True

        assert run()

    def test_add_long_double_numeric_values(self):
        """Test to ensure large integer and float numbers are handled correctly for long and double fields"""
        test_case = [
            ({"_id": "1", "int_field_1": 2147483647}, False, "maximum positive integer that can be handled by int"),
            ({"_id": "2", "int_field_1": -2147483647}, False, "maximum negative integer that can be handled by int"),
            ({"_id": "3", "int_field_1": 2147483648}, True,
             "integer slightly larger than boundary so can't be handled by int"),
            ({"_id": "4", "long_field_1": 2147483648}, False,
             "integer slightly larger than boundary can be handled by long"),
            ({"_id": "5", "int_field_1": -2147483648}, True,
             "integer slightly smaller than boundary so can't be handled by int"),
            ({"_id": "6", "long_field_1": -2147483648}, False,
             "integer slightly larger than boundary can be handled by long"),
            ({"_id": "7", "float_field_1": 3.4028235e38}, False, "maximum positive float that can be handled by float"),
            (
                {"_id": "8", "float_field_1": -3.4028235e38}, False,
                "maximum negative float that can be handled by float"),
            ({"_id": "9", "float_field_1": 3.4028235e40}, True,
             "float slightly larger than boundary can't be handled by float"),
            ({"_id": "10", "double_field_1": 3.4028235e40}, False,
             "float slightly larger than boundary can be handled by double"),
            ({"_id": "13", "long_field_1": 1}, False, "small positive integer"),
            ({"_id": "14", "long_field_1": -1}, False, "small negative integer"),
            ({"_id": "15", "long_field_1": 100232142864}, False, "large positive integer that can't be handled by int"),
            ({"_id": "16", "long_field_1": -923217213}, False, "large negative integer that can't be handled by int"),
            ({"_id": "17", 'long_field_1': int("1" * 50)}, True,
             "overlarge positive integer, should raise error in long field"),
            ({"_id": "18", 'long_field_1': -1 * int("1" * 50)}, True,
             "overlarge negative integer, should raise error in long field"),
            ({"_id": "19", "double_field_1": 1e10}, False, "large positive integer mathematical expression"),
            ({"_id": "20", "double_field_1": -1e12}, False, "large negative integer mathematical expression"),
            ({"_id": "21", "double_field_1": 1e10 + 0.123249357987123}, False, "large positive float"),
            ({"_id": "22", "double_field_1": -1e10 + 0.123249357987123}, False, "large negative float"),
            ({"_id": "23", "array_double_field_1": [1e10, 1e10 + 0.123249357987123]}, False, "large float array"),
        ]

        for doc, error, msg in test_case:
            with self.subTest(msg):
                res = self.client.index(self.text_index_name).add_documents(documents=[doc])
                self.assertEqual(res['errors'], error)
                if error:
                    self.assertIn("Invalid value", res['items'][0]['error'])
                else:
                    document_id = doc["_id"]
                    returned_doc = self.client.index(self.text_index_name).get_document(document_id=document_id,
                                                                                        expose_facets=False)
                    # Ensure we get the same document back for those that are valid
                    self.assertEqual(doc, returned_doc)

    def test_long_double_numeric_values_edge_case(self):
        """We test some edge cases here for clarity"""
        test_case = [
            ({"_id": "1", "float_field_1": 1e-50},
             {"_id": "1", "float_field_1": 0},
             "small positive float will be rounded to 0"),
            ({"_id": "2", "float_field_1": -1e-50},
             {"_id": "2", "float_field_1": 0},
             "small negative float will be rounded to 0"),
        ]

        for doc, expected_doc, msg in test_case:
            with self.subTest(msg):
                res = self.client.index(self.text_index_name).add_documents(documents=[doc])
                self.assertFalse(res['errors'])
                document_id = doc["_id"]
                returned_doc = self.client.index(self.text_index_name).get_document(
                    document_id, False
                )
                self.assertEqual(expected_doc, returned_doc)

    def test_custom_vector_doc(self):
        """
        Tests the custom_vector field type.
        Ensures the following features work on this field:
        1. lexical search
        2. filter string search
        3. tensor search
        4. get document
        """

        DEFAULT_DIMENSIONS = 384

        add_docs_res = self.client.index(index_name=self.text_index_name).add_documents(
            documents=[
                {
                    "custom_vector_field_1": {
                        "content": "custom vector text",
                        "vector": [1.0 for _ in range(DEFAULT_DIMENSIONS)],
                    },
                    "content": "normal text",
                    "_id": "doc1",
                },
                {
                    "content": "second doc",
                    "_id": "doc2"
                }
            ])

        # lexical search test
        lexical_res = self.client.index(self.text_index_name).search(
            "custom vector text", search_method="lexical")
        assert lexical_res["hits"][0]["_id"] == "doc1"

        # filter string test
        filtering_res = self.client.index(self.text_index_name).search(
            "", filter_string="custom_vector_field_1:(custom vector text)")
        assert filtering_res["hits"][0]["_id"] == "doc1"

        # tensor search test
        tensor_res = self.client.index(self.text_index_name).search(q={"dummy text": 0}, context={
            "tensor": [{"vector": [1.0 for _ in range(DEFAULT_DIMENSIONS)], "weight": 1}]})
        assert tensor_res["hits"][0]["_id"] == "doc1"

        # get document test
        doc_res = self.client.index(self.text_index_name).get_document(
            document_id="doc1",
            expose_facets=True
        )
        assert doc_res["custom_vector_field_1"] == "custom vector text"
        assert doc_res['_tensor_facets'][0]["custom_vector_field_1"] == "custom vector text"
        assert doc_res['_tensor_facets'][0]['_embedding'] == [1.0 for _ in range(DEFAULT_DIMENSIONS)]

    @unittest.skip(reason="Temporarily skipped as languagebind model is not available")
    @pytest.mark.cuda_test
    def test_add_multimodal_single_documents(self):
        documents = [
            {
                "video_field_3": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
                "_id": "1"
            },
            {
                "audio_field_2": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3",
                "_id": "2"
            },
            {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "3"
            },
            {
                "text_field_3": "hello there padawan. Today you will begin your training to be a Jedi",
                "_id": "4"
            },
        ]
        
        res = self.client.index(self.structured_languagebind_index_name).add_documents(documents)
        for item in res['items']:
            self.assertEqual(200, item['status'])

        get_res = self.client.index(self.structured_languagebind_index_name).get_documents(
            document_ids=["1", "2", "3", "4"],
            expose_facets=True
        )
        for doc in get_res['results']:
            tensor_facets = doc['_tensor_facets']
            self.assertIn('_embedding', tensor_facets[0])
            self.assertEqual(len(tensor_facets[0]['_embedding']), 768)

    @unittest.skip(reason="Temporarily skipped as languagebind model is not available")
    @pytest.mark.cuda_test
    def test_add_documents_with_invalid_media_fields(self):
        documents = [
            {
                "video_field_3": "This is text, not a video URL",
                "_id": "1"
            },
            {
                "audio_field_2": "This is text, not an audio URL",
                "_id": "2"
            },
            {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "3"
            },
            {
                "text_field_3": "This is a valid text field",
                "_id": "4"
            },
        ]
        
        res = self.client.index(self.structured_languagebind_index_name).add_documents(documents)

        # Check that the batch operation didn't fail
        self.assertTrue(res['errors'])
        
        # Check individual document statuses
        self.assertEqual(400, res['items'][0]['status'])  # Invalid video field
        self.assertEqual(400, res['items'][1]['status'])  # Invalid audio field
        self.assertEqual(200, res['items'][2]['status'])  # Valid image field
        self.assertEqual(200, res['items'][3]['status'])  # Valid text field
        
        # Verify that valid documents were added
        get_res = self.client.index(self.structured_languagebind_index_name).get_documents(
            document_ids=["3", "4"],
            expose_facets=True
        )
        
        self.assertEqual(2, len(get_res['results']))
        
        for doc in get_res['results']:
            tensor_facets = doc['_tensor_facets']
            self.assertIn('_embedding', tensor_facets[0])
            self.assertEqual(len(tensor_facets[0]['_embedding']), 768)
        
        # Verify that invalid documents were not added
        with self.assertRaises(MarqoWebError):
            self.client.index(self.structured_languagebind_index_name).get_document(document_id="1")
        
        with self.assertRaises(MarqoWebError):
            self.client.index(self.structured_languagebind_index_name).get_document(document_id="2")

    @unittest.skip(reason="Temporarily skipped as languagebind model is not available")
    @pytest.mark.cuda_test
    def test_add_documents_with_mismatched_media_fields(self):
        documents = [
            {
                "video_field_3": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3",
                "_id": "1"
            },
            {
                "audio_field_2": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
                "_id": "2"
            },
            {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "3"
            },
            {
                "text_field_3": "This is a valid text field",
                "_id": "4"
            },
        ]
        
        res = self.client.index(self.structured_languagebind_index_name).add_documents(documents)

        # Check that the batch operation had errors
        self.assertTrue(res['errors'])
        
        # Check individual document statuses and error messages
        self.assertEqual(400, res['items'][0]['status'])  # Audio in video field
        self.assertIn("Invalid video file", res['items'][0]['error'])
        
        self.assertEqual(400, res['items'][1]['status'])  # Video in audio field
        self.assertIn("Invalid audio file", res['items'][1]['error'])
        
        self.assertEqual(200, res['items'][2]['status'])  # Valid image field
        self.assertEqual(200, res['items'][3]['status'])  # Valid text field
        
        # Verify that valid documents were added
        get_res = self.client.index(self.structured_languagebind_index_name).get_documents(
            document_ids=["3", "4"],
            expose_facets=True
        )
        
        self.assertEqual(2, len(get_res['results']))
        
        for doc in get_res['results']:
            tensor_facets = doc['_tensor_facets']
            self.assertIn('_embedding', tensor_facets[0])
            self.assertEqual(len(tensor_facets[0]['_embedding']), 768)
        
        # Verify that invalid documents were not added
        with self.assertRaises(MarqoWebError):
            self.client.index(self.structured_languagebind_index_name).get_document(document_id="1")
        
        with self.assertRaises(MarqoWebError):
            self.client.index(self.structured_languagebind_index_name).get_document(document_id="2")

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
            ])

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
            ])

        self.assertEqual(add_docs_res_normalized["errors"], True)
        self.assertEqual(add_docs_res_normalized["items"][0]["status"], 400)
        self.assertIn("Field custom_vector_field_1 has zero magnitude vector, cannot normalize.", add_docs_res_normalized["items"][0]["message"])
        self.assertEqual(add_docs_res_normalized["items"][0]["code"], "invalid_argument")
        self.assertEqual(add_docs_res_normalized["items"][0]["_id"], "doc1")

        self.assertEqual(add_docs_res_normalized["items"][1]["status"], 200)
        self.assertEqual(add_docs_res_normalized["items"][1]["_id"], "doc2")