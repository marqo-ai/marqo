import uuid
import threading
import time

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestCreateIndex(MarqoTestCase):

    def setUp(self) -> None:
        """As this test class is testing index creation,
        we need to create/delete index before/after each test"""
        super().setUp()
        self.index_name = "test_index"

    def tearDown(self):
        super().tearDown()
        try:
            self.client.delete_index(index_name=self.index_name)
        except MarqoWebError:
            pass

    def test_simple_index_creation(self):
        self.client.create_index(index_name=self.index_name)
        self.client.index(self.index_name).add_documents([{"test": "test"}], tensor_fields=["test"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        index_settings = self.client.index(self.index_name).get_settings()

        expected_settings = {
            'type': 'unstructured',
            'treatUrlsAndPointersAsImages': False,
            'treatUrlsAndPointersAsMedia': False,
            'filterStringMaxLength': 50,
            'model': 'hf/e5-base-v2',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
            'imagePreprocessing': {},
            'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
            'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
            'vectorNumericType': 'float',
            'annParameters': {
                'spaceType': 'prenormalized-angular', 'parameters': {
                    'efConstruction': 512, 'm': 16}
            }
        }
        self.assertEqual(expected_settings, index_settings)

    def test_create_unstructured_image_index(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=True, model="open_clip/ViT-B-32/laion400m_e32")
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test", "image"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(True, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("open_clip/ViT-B-32/laion400m_e32", index_settings['model'])

    def test_create_unstructured_text_index_custom_model(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=False,
                                 model="test-model",
                                 model_properties={"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                                   "dimensions": 384,
                                                   "tokens": 128,
                                                   "type": "hf"}
                                 )
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(False, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("test-model", index_settings['model'])
        self.assertEqual({"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                          "dimensions": 384,
                          "tokens": 128,
                          "type": "hf"}, index_settings['modelProperties'])

    def test_created_unstructured_image_index_with_preprocessing(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=True,
                                 model="open_clip/ViT-B-16/laion400m_e31",
                                 image_preprocessing={"patchMethod": "simple"})
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test", "image"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(True, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("open_clip/ViT-B-16/laion400m_e31", index_settings['model'])
        self.assertEqual("simple", index_settings['imagePreprocessing']['patchMethod'])

    def test_create_invalid_unstructured_languagebind_index(self):
        with self.assertRaises(MarqoWebError) as e:
            res = self.client.create_index(
                index_name=self.index_name,
                type="unstructured",
                model="LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                video_preprocessing={
                    "splitLength": 10,
                    "splitOverlap": 3
                },
                treat_urls_and_pointers_as_media=True,
                treat_urls_and_pointers_as_images=False
            )

    def test_create_unstructured_index_with_languagebind(self):
        self.client.create_index(
            index_name=self.index_name,
            type="unstructured",
            model="LanguageBind/Video_V1.5_FT_Audio_FT_Image",
            video_preprocessing={
                "splitLength": 10,
                "splitOverlap": 3
            },
            audio_preprocessing={
                "splitLength": 10,
                "splitOverlap": 3
            },
            treat_urls_and_pointers_as_media=True,
            treat_urls_and_pointers_as_images=True
        )

        index_settings = self.client.index(self.index_name).get_settings()

        expected_settings = {
            "type": "unstructured",
            "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 2,
                "splitMethod": "sentence",
                "splitOverlap": 0
            },
            "imagePreprocessing": {},
            "videoPreprocessing": {
                "splitLength": 10,
                "splitOverlap": 3
            },
            'filterStringMaxLength': 50,
            "audioPreprocessing": {
                "splitLength": 10,
                "splitOverlap": 3
            },
            "treatUrlsAndPointersAsMedia": True,
            "treatUrlsAndPointersAsImages": True,
            "vectorNumericType": "float",
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {
                    "efConstruction": 512,
                    "m": 16
                }
            }
        }

        self.assertEqual(expected_settings, index_settings)

    def test_create_structured_index_with_languagebind(self):
        self.client.create_index(
            index_name=self.index_name,
            type="structured",
            model="LanguageBind/Video_V1.5_FT_Audio_FT_Image",
            all_fields=[
                {"name": "text_field_1", "type": "text"},
                {"name": "text_field_2", "type": "text"},
                {"name": "video_field_1", "type": "video_pointer"},
                {"name": "video_field_2", "type": "video_pointer"},
                {"name": "audio_field", "type": "audio_pointer"},
                {"name": "image_field", "type": "image_pointer"}
            ],
            tensor_fields=["text_field_1", "text_field_2",
                        "video_field_1", "video_field_2", "audio_field", "image_field"],
            audio_preprocessing={
                "splitLength": 10,
                "splitOverlap": 3
            },
            video_preprocessing={
                "splitLength": 10,
                "splitOverlap": 3
            }
        )

        index_settings = self.client.index(self.index_name).get_settings()

        expected_settings = {
            "type": "structured",
            "vectorNumericType": "float",
            "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 2,
                "splitMethod": "sentence",
                "splitOverlap": 0
            },
            "imagePreprocessing": {},
            "audioPreprocessing": {
                "splitLength": 10,
                "splitOverlap": 3
            },
            "videoPreprocessing": {
                "splitLength": 10,
                "splitOverlap": 3
            },
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {
                    "efConstruction": 512,
                    "m": 16
                }
            },
            "tensorFields": ["text_field_1", "text_field_2",
                             "video_field_1", "video_field_2", "audio_field", "image_field"],
            "allFields": [
                {"features": [], "name": "text_field_1", "type": "text"},
                {"features": [], "name": "text_field_2", "type": "text"},
                {"features": [], "name": "video_field_1", "type": "video_pointer"},
                {"features": [], "name": "video_field_2", "type": "video_pointer"},
                {"features": [], "name": "audio_field", "type": "audio_pointer"},
                {"features": [], "name": "image_field", "type": "image_pointer"},
            ]
        }

        self.assertEqual(expected_settings, index_settings)

    def test_create_simple_structured_index(self):
        self.client.create_index(index_name=self.index_name, type="structured",
                                 model="hf/all_datasets_v4_MiniLM-L6",
                                 all_fields=[{"name": "test", "type": "text",
                                              "features": ["lexical_search"]}],
                                 tensor_fields=["test"])
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        expected_index_settings = {
            'type': 'structured',
            'allFields': [{'name': 'test', 'type': 'text', 'features': ['lexical_search']}],
            'tensorFields': ['test'],
            'model': 'hf/all_datasets_v4_MiniLM-L6',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
            'imagePreprocessing': {},
            'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
            'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
            'vectorNumericType': 'float',
            'annParameters': {'spaceType': 'prenormalized-angular', 'parameters': {'efConstruction': 512, 'm': 16}}}
        self.assertEqual(expected_index_settings, index_settings)

    def test_create_structured_image_index(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="open_clip/ViT-B-32/laion400m_e32",
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]},
                                             {"name": "image", "type": "image_pointer"}],
                                 tensor_fields=["test", "image"])
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]

        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()

        self.assertEqual(["test", "image"], index_settings["tensorFields"])
        self.assertEqual("open_clip/ViT-B-32/laion400m_e32", index_settings["model"])

    def test_create_structured_index_with_custom_model(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="test-model",
                                 model_properties={"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                                   "dimensions": 384,
                                                   "tokens": 128,
                                                   "type": "hf"},
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]}],
                                 tensor_fields=["test"])
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual("test-model", index_settings['model'])
        self.assertEqual({"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                          "dimensions": 384,
                          "tokens": 128,
                          "type": "hf"}, index_settings['modelProperties'])

    def test_create_structured_image_index_with_preprocessing(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="open_clip/ViT-B-16/laion400m_e31",
                                 image_preprocessing={"patchMethod": "simple"},
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]},
                                             {"name": "image", "type": "image_pointer"}],
                                 tensor_fields=["test", "image"])
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]

        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()

        self.assertEqual(["test", "image"], index_settings["tensorFields"])
        self.assertEqual("open_clip/ViT-B-16/laion400m_e31", index_settings["model"])
        self.assertEqual("simple", index_settings['imagePreprocessing']['patchMethod'])

    def test_dash_in_index_name_structured(self):
        index_name = "test-index-test-index" + str(uuid.uuid4())
        self.client.create_index(index_name, type="structured",
                                 all_fields=[{"name": "title", "type": "text"}],
                                 tensor_fields=["title"])
        self.indexes_to_delete.append(index_name)

    def test_dash_in_index_name_unstructured(self):
        index_name = "test-index-test-index" + str(uuid.uuid4())
        self.client.create_index(index_name, type="unstructured")
        self.indexes_to_delete.append(index_name)

    def test_create_structured_image_index_with_custom_vector(self):
        """
        Tests if you can use the Python client to create a structured index with a custom vector
        """
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="open_clip/ViT-B-32/laion400m_e31",
                                 all_fields=[{"name": "my_custom_vector", "type": "custom_vector",
                                              "features": ["lexical_search", "filter"]}],
                                 tensor_fields=["my_custom_vector"],
                                 ann_parameters={
                                    "spaceType": "angular",
                                    "parameters": {"efConstruction": 512, "m": 16}
                                })

        # Random vectors for example purposes. replace these with your own.
        example_vector_1 = [i for i in range(512)]
        example_vector_2 = [1 / (i + 1) for i in range(512)]

        # We add the custom vector documents into our structured index.
        # We do NOT use mappings for custom vectors here.
        res = self.client.index(self.index_name).add_documents(
            documents=[
                {
                    "_id": "doc1",
                    "my_custom_vector": {
                        "vector": example_vector_1,
                        "content": "Singing audio file"
                    }
                },
                {
                    "_id": "doc2",
                    "my_custom_vector": {
                        "vector": example_vector_2,
                        "content": "Podcast audio file"
                    }
                }
            ]
        )
        self.assertEqual(res["errors"], False)

        # Tensor Search
        res = self.client.index(self.index_name).search(
            q={"dummy text": 0},
            context={"tensor":
                         [{"vector": example_vector_1, "weight": 1}]  # custom vector from doc1
                     }
        )
        self.assertEqual(res["hits"][0]["_id"], "doc1")

        # Make sure get documents works
        get_doc_res = self.client.index(self.index_name).get_documents(["doc1", "doc2"], expose_facets=True)
        self.assertEqual(get_doc_res["results"][0]["_id"], "doc1")
        self.assertEqual(get_doc_res["results"][0]["_found"], True)
        self.assertEqual(get_doc_res["results"][1]["_id"], "doc2")
        self.assertEqual(get_doc_res["results"][1]["_found"], True)

        # Lexical Search
        res = self.client.index(self.index_name).search(
            q="Podcast audio file", search_method="LEXICAL"
        )
        # self.assertEqual(res["hits"][0]["_id"], "doc2")     TODO: Investigate lexical search with custom vectors in structured

        # Filter search
        res = self.client.index(self.index_name).search(
            q="A rider is riding a horse jumping over the barrier.",
            filter_string="my_custom_vector:(Singing audio file)",
        )
        self.assertEqual(res["hits"][0]["_id"], "doc1")

        index_settings = self.client.index(self.index_name).get_settings()

        self.assertEqual([{'features': ['lexical_search', 'filter'], 'name': 'my_custom_vector', 'type': 'custom_vector'}],
                         index_settings['allFields'])

    def test_createIndexCanBlockOtherRequests(self):
        """Tests if create_index request can block other create/delete index requests."""
        # Create a new index

        index_name_1 = "test_index_" + str(uuid.uuid4())
        index_name_2 = "test_index_" + str(uuid.uuid4())

        def create_index():
            self.client.create_index(index_name=index_name_1)
            self.indexes_to_delete.append(index_name_1)

        t1 = threading.Thread(target=create_index)
        t1.start()
        time.sleep(0.5)

        try:
            with self.assertRaises(MarqoWebError) as e:
                self.client.create_index(index_name=index_name_2)
            self.assertIn("Your indexes are being updated. Please try again shortly.",
                          str(e.exception))

            with self.assertRaises(MarqoWebError) as e:
                self.client.delete_index(index_name=index_name_1)
            self.assertIn("Your indexes are being updated. Please try again shortly.",
                          str(e.exception))
        finally:
            t1.join()

    def test_deleteIndexCanBlockOtherRequests(self):
        """Test if delete_index request can block other create/delete index requests."""
        index_name_1 = "test_index_" + str(uuid.uuid4())
        index_name_2 = "test_index_" + str(uuid.uuid4())

        # Create a dummy index for deletion
        self.client.create_index(index_name=index_name_1)

        def delete_index():
            self.client.delete_index(index_name=index_name_1)

        t1 = threading.Thread(target=delete_index)
        t1.start()
        time.sleep(0.5)

        try:
            with self.assertRaises(MarqoWebError) as e:
                self.client.create_index(index_name=index_name_2)
            self.assertIn("Your indexes are being updated. Please try again shortly.",
                          str(e.exception))

            with self.assertRaises(MarqoWebError) as e:
                self.client.delete_index(index_name=index_name_1)
            self.assertIn("Your indexes are being updated. Please try again shortly.",
                          str(e.exception))
        finally:
            t1.join()

    def test_indexNotFoundErrorNotRaised(self):
        """Test to ensure index_not_found error is not raised but returned as message in delete_index response"""
        index_name = "test_index_" + str(uuid.uuid4())
        res = self.client.delete_index(index_name=index_name)
        self.assertEqual(res["message"], f"Index {index_name} not found")
        self.assertEqual(res["code"], "index_not_found")
