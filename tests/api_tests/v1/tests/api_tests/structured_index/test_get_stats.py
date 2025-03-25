import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestStructuredGetStats(MarqoTestCase):
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
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "non_tensor", "type": "text"},
                    {"name": "my_multi_modal_field",
                     "type": "multimodal_combination",
                     "dependentFields": {
                         "title": 0.5, "content": 0.5
                     }
                     },
                ],
                "tensorFields": ["title", "content", "my_multi_modal_field"],
            },
            {
                "indexName": cls.image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "title", "type": "text"},
                    {"name": "image_content", "type": "image_pointer"},
                    {"name": "non_tensor", "type": "text"},

                    {"name": "my_multi_modal_field",
                     "type": "multimodal_combination",
                     "dependentFields": {
                        "title": 0.5, "image_content": 0.5
                        }
                    },
                ],
                "tensorFields": ["title", "image_content", "my_multi_modal_field"],
            }
        ])
        cls.indexes_to_delete= [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    
    def test_get_status_response_format(self):
        res = self.client.index(self.text_index_name).get_stats()
        assert isinstance(res, dict)
        assert "numberOfVectors" in res
        assert "numberOfDocuments" in res
        assert "backend" in res
        assert "memoryUsedPercentage" in res["backend"]
        assert "storageUsedPercentage" in res["backend"]

    
    def test_get_status_response_results_text_index(self):
        """Ensure that the number of vectors and documents is correct, with or without mappings"""
        test_cases = [
            ([
                {"title": "test-2", "content": "test"},  # 3 vectors (with multimodal_field)
                {"title": "test-2", "content": "test", "non_tensor": "test"},  # 3 vectors (with multimodal_field)
                {"title": "test"},  # 2 vectors (with multimodal_field)
                {"non_tensor": "test"}  # 0 vector
            ], None, 4, 8, "No mappings"),
            ([
                 {"title": "test-2", "content": "test"},  # 3 vectors (with multimodal_field)
                 {"title": "test-2", "content": "test", "non_tensor": "test"},  # 3 vectors (with multimodal_field)
                 {"title": "test"},  # 2 vectors (with multimodal_field)
                 {"non_tensor": "test"}  # 0 vector
            ], {"my_multi_modal_field": {"type": "multimodal_combination", "weights": {"title": 0.5, "content": 0.8}}},
             4, 8, "With mappings"),
        ]

        for documents, mappings, number_of_documents, number_of_vectors, msg in test_cases:
            self.clear_indexes([self.text_index_name])
            with self.subTest(msg):
                self.client.index(self.text_index_name).add_documents(
                    documents=documents,
                    device="cpu",
                    mappings=mappings
                )

                res = self.client.index(self.text_index_name).get_stats()
                self.assertEqual(number_of_documents, res["numberOfDocuments"])
                self.assertEqual(number_of_vectors, res["numberOfVectors"])

    
    def test_get_status_response_results_image_index(self):
        """Ensure that the number of vectors and documents is correct, with or without mappings"""

        image_content = "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg"

        test_cases = [
            ([
                 {"title": "test-2", "image_content": image_content},  # 3 vectors (with multimodal_field)
                 # 3 vectors (with multimodal_field)
                 {"title": "test-2", "image_content": image_content, "non_tensor": "test"},
                 {"title": "test"},  # 2 vectors (with multimodal_field)
                 {"image_content": image_content},  # 2 vectors (with multimodal_field)
                 {"non_tensor": "test"}  # 0 vector
             ], None, 5, 10, "No mappings"),
            ([
                 {"title": "test-2", "image_content": image_content},  # 3 vectors (with multimodal_field)
                 # 3 vectors (with multimodal_field)
                 {"title": "test-2", "image_content": image_content, "non_tensor": "test"},
                 {"title": "test"},  # 2 vectors (with multimodal_field)
                 {"image_content": image_content},  # 2 vectors (with multimodal_field)
                 {"non_tensor": "test"}  # 0 vector
             ],
             {"my_multi_modal_field": {"type": "multimodal_combination",
                                        "weights": {"title": 0.5, "image_content": 0.8}}},
             5, 10, "With mappings"),
        ]

        for documents, mappings, number_of_documents, number_of_vectors, msg in test_cases:
            self.clear_indexes([self.image_index_name])
            with self.subTest(msg):
                self.client.index(self.image_index_name).add_documents(
                    documents=documents,
                    device="cpu",
                    mappings=mappings
                )

                res = self.client.index(self.image_index_name).get_stats()
                self.assertEqual(number_of_documents, res["numberOfDocuments"])
                self.assertEqual(number_of_vectors, res["numberOfVectors"])

    def test_get_status_error(self):
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index("A void index").get_stats()
        self. assertIn("index_not_found", str(cm.exception.message))