import unittest
import uuid
from typing import Dict

from tests.marqo_test import MarqoTestCase


def generate_structured_index_settings_dict(index_name, image_preprocessing_method) -> Dict:
    return {
        "indexName": index_name,
        "type": "structured",
        "model": "open_clip/ViT-B-32/openai",
        "allFields": [{"name": "image_content", "type": "image_pointer"},
                      {"name": "text_content", "type": "text"}],
        "tensorFields": ["image_content", "text_content"],
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


def generate_unstructured_index_settings_dict(index_name, image_preprocessing_method) -> Dict:
    return {
        "indexName": index_name,
        "type": "unstructured",
        "model": "open_clip/ViT-B-32/openai",
        "treatUrlsAndPointersAsImages": True,
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


@unittest.skip(reason="Skipped due to image chunk support is removed")
class TestUnstructuredImageChunking(MarqoTestCase):
    """Test for image chunking as a preprocessing step
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.unstructured_no_image_processing_index_name = (
                "unstructured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_simple_image_processing_index_name = (
                "unstructured_simple_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_frcnn_image_processing_index_name = (
                "unstructured_frcnn_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_dino_v1_image_processing_index_name = (
                "unstructured_dino_v1_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_dino_v2_image_processing_index_name = (
                "unstructured_dino_v2_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_marqo_yolo_image_processing_index_name = (
                "unstructured_marqo_yolo_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        cls.structured_no_image_processing_index_name = (
                "structured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_simple_image_processing_index_name = (
                "structured_simple_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_frcnn_image_processing_index_name = (
                "structured_frcnn_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_dino_v1_image_processing_index_name = (
                "structured_dino_v1_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_dino_v2_image_processing_index_name = (
                "structured_dino_v2_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_marqo_yolo_image_processing_index_name = (
                "structured_marqo_yolo_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        # create the structured indexes
        cls.create_indexes(
            [
                generate_structured_index_settings_dict(cls.structured_no_image_processing_index_name, None),
                generate_structured_index_settings_dict(cls.structured_simple_image_processing_index_name, "simple"),
                generate_structured_index_settings_dict(cls.structured_frcnn_image_processing_index_name, "frcnn"),
                generate_structured_index_settings_dict(cls.structured_dino_v1_image_processing_index_name, "dino-v1"),
                generate_structured_index_settings_dict(cls.structured_dino_v2_image_processing_index_name, "dino-v2"),
                generate_structured_index_settings_dict(cls.structured_marqo_yolo_image_processing_index_name, "marqo-yolo"),
            ]
        )

        # create the unstructured indexes
        cls.create_indexes(
            [
                generate_unstructured_index_settings_dict(cls.unstructured_no_image_processing_index_name, None),
                generate_unstructured_index_settings_dict(cls.unstructured_simple_image_processing_index_name, "simple"),
                generate_unstructured_index_settings_dict(cls.unstructured_frcnn_image_processing_index_name, "frcnn"),
                generate_unstructured_index_settings_dict(cls.unstructured_dino_v1_image_processing_index_name, "dino-v1"),
                generate_unstructured_index_settings_dict(cls.unstructured_dino_v2_image_processing_index_name, "dino-v2"),
                generate_unstructured_index_settings_dict(cls.unstructured_marqo_yolo_image_processing_index_name, "marqo-yolo"),
            ]
        )

        cls.indexes_to_delete = [
            cls.unstructured_no_image_processing_index_name,
            cls.unstructured_simple_image_processing_index_name,
            cls.unstructured_frcnn_image_processing_index_name,
            cls.unstructured_dino_v1_image_processing_index_name,
            cls.unstructured_dino_v2_image_processing_index_name,
            cls.unstructured_marqo_yolo_image_processing_index_name,

            cls.structured_no_image_processing_index_name,
            cls.structured_simple_image_processing_index_name,
            cls.structured_frcnn_image_processing_index_name,
            cls.structured_dino_v1_image_processing_index_name,
            cls.structured_dino_v2_image_processing_index_name,
            cls.structured_marqo_yolo_image_processing_index_name
        ]

    def test_image_no_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_no_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, no image preprocessing"),
            (self.structured_no_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, no image preprocessing"),
        ]

        document = {
            '_id': '1', # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)
                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be the location
                self.assertEqual(temp_file_name, results['hits'][0]['_highlights'][0]['image_content'])

    def test_image_simple_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_simple_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, simple image preprocessing"),
            (self.structured_simple_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, simple image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing the bounding box, in string format
                r = results['hits'][0]['_highlights'][0]['image_content']
                self.assertTrue(isinstance(eval(r), list))
                self.assertEqual(4, len(eval(r)))

    def test_image_frcnn_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_frcnn_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, frcnn image preprocessing"),
            (self.structured_frcnn_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, frcnn image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing
                r = results['hits'][0]['_highlights'][0]['image_content']
                self.assertTrue(isinstance(eval(r), list))
                self.assertEqual(4, len(eval(r)))
                
    def test_image_dino_v1_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_dino_v1_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, dino_v1 image preprocessing"),
            (self.structured_dino_v1_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, dino_v1 image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing
                r = results['hits'][0]['_highlights'][0]['image_content']
                self.assertTrue(isinstance(eval(r), list))
                self.assertEqual(4, len(eval(r)))
    
    def test_image_dino_v2_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_dino_v2_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, dino_v2 image preprocessing"),
            (self.structured_dino_v2_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, dino_v2 image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing
                r = results['hits'][0]['_highlights'][0]['image_content']
                self.assertTrue(isinstance(eval(r), list))
                self.assertEqual(4, len(eval(r)))
                
    def test_image_marqo_yolo_chunking(self):

        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_marqo_yolo_image_processing_index_name, {"tensor_fields": ["image_content"]},
             {"searchable_attributes": ["image_content"]}, "unstructured_index, marqo_yolo image preprocessing"),
            (self.structured_marqo_yolo_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, marqo_yolo image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing
                r = results['hits'][0]['_highlights'][0]['image_content']
                self.assertTrue(isinstance(eval(r), list))
                self.assertEqual(4, len(eval(r)))