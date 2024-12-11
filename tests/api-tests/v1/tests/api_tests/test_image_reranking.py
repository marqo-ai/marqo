import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


def generate_structured_index_settings_dict(index_name, image_preprocessing_method):
    return {
        "indexName": index_name,
        "type": "structured",
        "model": "open_clip/ViT-B-32/openai",
        "allFields": [{"name": "image_content_1", "type": "image_pointer"},
                      {"name": "image_content_2", "type": "image_pointer"},
                      {"name": "text_content", "type": "text"}],
        "tensorFields": ["image_content_1", "image_content_2", "text_content"],
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


class TestImageReranking(MarqoTestCase):
    """Test image reranking features. Note that this feature is available only for structured indexes as
    the feature requires searchable attributes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_no_image_processing_index_name = (
                "structured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_simple_image_processing_index_name = (
                "structured_simple_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        cls.create_indexes([
            generate_structured_index_settings_dict(cls.structured_no_image_processing_index_name, None),
            generate_structured_index_settings_dict(cls.structured_simple_image_processing_index_name, "simple"),
        ])

        cls.indexes_to_delete = [
            cls.structured_no_image_processing_index_name,
            cls.structured_simple_image_processing_index_name,
        ]

    def test_image_reranking(self):
        documents = [{'_id': '1',
                      'image_content_1': 'https://avatars.githubusercontent.com/u/13092433?v=4'},
                     {'_id': '2',
                      'image_content_1': 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png'},
                     ]
        index_name = self.structured_no_image_processing_index_name
        self.client.index(index_name).add_documents(documents)

        results_1 = self.client.index(index_name).search('brain', searchable_attributes=['image_content_1'])
        self.assertEqual(documents[0]['image_content_1'], results_1['hits'][0]['image_content_1'])
        results_2 = self.client.index(index_name).search('hippo', searchable_attributes=['image_content_1'])
        self.assertEqual(documents[1]['image_content_1'], results_2['hits'][0]['image_content_1'])
        # proper way to search over images with reranking
        reranking_results = self.client.index(index_name).search('brain', reranker='google/owlvit-base-patch32'
                                                                 , searchable_attributes=['image_content_1'])

        self.assertEqual(documents[0]['image_content_1'], reranking_results['hits'][0]['image_content_1'])
        self.assertIn("image_content_1", reranking_results['hits'][0]['_highlights'][0])

    def test_image_reranking_searchable_is_none(self):
        documents = [{'_id': '1',
                      'image_content_1': 'https://avatars.githubusercontent.com/u/13092433?v=4'},
                     {'_id': '2',
                      'image_content_1': 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png'},
                     ]
        index_name = self.structured_no_image_processing_index_name
        self.client.index(index_name).add_documents(documents)

        expected_error_message = 'searchable_attributes cannot be None'
        with self.assertRaises(MarqoWebError) as e:
            _ = self.client.index(index_name).search('brain', searchable_attributes=None,
                                                     reranker='google/owlvit-base-patch32')
        self.assertIn(expected_error_message, str(e.exception.message))

    def test_image_reranking_model_name_error(self):
        documents = [{'_id': '1',
                      'image_content_1': 'https://avatars.githubusercontent.com/u/13092433?v=4'},
                     {'_id': '2',
                      'image_content_1': 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png'},
                     ]
        index_name = self.structured_no_image_processing_index_name
        self.client.index(index_name).add_documents(documents)
        with self.assertRaises(MarqoWebError) as e:
            results = self.client.index(index_name).search('brain', searchable_attributes=['location'],
                                                           reranker='google/owlvi-base-patch32')
            self.assertIn("could not find model_name=", str(e.exception.message))

    def test_image_reranking_with_chunking(self):
        documents = [
            {
                '_id': '1',
                'text_content': 'the image chunking can (optionally) chunk the image into sub-patches (aking to segmenting text) by using either a learned model or simple box generation and cropping',
                'image_content_1': 'https://avatars.githubusercontent.com/u/13092433?v=4'
            },
            {
                '_id': '2',
                'text_content': 'ing either a learned model or simple box generation and cropping. brain',
                'image_content_1': 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png'
            },
        ]

        self.client.index(self.structured_simple_image_processing_index_name).add_documents(documents)

        results = self.client.index(self.structured_simple_image_processing_index_name). \
            search('brain', searchable_attributes=["image_content_1"])
        r = results['hits'][0]['_highlights'][0]['image_content_1']
        self.assertTrue(isinstance(eval(r), list))
        self.assertEqual(4, len(eval(r)))
