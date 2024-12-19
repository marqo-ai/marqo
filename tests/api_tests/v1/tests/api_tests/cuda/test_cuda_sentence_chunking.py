import uuid

import pytest

from tests.marqo_test import MarqoTestCase


@pytest.mark.cuda_test
class TestCudaSentenceChunking(MarqoTestCase):
    """Test for sentence chunking

    Assumptions:
        - Local OpenSearch (not S2Search)
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # A very large split length
        cls.large_structured_index_name = "structured_large" + str(uuid.uuid4()).replace('-', '')
        # A standard split length 2, with 0 split overlap
        cls.standard_structured_index_name = "structured_standard" + str(uuid.uuid4()).replace('-', '')
        # A standard split length 2, with 1 split overlap
        cls.overlap_structured_index_name = "structured_overlap" + str(uuid.uuid4()).replace('-', '')

        cls.large_unstructured_index_name = "unstructured_large" + str(uuid.uuid4()).replace('-', '')
        cls.standard_unstructured_index_name = "unstructured_standard" + str(uuid.uuid4()).replace('-', '')
        cls.overlap_unstructured_index_name = "unstructured_overlap" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            # Structured Indexes
            {
                "indexName": cls.large_structured_index_name,
                "type": "structured",
                "textPreprocessing": {
                    "splitLength": int(1e3),
                    "splitOverlap": 0,
                    "splitMethod": "sentence"
                },
                "allFields": [{"name": "text_field_1", "type": "text"},
                              {"name": "text_field_2", "type": "text"},
                              {"name": "text_field_3", "type": "text"}],
                "tensorFields": ["text_field_1", "text_field_2", "text_field_3"]
            },
            {
                "indexName": cls.standard_structured_index_name,
                "type": "structured",
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence"
                },
                "allFields": [{"name": "text_field_1", "type": "text"},
                              {"name": "text_field_2", "type": "text"},
                              {"name": "text_field_3", "type": "text"}],
                "tensorFields": ["text_field_1", "text_field_2", "text_field_3"]
            },
            {
                "indexName": cls.overlap_structured_index_name,
                "type": "structured",
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 1,
                    "splitMethod": "sentence"
                },
                "allFields": [{"name": "text_field_1", "type": "text"},
                              {"name": "text_field_2", "type": "text"},
                              {"name": "text_field_3", "type": "text"}],
                "tensorFields": ["text_field_1", "text_field_2", "text_field_3"]
            },
            # Unstructured Indexes
            {
                "indexName": cls.large_unstructured_index_name,
                "type": "unstructured",
                "textPreprocessing": {
                    "splitLength": int(1e3),
                    "splitOverlap": 0,
                    "splitMethod": "sentence"
                },
            },
            {
                "indexName": cls.standard_unstructured_index_name,
                "type": "unstructured",
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence"
                },
            },
            {
                "indexName": cls.overlap_unstructured_index_name,
                "type": "unstructured",
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 1,
                    "splitMethod": "sentence"
                },
            }
        ])

        cls.indexes_to_delete = [
            cls.large_structured_index_name,
            cls.standard_structured_index_name,
            cls.overlap_structured_index_name,

            cls.large_unstructured_index_name,
            cls.standard_unstructured_index_name,
            cls.overlap_unstructured_index_name
        ]

    def test_sentence_no_chunking(self):
        document = {'_id': '1',  # '_id' can be provided but is not required
                    'text_field_1': 'hello. how are you. another one.',
                    'text_field_2': 'the image chunking. can (optionally) chunk. the image into sub-patches (aking to segmenting text). by using either. a learned model. or simple box generation and cropping.',
                    'text_field_3': 'sasasasaifjfnonfqeno asadsdljknjdfln'}

        unstructured_tensor_fields = ["text_field_1", "text_field_2", "text_field_3"]

        for index_name in [self.large_structured_index_name, self.large_structured_index_name]:
            with self.subTest(index_name):
                self.client.index(index_name).add_documents([document], tensor_fields=unstructured_tensor_fields if \
                    index_name.startswith("un") else None, device="cuda")

                # test the search works
                results = self.client.index(index_name).search('hello how are you', device="cuda")
                self.assertEqual(document["text_field_1"], results['hits'][0]['text_field_1'])
                self.assertEqual(document["text_field_2"], results['hits'][0]['text_field_2'])
                self.assertEqual(document["text_field_3"], results['hits'][0]['text_field_3'])
                self.assertEqual(document["text_field_1"], results["hits"][0]["_highlights"][0]["text_field_1"])

    def test_sentence_chunking_no_overlap(self):
        test_cases = [
            ('hello. how are you.', 'hello. how are you.'),
            ('the image into sub-patches (aking to segmenting text). by using either.',
             'the image into sub-patches (aking to segmenting text). by using either.'),
            ('sasasasaifjfnonfqeno asadsdljknjdfln', 'sasasasaifjfnonfqeno asadsdljknjdfln'),
            ('can (optionally) chunk.', "the image chunking. can (optionally) chunk."),
            ("the image chunking. can (optionally) chunk.",
             "the image chunking. can (optionally) chunk."),
            ("the image into sub-patches (aking to segmenting text). by using either.",
             "the image into sub-patches (aking to segmenting text). by using either.")
        ]

        document = {'_id': '1',  # '_id' can be provided but is not required
                    'text_field_1': 'hello. how are you. another one.',
                    'text_field_2': 'the image chunking. can (optionally) chunk. the image into sub-patches (aking to segmenting text). by using either. a learned model. or simple box generation and cropping.',
                    'text_field_3': 'sasasasaifjfnonfqeno asadsdljknjdfln'}

        unstructured_tensor_fields = ["text_field_1", "text_field_2", "text_field_3"]

        for index_name in [self.standard_unstructured_index_name, self.standard_unstructured_index_name]:
            for search_term, expected_highlights_chunk in test_cases:
                with self.subTest(f"{search_term}, {index_name}"):
                    self.client.index(index_name).add_documents(
                        documents=[document], tensor_fields=unstructured_tensor_fields if \
                            index_name.startswith("un") else None, device="cuda"
                    )

                    res = self.client.index(index_name).search(search_term, device="cuda")
                    returned_highlights = list(res["hits"][0]["_highlights"][0].values())[0]
                    self.assertEqual(expected_highlights_chunk, returned_highlights)

    def test_sentence_chunking_overlap(self):
        test_cases = [
            ('hello. how are you.', 'hello. how are you.'),
            ('the image into sub-patches (aking to segmenting text). by using either.',
             'the image into sub-patches (aking to segmenting text). by using either.'),
            ('sasasasaifjfnonfqeno asadsdljknjdfln', 'sasasasaifjfnonfqeno asadsdljknjdfln'),
            ('can (optionally) chunk.', "the image chunking. can (optionally) chunk."),
            ("can (optionally) chunk. the image into sub-patches (aking to segmenting text).",
             "can (optionally) chunk. the image into sub-patches (aking to segmenting text)."),
            ("the image into sub-patches (aking to segmenting text). by using either.",
             "the image into sub-patches (aking to segmenting text). by using either.")
        ]

        document = {'_id': '1',  # '_id' can be provided but is not required
                    'text_field_1': 'hello. how are you. another one.',
                    'text_field_2': 'the image chunking. can (optionally) chunk. '
                                    'the image into sub-patches (aking to segmenting text). '
                                    'by using either. a learned model. or simple box generation and cropping.',
                    'text_field_3': 'sasasasaifjfnonfqeno asadsdljknjdfln'}

        unstructured_tensor_fields = ["text_field_1", "text_field_2", "text_field_3"]

        for index_name in [self.overlap_unstructured_index_name, self.overlap_unstructured_index_name]:
            for search_term, expected_highlights_chunk in test_cases:
                with self.subTest(f"{search_term}, {index_name}"):
                    self.client.index(index_name).add_documents(
                        documents=[document], tensor_fields=unstructured_tensor_fields if \
                            index_name.startswith("un") else None, device="cuda"
                    )

                    res = self.client.index(index_name).search(search_term, device="cuda")
                    returned_highlights = list(res["hits"][0]["_highlights"][0].values())[0]
                    self.assertEqual(expected_highlights_chunk, returned_highlights)
