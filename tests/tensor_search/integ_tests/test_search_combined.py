import os
import uuid
from unittest import mock
import torch
import pytest

import marqo.core.exceptions as core_exceptions
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase, TestImageUrls
from marqo import exceptions as base_exceptions
from marqo.core.models.marqo_query import MarqoLexicalQuery
from marqo.core.models.score_modifier import ScoreModifierType, ScoreModifier
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.tensor_search.models.api_models import SearchQuery
from pydantic import ValidationError


class TestSearch(MarqoTestCase):
    """
    Combined tests for unstructured and structured search.
    Currently only supports filtering tests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        unstructured_default_text_index_encoded_name = cls.unstructured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', '')
        )

        unstructured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True
        )

        unstructured_image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_image_index_with_random_model = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_languagebind_index = cls.unstructured_marqo_index_request(
            model=Model(name='LanguageBind/Video_V1.5_FT_Audio_FT_Image'),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True
        )

        # STRUCTURED indexes
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_4", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_5", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_6", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_7", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_8", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="int_field_1", type=FieldType.Int,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="float_field_1", type=FieldType.Float,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="bool_field_1", type=FieldType.Bool,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="bool_field_2", type=FieldType.Bool,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="long_field_1", type=FieldType.Long, features=[FieldFeature.Filter]),
                FieldRequest(name="double_field_1", type=FieldType.Double, features=[FieldFeature.Filter]),
                FieldRequest(name="custom_vector_field_1", type=FieldType.CustomVector, features=[FieldFeature.Filter]),
                FieldRequest(name="multimodal_field_1", type=FieldType.MultimodalCombination,
                             dependent_fields={"text_field_7": 0.1, "text_field_8": 0.1})
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3",
                           "text_field_4", "text_field_5", "text_field_6",
                           "custom_vector_field_1", "multimodal_field_1"]
        )

        structured_default_text_index_encoded_name = cls.structured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
        )

        structured_default_image_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "text_field_3", "image_field_1", "image_field_2"]
        )

        structured_image_index_with_random_model = cls.structured_marqo_index_request(
            model=Model(name='random/small'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer)
            ],
            tensor_fields=["text_field_1", "text_field_2", "image_field_1"]
        )

        structured_languagebind_index = cls.structured_marqo_index_request(
            name="my-multimodal-index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="video_field_1", type=FieldType.VideoPointer),
                FieldRequest(name="audio_field_1", type=FieldType.AudioPointer),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
            ],
            model=Model(name="LanguageBind/Video_V1.5_FT_Audio_FT_Image"),
            tensor_fields=["text_field_1",
                        "video_field_1", "audio_field_1", "image_field_1"],
            normalize_embeddings=True,
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            unstructured_default_text_index_encoded_name,
            unstructured_default_image_index,
            unstructured_image_index_with_chunking,
            unstructured_image_index_with_random_model,
            unstructured_languagebind_index,
            structured_default_text_index,
            structured_default_text_index_encoded_name,
            structured_default_image_index,
            structured_image_index_with_random_model,
            structured_languagebind_index
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.unstructured_default_text_index_encoded_name = cls.indexes[1]
        cls.unstructured_default_image_index = cls.indexes[2]
        cls.unstructured_image_index_with_chunking = cls.indexes[3]
        cls.unstructured_image_index_with_random_model = cls.indexes[4]
        cls.unstructured_languagebind_index = cls.indexes[5]
        cls.structured_default_text_index = cls.indexes[6]
        cls.structured_default_text_index_encoded_name = cls.indexes[7]
        cls.structured_default_image_index = cls.indexes[8]
        cls.structured_image_index_with_random_model = cls.indexes[9]
        cls.structured_languagebind_index = cls.indexes[10]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    @pytest.mark.largemodel
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
    def test_search_video(self):
        documents = [
            {"video_field_1": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4", "_id": "1"},
            # Replace the audio link with something marqo-hosted
            {"audio_field_1": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3", "_id": "2"}, 
            {"image_field_1": TestImageUrls.HIPPO_REALISTIC_LARGE.value, "_id": "3"},
            # {"image_field_1": TestImageUrls.HIPPO_REALISTIC.value, "_id": "5"}, # png image with palette is not supported
            {"text_field_1": "hello there padawan. Today you will begin your training to be a Jedi", "_id": "4"},
        ]
        for index in [self.unstructured_languagebind_index, self.structured_languagebind_index]:
            with self.subTest(index=index.type):
                response = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=documents,
                        tensor_fields=["text_field_1",
                        "video_field_1", "audio_field_1", "image_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Search using the video
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4"
                )

                # Assertions
                self.assertEqual(len(results['hits']), 3)  # 3 documents should be returned (limit=3)
                self.assertEqual(results['hits'][0]['_id'], "1")  # The video document should be the top result
                self.assertGreater(results['hits'][0]['_score'], results['hits'][1]['_score'])  # Video should have higher score

    @pytest.mark.largemodel
    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
    def test_search_audio(self):
        documents = [
            {"video_field_1": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4", "_id": "1"},
            # Replace the audio link with something marqo-hosted
            {"audio_field_1": "https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3", "_id": "2"}, 
            {"image_field_1": TestImageUrls.HIPPO_REALISTIC_LARGE.value, "_id": "3"},
            # {"image_field_1": TestImageUrls.HIPPO_REALISTIC.value, "_id": "5"},  # png file with palette is not supported
            {"text_field_1": "hello there padawan. Today you will begin your training to be a Jedi", "_id": "4"},
        ]
        for index in [self.unstructured_languagebind_index, self.structured_languagebind_index]:
            with self.subTest(index=index.type):
                response = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=documents,
                        tensor_fields=["text_field_1",
                        "video_field_1", "audio_field_1", "image_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Search using the audio
                results = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="https://marqo-ecs-50-audio-test-dataset.s3.amazonaws.com/audios/marqo-audio-test.mp3"
                )
                
                # Assertions
                self.assertEqual(len(results['hits']), 3)  # 3 documents should be returned (limit=3)
                self.assertEqual(results['hits'][0]['_id'], "2")  # The audio document should be the top result
                self.assertGreater(results['hits'][0]['_score'], results['hits'][1]['_score'])  # Audio should have higher score


    def test_filtering_list_case_tensor(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(type=index.type):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1235", "text_field_1": "some text", "list_field_1": ["tag1", "tag2 some"]}
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] if isinstance(index,
                                                                                                     UnstructuredMarqoIndex) else None
                    )
                )

                test_cases = [
                    ("list_field_1:tag1", 1, "1235", True),
                    ("list_field_1:tag55", 0, None, False),
                    ("text_field_3:b", 1, "5678", True),
                    ("list_field_1:tag2", 0, None, False),
                    ("list_field_1:(tag2 some)", 1, "1235", True),
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        # As long as at least 1 tag in the list overlaps, it's a success.
                        ("list_field_1 in (tag1, tag5)", 1, "1235", True),
                        ("list_field_1 in ((tag2 some), random)", 1, "1235", True),
                        ("list_field_1 in (tag2, random)", 0, None, False),  # incomplete match for "tag2 some"
                        ("text_field_3 in (b, c)", 1, "5678", True),
                        ("text_field_3 in (a, c, (b but wrong))", 0, None, False),  # incomplete match for "b"
                        ("int_field_1 in (1, 2, 3)", 1, "1234", True)  # int
                    ]

                for filter_query, expected_count, expected_id, highlight_exists in test_cases:
                    with self.subTest(filter_query=filter_query):
                        res = tensor_search.search(
                            index_name=index.name, config=self.config, text="", filter=filter_query)

                        assert len(res["hits"]) == expected_count
                        if expected_id:
                            assert res["hits"][0]["_id"] == expected_id
                            assert ("_highlights" in res["hits"][0]) == highlight_exists

    def test_filtering_list_case_lexical(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1235", "text_field_1": "some text", "list_field_1": ["tag1", "tag2 some"]}
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    ),
                )

                test_cases = [
                    ("list_field_1:tag1", 1, "1235"),
                    ("list_field_1:tag55", 0, None),
                    ("text_field_3:b", 1, "5678"),
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        # As long as at least 1 tag in the list overlaps, it's a success.
                        ("list_field_1 in (tag1, tag5)", 1, "1235"),
                        ("list_field_1 in ((tag2 some), random)", 1, "1235"),
                        ("list_field_1 in (tag2, random)", 0, None),  # incomplete match for "tag2 some"
                        ("text_field_3 in (b, c)", 1, "5678"),
                        ("text_field_3 in (a, c, (b but wrong))", 0, None),  # incomplete match for "b"
                        ("int_field_1 in (1, 2, 3, 4)", 1, "1234")  # int
                    ]

                for filter_string, expected_hits, expected_id in test_cases:
                    with self.subTest(
                            f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_id}"):
                        res = tensor_search.search(
                            index_name=index.name, config=self.config, text="some",
                            search_method=SearchMethod.LEXICAL, filter=filter_string
                        )
                        self.assertEqual(expected_hits, len(res["hits"]))
                        if expected_id:
                            self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_filtering_list_case_image(self):
        for index in [self.unstructured_default_image_index, self.structured_default_image_index]:
            with self.subTest(index=index):
                hippo_img = TestImageUrls.HIPPO_REALISTIC.value
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"image_field_1": hippo_img, "text_field_1": "some text", "text_field_2": "baaadd",
                             "_id": "5678",
                             "text_field_3": "b"},
                            {"image_field_1": hippo_img, "text_field_1": "some text",
                             "text_field_2": "Close match hehehe",
                             "_id": "1234", "int_field_1": 2},
                            {"image_field_1": hippo_img, "text_field_1": "some text", "_id": "1235",
                             "list_field_1": ["tag1", "tag2 some"]}
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3", "image_field_1"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                test_cases = [
                    ("list_field_1:tag1", 1, "1235"),
                    ("list_field_1:tag55", 0, None),
                    ("text_field_3:b", 1, "5678"),
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        # As long as at least 1 tag in the list overlaps, it's a success.
                        ("list_field_1 in (tag1, tag5)", 1, "1235"),
                        ("list_field_1 in ((tag2 some), random)", 1, "1235"),
                        ("list_field_1 in (tag2, random)", 0, None),  # incomplete match for "tag2 some"
                        ("text_field_3 in (b, c)", 1, "5678"),
                        ("text_field_3 in (a, c, (b but wrong))", 0, None),  # incomplete match for "b"
                    ]

                for filter_string, expected_hits, expected_id in test_cases:
                    with self.subTest(
                            f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_id}"):
                        res = tensor_search.search(
                            index_name=index.name, config=self.config, text="some",
                            search_method=SearchMethod.TENSOR, filter=filter_string
                        )

                        self.assertEqual(expected_hits, len(res["hits"]))
                        if expected_id:
                            self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_filtering(self):
        # TODO: remove
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents first
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1233", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "bool_field_1": True},
                            {"_id": "1232", "text_field_1": "true"},
                            {"_id": "1231", "text_field_1": "some text", "bool_field_2": False},
                            {"_id": "in1", "text_field_1": "random1", "int_field_1": 100,
                             "text_field_7": "multimodal red herring"},

                            {"_id": "in2", "text_field_1": "blahblah", "int_field_1": 200, "long_field_1": 300,
                             "text_field_7": "multimodal correct",
                             "text_field_8": "multimodal correct",
                             "custom_vector_field_1": {
                                 "content": "custom vector text!",
                                 "vector": [i for i in range(384)]
                             }},
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3", "custom_vector_field_1",
                                       "multimodal_field_1"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "custom_vector_field_1": {
                                "type": "custom_vector",
                            },
                            "multimodal_field_1": {
                                "type": "multimodal_combination",
                                "weights": {
                                    "text_field_7": 0.1,
                                    "text_field_8": 0.1
                                }
                            }
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Define test parameters
                test_cases = [
                    ("text_field_3:c", 0, None),
                    ("int_field_1:2", 1, ["1234"]),
                    ("text_field_3:b", 1, ["5678"]),
                    ("int_field_1:5", 0, None),
                    ("int_field_1:[5 TO 30]", 0, None),
                    ("int_field_1:[0 TO 30]", 1, ["1234"]),
                    ("bool_field_1:true", 1, ["1233"]),
                    ("bool_field_1:True", 1, ["1233"]),
                    ("bool_field_1:tRue", 1, ["1233"]),
                    ("bool_field_2:false", 1, ["1231"]),
                    ("bool_field_1:false", 0, None),  # no hits for bool_field_1=false
                    ("bool_field_1:some_value", 0, None),  # no hits for bool_field_1 not boolean
                    ("int_field_1:[0 TO 30] OR bool_field_1:true", 2, None),
                    ("(int_field_1:[0 TO 30] AND int_field_1:2) AND text_field_1:(some text)", 1, ["1234"]),
                    ("text_field_1:true", 1, ["1232"]),  # string field with boolean-like value
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        # normal string in
                        ("text_field_1 in (random1, true)", 2, ["in1", "1232"]),
                        # normal int in
                        ("int_field_1 in (100, 200)", 2, ["in1", "in2"]),
                        # normal long in
                        ("long_field_1 in (299, 300)", 1, ["in2"]),
                        # normal custom vector in
                        ("custom_vector_field_1 in ((custom vector text!))", 1, ["in2"]),
                        # multimodal subfield in
                        ("text_field_7 in ((multimodal correct)) AND text_field_8 in ((multimodal correct))", 1,
                         ["in2"]),
                        # in with AND
                        ("text_field_1 in (random1, true) AND int_field_1:100", 1, ["in1"]),
                        # in with OR
                        ("text_field_1 in (random1, true) OR text_field_2:baaadd", 3, ["in1", "1232", "5678"]),
                        # in with RANGE
                        ("text_field_1 in (random1, true) OR int_field_1:[90 TO 210]", 3, ["in1", "1232", "in2"]),
                        # in with 1 result, 1 list item
                        ("text_field_1 in (random1)", 1, ["in1"]),
                        # in with no results
                        ("text_field_1 in (blahblahblah)", 0, None),
                        # NOT with in
                        ("NOT text_field_1 in (random1, true)", 5, ["5678", "1234", "1233", "1231", "in2"]),
                        # combining string in and int in
                        ("text_field_1 in (random1, true) AND int_field_1 in (100, 200)", 1, ["in1"]),
                        # int in with no results
                        ("int_field_1 in (123, 456, 789)", 0, None),
                        # Filtering on empty string returns no results
                        ("text_field_1 in ()", 0, None),
                    ]
                for filter_string, expected_hits, expected_ids in test_cases:
                    with self.subTest(
                            f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_ids}"):
                        res = tensor_search.search(
                            config=self.config, index_name=index.name, text="", result_count=5,
                            filter=filter_string, verbose=0
                        )

                        self.assertEqual(expected_hits, len(res["hits"]))
                        if expected_ids:
                            self.assertEqual(set(expected_ids), {hit["_id"] for hit in res["hits"]})

    def test_filter_unstructured_index_in_keyword_fails(self):
        test_cases = [
            "text_field_1 in (random1, true)",
            "int_field_1 in (100, 200)",
            "long_field_1 in (299, 300)",
            "text_field_1 in (random1, true) AND int_field_1:100",
            "text_field_1 in (random1, true) OR text_field_2:baaadd",
            "text_field_1 in (random1, true) OR int_field_1:[90 TO 210]",
            "text_field_1 in (random1)",
            "NOT text_field_1 in (random1, true)",
            "text_field_1 IN (random1, true) AND int_field_1 in (100, 200)",
            "text_field_1 IN ()"
        ]

        for case in test_cases:
            with self.subTest(case=case):
                with self.assertRaises(base_exceptions.InvalidArgumentError) as cm:
                    tensor_search.search(config=self.config, index_name=self.unstructured_default_text_index.name,
                                         text="", filter=case)

                self.assertIn("'IN' filter keyword is not yet supported for unstructured", str(cm.exception))

    def test_filter_id(self):
        """
        Test filtering by _id
        """
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field_1": "some text"},
                            {"_id": "doc1", "text_field_1": "some text"},
                            {"_id": "doc5", "text_field_1": "some text"},
                            {"_id": "50", "text_field_1": "some text"},
                        ],
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                test_cases = [
                    ("_id:1", 1, ["1"]),
                    ("_id:doc1", 1, ["doc1"]),
                    ("_id:51", 0, None),
                    ("_id:1 OR _id:doc1", 2, ["1", "doc1"]),  # or condition
                    ("_id:1 OR _id:doc1 OR _id:50", 3, ["1", "doc1", "50"]),  # or condition, longer
                    ("_id:1 OR _id:doc1 OR _id:50 OR _id:51", 3, ["1", "doc1", "50"]),
                    # or condition with non-existent id
                    ("_id:1 AND _id:doc1", 0, None),  # and condition
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        ("_id in (1)", 1, ["1"]),
                        ("_id in (doc1, (random garbage id))", 1, ["doc1"]),
                        ("_id in (51)", 0, None),  # non-existent doc
                        ("_id in (1, doc1)", 2, ["1", "doc1"]),
                        ("_id in (1, doc1, 50)", 3, ["1", "doc1", "50"]),
                        ("_id in (1, doc1, 50, (random id))", 3, ["1", "doc1", "50"]),
                        ("_id in (1, doc1) OR _id:doc5", 3, ["1", "doc1", "doc5"]),  # combine in and equality
                        ("_id in (1) AND _id in (doc1)", 0, None),  # and condition
                    ]

                for filter_string, expected_hits, expected_ids in test_cases:
                    with self.subTest(f"filter_string={filter_string}, expected_hits={expected_hits}"):
                        res = tensor_search.search(
                            config=self.config, index_name=index.name, text="some text", filter=filter_string,
                        )

                        self.assertEqual(expected_hits, len(res["hits"]))
                        if expected_ids:
                            self.assertEqual(set(expected_ids), {hit["_id"] for hit in res["hits"]})

    def test_filter_spaced_fields(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1233", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "bool_field_1": True},
                            {"_id": "344", "text_field_1": "some text", "float_field_1": 0.548, "bool_field_1": True},
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Define test parameters as tuples (filter_string, expected_hits, expected_ids)
                test_cases = [
                    ("text_field_2:baaadd", 1, ["5678"]),
                    ("text_field_2:(Close match hehehe)", 2, ["1234", "1233"]),
                    ("(float_field_1:[0 TO 1]) AND (text_field_1:(some text))", 1, ["344"])
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    test_cases += [
                        ("text_field_2 in ((Close match hehehe), (something else))", 2, ["1234", "1233"]),
                        ("(float_field_1:[0 TO 1]) AND (text_field_1 in ((some text)))", 1, ["344"])
                    ]

                for filter_string, expected_hits, expected_ids in test_cases:
                    with self.subTest(f"filter_string={filter_string}, expected_hits={expected_hits}"):
                        res = tensor_search.search(
                            config=self.config, index_name=index.name, text='',
                            filter=filter_string, verbose=0
                        )

                        self.assertEqual(expected_hits, len(res["hits"]))
                        for expected_id in expected_ids:
                            self.assertIn(expected_id, [hit['_id'] for hit in res['hits']])

    def test_filtering_bad_syntax(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index):
                # Adding documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1233", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "bool_field_1": True}
                        ],
                        tensor_fields=["text_field_1", "text_field_2"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Define test parameters as tuples (filter_string)
                bad_filter_strings = [
                    "(text_field_2):baaadd",  # Incorrect syntax for field name with space
                    "(int_field_1:[0 TO 30] and int_field_1:2) AND text_field_1:(some text)",  # and instead of AND here
                    "",  # Empty filter string
                ]

                # Only test IN functionality for structured indexes
                if isinstance(index, StructuredMarqoIndex):
                    bad_filter_strings += [
                        "text_field_2 IN (1, 2 OR 3)",  # OR in IN term
                        "text_field_2 IN (1, 2 AND 3)",  # AND in IN term
                        "text_field_2 IN (1, 2 NOT 3)",  # NOT in IN term
                        "text_field_2 IN (1, 2, 3))",  # extra parenthesis in IN term
                        "text_field_2 IN (val1, val 2, val3)",  # ungrouped space in IN term
                        "text_field_2 IN 1, 2, 3)"  # IN term with no opening parenthesis
                    ]

                for filter_string in bad_filter_strings:
                    with self.subTest(f"filter_string={filter_string}"):
                        with self.assertRaises(core_exceptions.FilterStringParsingError):
                            tensor_search.search(
                                config=self.config, index_name=index.name, text="some text",
                                result_count=3, filter=filter_string, verbose=0
                            )

    def test_filtering_in_with_wrong_type(self):
        # TODO: add unstructured when it in is supported
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Adding documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd", "text_field_3": "b"},
                            {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "int_field_1": 2},
                            {"_id": "1233", "text_field_1": "some text", "text_field_2": "Close match hehehe",
                             "bool_field_1": True, "float_field_1": 1.2, "double_field_1": 2.4}
                        ],
                        tensor_fields=["text_field_1", "text_field_2"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Define test parameters as tuples (filter_string)
                bad_filter_strings = [
                    ("int_field_1 IN (1,2,not_int)", "'not_int', which is not of type 'int'"),
                    ("float_field_1 IN (1.2, 1.3, 2.4)", "unsupported type: 'float'"),
                    ("double_field_1 IN (1.2, 1.3, 2.4)", "unsupported type: 'double'"),
                    ("bool_field_1 IN (true)", "unsupported type: 'bool'")
                ]

                for filter_string, error_message in bad_filter_strings:
                    with self.subTest(f"filter_string={filter_string}"):
                        with self.assertRaises(core_exceptions.InvalidDataTypeError) as cm:
                            tensor_search.search(
                                config=self.config, index_name=index.name, text="some text",
                                result_count=3, filter=filter_string, verbose=0
                            )

                        self.assertIn(error_message, str(cm.exception))

    def test_search_vectoriseIsCalledWithEnableCacheTrue(self):
        """Ensure vectorise is called with enable_cache=True when calling search."""
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            dummy_return = [[1.0, ] * 384, ]
            with (self.subTest(index=index)):
                with mock.patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return) as \
                        mock_vectorise:
                    tensor_search.search(text="some text", index_name=index.name, config=self.config)
                    mock_vectorise.assert_called_once()
                    args, kwargs = mock_vectorise.call_args
                    self.assertTrue(kwargs["enable_cache"])
                mock_vectorise.reset_mock()

    def test_empty_lexical_query(self):
        """
        Test that no documents are returned for an empty lexical query.
        Expected behavior:
        - No documents are returned for an empty query
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text": "document_1"},
                            {"_id": "2", "text": "document_2"},
                            {"_id": "3", "text": "document_3"},
                            {"_id": "4", "text": "document_4", "my_list": ["tag1", "tag2 some"]},
                        ],
                        tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None

                    )
                )

                # Assert that no documents are returned for an empty query
                res = tensor_search.search(text="", config=self.config, index_name=index.name,
                                           search_method=SearchMethod.LEXICAL, result_count=10)
                self.assertIn("hits", res)
                self.assertEqual(0, len(res['hits']))

    def test_wildcard_lexical_query(self):
        """
        Test that the wildcard '*' lexical query works for both structured and unstructured indexes.
        Expected behavior:
        - All documents are returned for a '*' query or with filter applied if applicable
        - Other wildcards are interpreted literally and not supported
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field_1": "document_1"},
                            {"_id": "2", "text_field_1": "document_2"},
                            {"_id": "3", "text_field_1": "document_3"},
                            {"_id": "4", "text_field_1": "document_4",
                             "list_field_1": ["tag1", "tag2 some"]},
                        ],
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None

                    )
                )

                # Assert that all documents are returned for a wildcard query
                res = tensor_search.search(text="*", config=self.config, index_name=index.name,
                                           search_method=SearchMethod.LEXICAL, result_count=10)

                self.assertIn("hits", res)
                self.assertEqual(4, len(res['hits']))

                # Subtests for variations
                variations = [
                    ("*", 4, None),
                    ("*", 1, "list_field_1:tag1"),
                    ('"*"', 0, None),
                    ('"exact" *', 0, None),
                    ('"*" optional', 0, None)
                ]

                for query, expected_count, filter_term in variations:
                    with self.subTest(query=query, filter=filter_term):
                        res = tensor_search.search(text=query, config=self.config,
                                                   index_name=index.name,
                                                   search_method=SearchMethod.LEXICAL, result_count=10,
                                                   filter=filter_term)
                        print(res)
                        self.assertIn("hits", res)
                        self.assertEqual(expected_count, len(res['hits']))

    def test_LexicalSearchResultsScore(self):
        """A test to ensure that the score is returned for lexical search results and the scores are greater than 0."""
        docs = [
            {"_id": "11", "text_field_1": "field_1_document_1"},
            {"_id": "12", "text_field_1": "field_1_document_2"},
            {"_id": "21", "text_field_2": "field_2_document_1"},
            {"_id": "22", "text_field_2": "field_2_document_2"}
        ]

        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(msg=index.type):
                tensor_fields = ["text_field_1", "text_field_2"] \
                    if isinstance(index, UnstructuredMarqoIndex) else None
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )

                # Test search with text_field_1
                res = tensor_search.search(
                    text="field_1_document_1", config=self.config, index_name=index.name,
                    search_method=SearchMethod.LEXICAL, result_count=10
                )

                self.assertEqual(1, len(res["hits"]))
                self.assertEqual("11", res["hits"][0]["_id"])
                self.assertTrue(0 < res["hits"][0]["_score"], f"score: {res['hits'][0]['_score']}")

                # Test search with text_field_2
                res = tensor_search.search(
                    text="field_2_document_1", config=self.config, index_name=index.name,
                    search_method=SearchMethod.LEXICAL, result_count=10
                )

                self.assertEqual(1, len(res["hits"]))
                self.assertEqual("21", res["hits"][0]["_id"])
                self.assertTrue(0 < res["hits"][0]["_score"], f"score: {res['hits'][0]['_score']}")

    def test_get_lexical_search_term(self):
        # Create Vespa indexes
        structured_vespa_index = StructuredVespaIndex(self.structured_default_text_index)
        unstructured_vespa_index = UnstructuredVespaIndex(self.unstructured_default_text_index)

        # List of (VespaIndex, method_name) tuples to test
        indexes_to_test = [
            (structured_vespa_index, '_get_lexical_search_term'),
            (unstructured_vespa_index, '_to_vespa_lexical_query')
        ]

        for index, method_name in indexes_to_test:
            with self.subTest(index_type=type(index).__name__):
                # Mock the _get_lexical_contains_term method for StructuredVespaIndex
                if isinstance(index, StructuredVespaIndex):
                    index._get_lexical_contains_term = mock.MagicMock(
                        side_effect=lambda phrase, query: f'contains("{phrase}")')

                # Test cases
                test_cases = [
                    # Test with score modifiers (should use OR)
                    (
                        MarqoLexicalQuery(
                            index_name="test_index",
                            limit=10,
                            or_phrases=["term1", "term2"],
                            and_phrases=[],
                            score_modifiers=[ScoreModifier(field="field1", weight=1.0, type=ScoreModifierType.Multiply)]
                        ),
                        'contains("term1") OR contains("term2")' if isinstance(index, StructuredVespaIndex)
                        else '(default contains "term1" OR default contains "term2")'
                    ),
                    # Test without score modifiers (should use weakAnd)
                    (
                        MarqoLexicalQuery(
                            index_name="test_index",
                            limit=10,
                            or_phrases=["term1", "term2"],
                            and_phrases=[]
                        ),
                        'weakAnd(contains("term1"), contains("term2"))' if isinstance(index, StructuredVespaIndex)
                        else '(weakAnd(default contains "term1", default contains "term2"))'
                    ),
                    # Test with both OR and AND phrases
                    (
                        MarqoLexicalQuery(
                            index_name="test_index",
                            limit=10,
                            or_phrases=["term1", "term2"],
                            and_phrases=["term3", "term4"]
                        ),
                        '(weakAnd(contains("term1"), contains("term2"))) AND (contains("term3") AND contains("term4"))'
                        if isinstance(index, StructuredVespaIndex)
                        else '((weakAnd(default contains "term1", default contains "term2")) '
                             'AND (default contains "term3" AND default contains "term4"))'
                    ),
                ]

                for query, expected_result in test_cases:
                    if isinstance(index, StructuredVespaIndex):
                        result = getattr(index, method_name)(query)
                    else:
                        # For UnstructuredVespaIndex, we need to extract the search term from the full query
                        full_query = getattr(index, method_name)(query)
                        result = full_query['yql'].split('where')[1].strip()

                    self.assertEqual(result, expected_result)

    def test_search_query_ExpectedErrorRaisedForInvalidSearchMethod(self):
        """Test that the ValidationError is raised when an incorrect search method is provided."""
        invalid_search_methods = [
            ("", "Empty string"),
            (1, "Integer"),
            ([], "List"),
            ({"searchMethod": "LEXICAL"}, "Dictionary"),
        ]
        for search_method, search_method_type in invalid_search_methods:
            with self.subTest(search_method_type=search_method_type):
                with self.assertRaises(ValidationError) as cm:
                    _ = SearchQuery(q="test", search_method=search_method)
                self.assertIn("search_method", str(cm.exception))

    def test_search_query_CanAcceptDifferentSearchMethods(self):
        """Test that the SearchQuery can accept different search methods."""
        valid_search_methods = [
            ("lexical", SearchMethod.LEXICAL, "lowercase lexical"),
            ("teNsor", SearchMethod.TENSOR, "mixed case tensor"),
            ("hybrid", SearchMethod.HYBRID, "mixed case hybrid"),
            (None, SearchMethod.TENSOR, "None"),
        ]
        for search_method, expected_search_method, search_method_type in valid_search_methods:
            with self.subTest(search_method_type=search_method_type):
                search_query = SearchQuery(q="test", searchMethod=search_method)
                self.assertEqual(expected_search_method, search_query.searchMethod)

        # A special case for no search method provided
        search_query = SearchQuery(q="test")
        self.assertEqual(SearchMethod.TENSOR, search_query.searchMethod)