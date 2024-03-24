import math
import os
import random
import uuid
from unittest import mock

import requests

import marqo.core.exceptions as core_exceptions
from marqo.api import exceptions as errors
from marqo.api.exceptions import IndexNotFoundError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference.s2_inference import get_model_properties_from_registry
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestSearchStructured(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.structured_marqo_index_request(
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
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3",
                           "text_field_4", "text_field_5", "text_field_6"]
        )
        default_text_index_encoded_name = cls.structured_marqo_index_request(
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

        default_image_index = cls.structured_marqo_index_request(
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

        image_index_with_random_model = cls.structured_marqo_index_request(
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

        cls.indexes = cls.create_indexes([
            default_text_index,
            default_text_index_encoded_name,
            default_image_index,
            image_index_with_random_model
        ])

        cls.default_text_index = default_text_index.name
        cls.default_text_index_encoded_name = default_text_index_encoded_name.name
        cls.default_image_index = default_image_index.name
        cls.image_index_with_random_model = image_index_with_random_model.name

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    # TODO - Test efSearch parameter
    # TODO - Test approximate parameter
    # TODO - Test graceful degradation detection with approximate=False
    # TODO - Test timeout parameter
    def test_each_doc_returned_once(self):
        """Each doc should be returned once, even if it matches multiple times"""
        tests = [
            (self.default_text_index, 'Standard index name'),
            (self.default_text_index_encoded_name, 'Index name requiring encoding'),
        ]
        for index_name, desc in tests:
            with self.subTest(desc):
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=[
                            {"text_field_1": "Exact match hehehe efgh ",
                             "text_field_2": "baaadd efgh ",
                             "text_field_3": "some field efgh ",
                             "_id": "5678"},
                            {"text_field_1": "shouldn't really match ",
                             "text_field_2": "Nope.....",
                             "text_field_3": "Random text here efgh ",
                             "_id": "1234"},
                        ]
                    )
                )

                search_res = tensor_search._vector_text_search(
                    config=self.config, index_name=index_name,
                    query=" efgh ", result_count=10, device="cpu"
                )
                assert len(search_res['hits']) == 2

    #
    # def test_search_with_searchable_attributes_max_attributes_is_none(self):
    #     # No patch needed, MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES is not set
    #     add_docs_caller(
    #         config=self.config, index_name=self.default_text_index, docs=[
    #             {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "5678"},
    #             {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
    #         ], )
    #     tensor_search.search(
    #         config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
    #         searchable_attributes=["other field"]
    #     )
    #
    # @mock.patch.dict(os.environ, {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': f"{sys.maxsize}"}})
    # def test_search_with_no_searchable_attributes_but_max_searchable_attributes_env_set(self):
    #     with self.assertRaises(InvalidArgError):
    #         add_docs_caller(
    #             config=self.config, index_name=self.default_text_index, docs=[
    #                 {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "5678"},
    #                 {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
    #             ], )
    #         tensor_search.search(
    #             config=self.config, index_name=self.default_text_index, text="Exact match hehehe"
    #         )
    #

    def test_vector_text_search_no_device(self):
        try:
            search_res = tensor_search._vector_text_search(
                config=self.config, index_name=self.default_text_index,
                result_count=5, query="some text...")
            raise AssertionError
        except errors.InternalError:
            pass

    def test_vector_search_against_empty_index(self):
        search_res = tensor_search._vector_text_search(
            config=self.config, index_name=self.default_text_index,
            result_count=5, query="some text...", device="cpu")
        assert {'hits': []} == search_res

    def test_vector_search_against_non_existent_index(self):
        try:
            tensor_search._vector_text_search(
                config=self.config, index_name="some-non-existent-index",
                result_count=5, query="some text...", device="cpu")
        except IndexNotFoundError as s:
            pass

    def test_vector_search_long_query_string(self):
        query_text = """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
                    The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
                    """

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"_id": "12345",
                     "text_field_1": "The Guardian is a newspaper, read in the UK and other places around the world", },
                    {"_id": "abc12334",
                     "text_field_1": "Grandma Jo's family recipe.",
                     "text_field_2": "1. Cook meat. 2: Dice Onions. 3: Serve."}
                ]
            )
        )

        res = tensor_search._vector_text_search(
            config=self.config, index_name=self.default_text_index, query=query_text, device="cpu"
        )

        assert len(res["hits"]) == 2

    def test_search_edge_case(self):
        """We ran into bugs with this doc"""
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "1",
                        "text_field_1": "In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium. They include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements.",
                        "text_field_2": "In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium. They include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements."
                    },
                    {
                        "_id": "2",
                        "text_field_1": "defgh",
                        "text_field_2": "on the mat"
                    }
                ]
            )
        )

        res = tensor_search.search(
            text="In addition to NiS collection fire assay for a five element",
            config=self.config, index_name=self.default_text_index
        )

        self.assertEqual(2, len(res["hits"]))

    def test_search_format(self):
        """Is the result formatted correctly?"""
        q = "Exact match hehehe"

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "5678",
                        "text_field_1": "Exact match hehehe",
                        "text_field_2": "baaadd",
                        "text_field_3": "res res res"
                    },
                    {
                        "_id": "1234",
                        "text_field_1": "random text",
                        "text_field_2": "Close match hehehe"
                    },
                    {
                        "_id": "9000",
                        "text_field_1": "somewhat match"
                    }
                ]
            )
        )

        search_res = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text=q, result_count=50
        )

        assert "processingTimeMs" in search_res
        assert search_res["processingTimeMs"] > 0
        assert isinstance(search_res["processingTimeMs"], int)

        assert "query" in search_res
        assert search_res["query"] == q

        assert "limit" in search_res
        assert search_res["limit"] == 50

    def test_search_format_empty(self):
        """Is the result formatted correctly? - on an emtpy index?"""
        search_res = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text=""
        )
        assert "processingTimeMs" in search_res
        assert search_res["processingTimeMs"] > 0
        assert isinstance(search_res["processingTimeMs"], int)

        assert "query" in search_res
        assert search_res["query"] == ""

        assert "limit" in search_res
        assert search_res["limit"] > 0

    def test_result_count_validation(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "5678",
                        "text_field_1": "Exact match hehehe",  # previously "abc"
                        "text_field_2": "baaadd",  # previously "other field"
                        "text_field_3": "res res res"  # previously "Cool Field 1"
                    },
                    {
                        "_id": "1234",
                        "text_field_1": "random text",  # previously "abc"
                        "text_field_2": "Close match hehehe"  # previously "other field"
                    },
                    {
                        "_id": "9000",
                        "text_field_1": "somewhat match"  # previously "Cool Field 1"
                    }
                ]
            )
        )

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # too big
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=-1
            )

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # violates total results
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=1000000
            )
            raise AssertionError

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # violates max limit of 1000
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=1001
            )
            raise AssertionError

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # violates max offset of 10,000
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=1,
                offset=10001
            )
            raise AssertionError

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # should not work with 0
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=0
            )
            raise AssertionError

        # should work with 1:
        search_res = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
            result_count=1
        )
        assert len(search_res['hits']) >= 1

    def test_highlights_tensor(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd"},
                    {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe"}
                ]
            )
        )

        tensor_highlights = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text="some text", highlights=True)
        self.assertEqual(2, len(tensor_highlights["hits"]))

        for hit in tensor_highlights["hits"]:
            assert "_highlights" in hit

        tensor_no_highlights = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text="some text", highlights=False)
        self.assertEqual(2, len(tensor_highlights["hits"]))
        for hit in tensor_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_highlights_lexical(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"_id": "5678", "text_field_1": "some text", "text_field_2": "baaadd"},
                    {"_id": "1234", "text_field_1": "some text", "text_field_2": "Close match hehehe"}
                ]
            )
        )
        lexical_highlights = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text="some text",
            search_method=SearchMethod.LEXICAL, highlights=True)
        assert len(lexical_highlights["hits"]) == 2
        for hit in lexical_highlights["hits"]:
            assert "_highlights" in hit

        lexical_no_highlights = tensor_search.search(
            config=self.config, index_name=self.default_text_index, text="some text",
            search_method=SearchMethod.LEXICAL, highlights=False)
        assert len(lexical_no_highlights["hits"]) == 2
        for hit in lexical_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_search_int_field(self):
        """doesn't error out if there is a random int field"""
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "5678",
                        "text_field_1": "some text",
                        "text_field_2": "baaadd",
                        "int_field_1": 144
                    },
                    {
                        "_id": "1234",
                        "text_field_1": "some text",
                        "text_field_2": "Close match hehehe",
                        "int_field_1": 88
                    }
                ]
            )
        )
        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            with self.subTest(f"search_method={search_method}"):
                s_res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="cool match",
                    search_method=search_method)
                assert len(s_res["hits"]) > 0

    # TODO: All filtering tests have been moved to test_search_combined.py
    # Do the same with all other tests.

    def test_set_device(self):
        """calling search with a specified device overrides MARQO_BEST_AVAILABLE_DEVICE"""

        mock_vectorise = mock.MagicMock()

        # Get vector dimension of the default BERT model
        DEFAULT_MODEL_DIMENSION = get_model_properties_from_registry("hf/all_datasets_v4_MiniLM-L6")["dimensions"]
        mock_vectorise.return_value = [[0, ] * DEFAULT_MODEL_DIMENSION]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="some text",
                search_method=SearchMethod.TENSOR, highlights=True, device="cuda:123")
            return True

        assert run()
        assert os.environ["MARQO_BEST_AVAILABLE_DEVICE"] == "cpu"
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:123"

    def test_search_other_types_subsearch(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[{
                    "int_field_1": 1,
                    "float_field_1": 1.2,
                    "bool_field_1": True,
                    "text_field_1": "blah"
                }]
                # Note: The tensor_fields parameter is kept as the original field name is maintained
            )
        )
        for to_search in [1, 1.2, True, "blah"]:
            assert "hits" in tensor_search._lexical_search(
                text=str(to_search), config=self.config, index_name=self.default_text_index,
            )
            assert "hits" in tensor_search._vector_text_search(
                query=str(to_search), config=self.config, index_name=self.default_text_index, device="cpu"
            )

    def test_search_other_types_top_search(self):
        docs = [{
            "int_field_1": 1,
            "float_field_1": 1.2,
            "bool_field_1": True,
            "text_field_1": "blah"
        }]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                # Note: Keeping tensor_fields as "some_str" since it corresponds to "text_field_1"
            )
        )

        for field, to_search in docs[0].items():
            assert "hits" in tensor_search.search(
                text=str(to_search), config=self.config, index_name=self.default_text_index,
                search_method=SearchMethod.TENSOR, filter=f"{field}:{to_search}"
            )

            assert "hits" in tensor_search.search(
                text=str(to_search), config=self.config, index_name=self.default_text_index,
                search_method=SearchMethod.LEXICAL, filter=f"{field}:{to_search}"
            )

    def test_lexical_filtering(self):
        # Adding documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "123456",
                        "text_field_1": "The captain bravely lead her followers into battle. She directed her soldiers to and fro.",
                        "text_field_2": "some text",
                        "text_field_3": "other things",
                        "bool_field_1": True,
                        "float_field_1": 0.61
                    },
                    {
                        "_id": "other doc",
                        "float_field_1": 0.66,
                        "text_field_4": "some text too",
                        "int_field_1": 5,
                        "text_field_5": "234",  # Assuming 'fake_int' is actually a string
                        "text_field_6": "1.23",  # Assuming 'fake_float' is actually a string
                        "text_field_7": "gap"  # Assuming 'gapped field_name' is a valid field
                    }
                ]
            )
        )

        # Define test parameters as tuples (filter_string, expected_hits, expected_id)
        test_parameters = [
            ("(bool_field_1:true AND float_field_1:[0.1 TO 0.75]) AND text_field_3:(other things)", 1, "123456"),
            ("text_field_8:1", 0, None),  # Assuming 'my_looLoo' is renamed to 'text_field_8'
            ("int_field_1:5", 1, "other doc"),
            ("int_field_1:[1 TO 10]", 1, "other doc"),
            ("float_field_1:0.61", 1, "123456"),
            ("text_field_3:(other things)", 1, "123456"),
            ("text_field_5:234", 1, "other doc"),
            ("text_field_6:1.23", 1, "other doc"),
            ("text_field_7:gap", 1, "other doc")
        ]

        for filter_string, expected_hits, expected_id in test_parameters:
            with self.subTest(f"filter_string={filter_string}, expected_hits={expected_hits}"):
                res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="some text",
                    result_count=3, filter=filter_string, search_method=SearchMethod.LEXICAL
                )

                if expected_hits == math.inf:
                    self.assertTrue(len(res["hits"]) >= 1)
                else:
                    self.assertEqual(expected_hits, len(res["hits"]))
                    if expected_id:
                        self.assertEqual(expected_id, res["hits"][0]["_id"])

    # def test_filter_on_id_and_more(self):
    #     """Test various filtering scenarios including _id and other conditions"""
    #     # Adding documents
    #     tensor_search.add_documents(
    #         config=self.config,
    #         add_docs_params=AddDocsParams(
    #             index_name=self.default_text_index,
    #             docs=[
    #                 {"abc": "some text", "other field": "baaadd", "_id": "5678", "status": "active"},
    #                 {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "status": "inactive"},
    #                 {"abc": "different text", "other field": "irrelevant", "_id": "9012", "status": "active"}
    #             ],
    #             tensor_fields=["abc", "other field"]
    #         )
    #     )
    #
    #     test_cases = [
    #         ("filter on id 5678", "_id:5678", 1, ["5678"]),
    #         ("filter on id 1234", "_id:1234", 1, ["1234"]),
    #         ("AND filter", "_id:5678 AND status:active", 1, ["5678"]),
    #         ("OR filter", "_id:5678 OR _id:1234", 2, ["5678", "1234"]),
    #         ("Complex filter", "_id:5678 OR (abc:some\ text AND status:inactive)", 2, ["5678", "1234"]),
    #         ("Non-ID field filter", "status:active", 2, ["5678", "9012"]),
    #         ("No result filter", "_id:0000", 0, [])
    #     ]
    #
    #     for name, filter_query, expected_count, expected_ids in test_cases:
    #         with self.subTest(name=name):
    #             res = tensor_search.search(
    #                 config=self.config, index_name=self.default_text_index, text="some text",
    #                 filter=filter_query
    #             )
    #             self.assertEqual(expected_count, len(res["hits"]))
    #             if expected_ids:
    #                 self.assertEqual(expected_ids, [hit["_id"] for hit in res["hits"]])

    def test_attributes_to_retrieve(self):
        docs = [
            {
                "text_field_1": "Exact match hehehe",
                "text_field_2": "baaadd",
                "text_field_3": "res res res",
                "text_field_4": "res res res haha",
                "text_field_5": "check check haha",
            }
        ]

        test_inputs = (
            ([], {"_id", "_score", "_highlights"}),
            (["text_field_1"], {"text_field_1", "_id", "_score", "_highlights"}),
            (["text_field_1", "text_field_2"], {"text_field_1", "text_field_2", "_id", "_score", "_highlights"}),
            (["text_field_1", "text_field_3"], {"text_field_1", "text_field_3", "_id", "_score", "_highlights"}),
            (["text_field_1", "text_field_3", "text_field_4"],
             {"text_field_1", "text_field_3", "text_field_4", "_id", "_score", "_highlights"}),
            (["text_field_1", "text_field_3", "text_field_4", "text_field_5"],
             {"text_field_1", "text_field_3", "text_field_4", "text_field_5", "_id", "_score", "_highlights"}),
            (["text_field_1", "text_field_2", "text_field_3", "text_field_4", "text_field_5"],
             {"text_field_1", "text_field_2", "text_field_3", "text_field_4", "text_field_5", "_id", "_score",
              "_highlights"}),
            # TODO Fix this subtest
            # Not running this test case until we solve the bool issue
            # (None, {"text_field_1", "text_field_2", "text_field_3", "text_field_4", "text_field_5", "_id", "_score",
            #         "_highlights"}),
        )

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
            )
        )

        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            for attributes_to_retrieve, expected_fields in test_inputs:
                with self.subTest(
                        f"search_method = {search_method}, attributes_to_retrieve={attributes_to_retrieve},"
                        f" expected_fields = {expected_fields}"):
                    res = tensor_search.search(
                        config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                        attributes_to_retrieve=attributes_to_retrieve, search_method=search_method
                    )

                    self.assertEqual(expected_fields, set(res["hits"][0].keys()))

    def test_limit_results(self):
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
        vocab = requests.get(vocab_source).text.splitlines()

        res = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.image_index_with_random_model,
                docs=[{"text_field_1": "a test of" + (" ".join(random.choices(population=vocab, k=10)))}
                      for _ in range(128)],
                # Assuming 'Title' is now 'text_field_1'
            )
        )

        search_text = "a test of"

        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            for max_doc in [2, 5, 10, 100]:
                with self.subTest(f"search_method={search_method}, max_doc={max_doc}"):
                    mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}

                    @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
                    def run():
                        res = half_search = tensor_search.search(
                            search_method=search_method,
                            config=self.config,
                            index_name=self.image_index_with_random_model,
                            text=search_text,
                            result_count=max_doc // 2
                        )

                        assert half_search['limit'] == max_doc // 2
                        assert len(half_search['hits']) == max_doc // 2

                        limit_search = tensor_search.search(
                            search_method=search_method,
                            config=self.config,
                            index_name=self.image_index_with_random_model,
                            text=search_text,
                            result_count=max_doc
                        )

                        self.assertEqual(max_doc, limit_search['limit'])
                        self.assertEqual(max_doc, len(limit_search['hits']))
                        try:
                            oversized_search = tensor_search.search(
                                search_method=search_method,
                                config=self.config,
                                index_name=self.image_index_with_random_model,
                                text=search_text,
                                result_count=max_doc + 1
                            )
                        except errors.IllegalRequestedDocCount:
                            pass
                        try:
                            very_oversized_search = tensor_search.search(
                                search_method=search_method,
                                config=self.config,
                                index_name=self.image_index_with_random_model,
                                text=search_text,
                                result_count=(max_doc + 1) * 2
                            )
                        except errors.IllegalRequestedDocCount:
                            pass
                        return True

                    assert run()

    def test_invalid_limit_results(self):
        """Ensure that proper errors are raised when the limit is bad"""

        bad_limit = [0, -1, 0.23, 2.56]

        for limit in bad_limit:
            with self.subTest(f"limit={limit}"):
                with self.assertRaises(errors.IllegalRequestedDocCount):
                    tensor_search.search(
                        config=self.config,
                        index_name=self.default_text_index,
                        text="",
                        result_count=limit
                    )

    def test_image_search_highlights(self):
        """Does the URL get returned as the highlight? (it should - because no rerankers are being used)"""
        url_1 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        url_2 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
        docs = [
            {"_id": "123", "image_field_1": url_1, "text_field_1": "irrelevant text"},
            {"_id": "789", "image_field_1": url_2},
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
            )
        )
        res = tensor_search.search(
            config=self.config, index_name=self.default_image_index,
            text="A hippo in the water", result_count=3,
        )
        assert len(res['hits']) == 2
        assert {hit['image_field_1'] for hit in res['hits']} == {url_2, url_1}
        assert {hit['_highlights'][0]['image_field_1'] for hit in res['hits']} == {url_2, url_1}

    def test_multi_search(self):
        docs = [
            {"text_field_1": "Doberman, canines, golden retrievers are humanity's best friends",
             "_id": 'dog_doc'},
            {"text_field_1": "All things poodles! Poodles are great pets",
             "_id": 'poodle_doc'},
            {"text_field_1": "Construction and scaffolding equipment",
             "_id": 'irrelevant_doc'}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
            )
        )
        queries_expected_ordering = [
            ({"Dogs": 2.0, "Poodles": -2}, ['dog_doc', 'irrelevant_doc', 'poodle_doc']),
            ("dogs", ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
            ({"dogs": 1}, ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
            ({"Dogs": -2.0, "Poodles": 2}, ['poodle_doc', 'irrelevant_doc', 'dog_doc']),
        ]

        for query, expected_ordering in queries_expected_ordering:
            with self.subTest(f"query={query}, expected_ordering={expected_ordering}"):
                res = tensor_search.search(
                    text=query,
                    index_name=self.default_text_index,
                    result_count=5,
                    config=self.config,
                    search_method=SearchMethod.TENSOR)

                for hit_position, _ in enumerate(res['hits']):
                    self.assertEqual(expected_ordering[hit_position], res['hits'][hit_position]['_id'])

    def test_multi_search_images(self):
        docs = [
            {
                "_id": 'realistic_hippo',
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
            },
            {
                "_id": 'artefact_hippo',
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
            }
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
            )
        )
        queries_expected_ordering = [
            ({"Nature photography": 2.0, "Artefact": -2}, ['realistic_hippo', 'artefact_hippo']),
            ({"Nature photography": -1.0, "Artefact": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"Nature photography": -1.5, "Artefact": 1.0, "hippo": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": -1.0,
              "blah": 1.0}, ['realistic_hippo', 'artefact_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
              "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0},
             ['artefact_hippo', 'realistic_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
              "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
              "artefact": 1.0, "photo realistic": -1},
             ['artefact_hippo', 'realistic_hippo']),
        ]
        for query, expected_ordering in queries_expected_ordering:
            with self.subTest(f"query={query}, expected_ordering={expected_ordering}"):
                res = tensor_search.search(
                    text=query,
                    index_name=self.default_image_index,
                    result_count=5,
                    config=self.config,
                    search_method=SearchMethod.TENSOR)
                for hit_position, _ in enumerate(res['hits']):
                    self.assertEqual(expected_ordering[hit_position], res['hits'][hit_position]['_id'])

    def test_multi_search_images_invalid_queries(self):
        docs = [
            {
                "_id": 'realistic_hippo',
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
            },
            {
                "_id": 'artefact_hippo',
                "text_field_1": "Some text about a weird forest"
            }
        ]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
            )
        )

        invalid_queries = [{}, None, {123: 123}, {'123': None},
                           {"https://marqo_not_real.com/image_1.png": 3}, set()]
        for q in invalid_queries:
            with self.subTest(f"query={q}"):
                with self.assertRaises(errors.InvalidArgError):
                    tensor_search.search(
                        text=q,
                        index_name=self.default_image_index,
                        result_count=5,
                        config=self.config,
                        search_method=SearchMethod.TENSOR)

    def test_multi_search_images_edge_cases(self):
        docs = [
            {
                "_id": 'realistic_hippo',
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
            },
            {
                "_id": 'artefact_hippo',
                "text_field_1": "Some text about a weird forest"
            }
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
            )
        )

        alright_queries = [{"v ": 1.2}, {"d ": 0}, {"vf": -1}]
        for q in alright_queries:
            with self.subTest(f"query={q}"):
                tensor_search.search(
                    text=q,
                    index_name=self.default_image_index,
                    result_count=5,
                    config=self.config,
                    search_method=SearchMethod.TENSOR)

    def test_multi_search_images_lexical(self):
        """Error if you try this"""
        docs = [
            {"_id": 'realistic_hippo', "image_field_1": "124"},
            {"_id": 'artefact_hippo', "text_field_1": "Some text about a weird forest"}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
            )
        )

        for bad_method in [SearchMethod.LEXICAL, "kjrnkjrn", ""]:
            with self.subTest(f"bad_method={bad_method}"):
                with self.assertRaises(errors.InvalidArgError):
                    tensor_search.search(
                        text={'something': 1},
                        index_name=self.default_text_index,
                        result_count=5,
                        config=self.config,
                        search_method=bad_method)

    def test_image_search(self):
        """This test is to ensure image search works as expected"""
        hippo_image = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        doc_dict = {
            'realistic_hippo': {"image_field_1": hippo_image, "_id": 'realistic_hippo'},
            'artefact_hippo': {"text_field_1": "Some text about a weird forest", "_id": 'artefact_hippo'}
        }

        docs = list(doc_dict.values())

        res = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
            )
        )
        res = tensor_search.search(
            text=hippo_image,
            index_name=self.default_image_index,
            result_count=5,
            config=self.config,
            search_method=SearchMethod.TENSOR)

        assert len(res['hits']) == 2
        for hit in res['hits']:
            original_doc = doc_dict[hit['_id']]
            assert len(hit['_highlights']) == 1
            highlight_field = list(hit['_highlights'][0].keys())[0]
            assert highlight_field in original_doc
            assert hit[highlight_field] == original_doc[highlight_field]

    def test_lexical_search_no_highlights_format(self):
        docs = [
            {"_id": "1", "text_field_1": "some text", "text_field_2": "Close match hehehe", "int_field_1": 1},
            {"_id": "2", "text_field_1": "some code", "text_field_2": "match", "int_field_1": 2},

        ]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
            )
        )

        lexical_search_result = tensor_search.search(
            text="some text",
            index_name=self.default_text_index,
            result_count=5,
            config=self.config,
            search_method=SearchMethod.LEXICAL,
        )
        self.assertEqual(2, len(lexical_search_result['hits']))
        for hit in lexical_search_result['hits']:
            self.assertIn("_highlights", hit)
            self.assertTrue(isinstance(hit["_highlights"], list))
            self.assertEqual(0, len(hit["_highlights"]))

    def test_tensor_search_highlights_format(self):
        docs = [
            {"_id": "1", "text_field_1": "some text", "text_field_2": "Close match hehehe", "int_field_1": 1},
            {"_id": "2", "text_field_1": "some code", "text_field_2": "match", "int_field_1": 2},

        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
            )
        )
        tensor_search_result = tensor_search.search(
            text="some text",
            index_name=self.default_text_index,
            config=self.config,
            search_method=SearchMethod.TENSOR
        )
        self.assertEqual(2, len(tensor_search_result['hits']))
        for hit in tensor_search_result['hits']:
            self.assertIn("_highlights", hit)
            self.assertTrue(isinstance(hit["_highlights"], list))
            self.assertEqual(1, len(hit["_highlights"]))  # We only have 1 highlight now
            self.assertTrue(isinstance(hit["_highlights"][0], dict))

    def test_filter_on_large_integer_and_float(self):
        valid_documents = [
            {'long_field_1': 1, '_id': '0', "text_field_1": "some text"},  # small positive integer
            {'long_field_1': -1, '_id': '1', "text_field_1": "some text"},  # small negative integer
            # large positive integer that can't be handled by int
            {'long_field_1': 100232142864, '_id': '2', "text_field_1": "some text"},
            # large negative integer that can't be handled by int
            {'long_field_1': -923217213, '_id': '3', "text_field_1": "some text"},
            # large positive integer mathematical expression
            {'double_field_1': 10000000000.0, '_id': '4', "text_field_1": "some text"},
            # large negative integer mathematical expression
            {'double_field_1': -1000000000000.0, '_id': '5', "text_field_1": "some text"},
            # large positive float
            {'double_field_1': 10000000000.12325, '_id': '6', "text_field_1": "some text"},
            # large negative float
            {'double_field_1': -9999999999.87675, '_id': '7', "text_field_1": "some text"}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=valid_documents,
            )
        )

        self.assertEqual(len(valid_documents),
                         self.monitoring.get_index_stats_by_name(self.default_text_index).number_of_documents)

        for document in valid_documents:
            for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
                numeric_field = list(document.keys())[0]
                numeric_value = document[numeric_field]
                filter_string = f"{numeric_field}:{numeric_value}"
                expected_document_ids = document["_id"]
                with self.subTest(f"filter_string = {filter_string}, "
                                  f"expected_document_ids = {expected_document_ids}, "
                                  f"search_method = {search_method}"):
                    res = tensor_search.search(
                        config=self.config, index_name=self.default_text_index, text="some text",
                        filter=filter_string, search_method=SearchMethod.LEXICAL
                    )
                    self.assertEqual(1, len(res["hits"]))
                    self.assertEqual(expected_document_ids, res["hits"][0]["_id"])

