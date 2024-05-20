import copy
import math
import os
import random
import uuid
from unittest import mock

import requests
from pydantic import ValidationError

import marqo.core.exceptions as core_exceptions
from marqo.api import exceptions as errors
from marqo.api.exceptions import IndexNotFoundError
from marqo.api.exceptions import InvalidArgError
from marqo.core.models.marqo_index import *
from marqo.s2_inference.s2_inference import get_model_properties_from_registry
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.search import SearchContext
from tests.marqo_test import MarqoTestCase


class TestSearchUnstructured(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )
        default_text_index_encoded_name = cls.unstructured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', '')
        )

        default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_random_model = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
            treat_urls_and_pointers_as_images=True
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
            default_text_index_encoded_name,
            default_image_index,
            image_index_with_chunking,
            image_index_with_random_model
        ])

        cls.default_text_index = default_text_index.name
        cls.default_text_index_encoded_name = default_text_index_encoded_name.name
        cls.default_image_index = default_image_index.name
        cls.image_index_with_chunking = image_index_with_chunking.name
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
                tensor_search.add_documents(config=self.config,
                                            add_docs_params=AddDocsParams(
                                                index_name=index_name,
                                                docs=[
                                                    {"abc": "Exact match hehehe efgh ", "other_field": "baaadd efgh ",
                                                     "_id": "5678", "finally": "some field efgh "},
                                                    {"abc": "shouldn't really match ", "other_field": "Nope.....",
                                                     "_id": "1234", "finally": "Random text here efgh "},
                                                ],
                                                tensor_fields=["abc", "other_field", "finally"],
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

    @staticmethod
    def strip_marqo_fields(doc, strip_id=False):
        """Strips Marqo fields from a returned doc to get the original doc"""
        copied = copy.deepcopy(doc)

        strip_fields = ["_highlights", "_score"]
        if strip_id:
            strip_fields += ["_id"]

        for to_strip in strip_fields:
            try:
                del copied[to_strip]
            except KeyError:
                pass
        return copied

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
                     "Desc": "The Guardian is newspaper, read in the UK and other places around the world"},
                    {"_id": "abc12334", "Title": "Grandma Jo's family recipe. ",
                     "Steps": "1. Cook meat. 2: Dice Onions. 3: Serve."}],
                tensor_fields=["Desc", "Title", "Steps"],
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
                index_name=self.default_text_index, docs=[
                    {
                        'text': 'In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium.hey include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements.',
                        'combined': 'In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium.hey include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements.',
                        "_id": "1"
                    },
                    {
                        "abc": "defgh",
                        "this_cat_sat": "on the mat",
                        "_id": "2"
                    }
                ],
                tensor_fields=["text", "combined", "abc", "this_cat_sat"]
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
                    {"abc": "Exact match hehehe", "other_field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
                    {"abc": "random text", "other_field": "Close match hehehe", "_id": "1234"},
                    {"Cool Field 1": "somewhat match", "_id": "9000"}
                ],
                tensor_fields=["abc", "other_field", "Cool Field 1"]
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

    def test_searchable_attributes_not_supported_in_unstructured_index(self):
        with self.assertRaises(errors.InvalidArgError) as ex:
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="",
                searchable_attributes=["None"], result_count=50
            )
        self.assertIn("searchable_attributes is not supported", str(ex.exception))

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
                    {"abc": "Exact match hehehe", "other_field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
                    {"abc": "random text", "other_field": "Close match hehehe", "_id": "1234"},
                    {"Cool Field 1": "somewhat match", "_id": "9000"}],
                tensor_fields=["abc", "other_field", "Cool Field 1"]
            )
        )

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # too big
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=-1
            )

        with self.assertRaises(errors.IllegalRequestedDocCount):
            # too small
            search_res = tensor_search.search(
                config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                result_count=1000000
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
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234"}],
                tensor_fields=["abc", "other_field"]
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
                index_name=self.default_text_index, docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234"}],
                tensor_fields=[]
            )
        )

        res = lexical_highlights = tensor_search.search(
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
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_int": 144},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "my_int": 88},
                ],
                tensor_fields=["abc", "other_field"]
            )
        )
        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            with self.subTest(f"search_method={search_method}"):
                s_res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="cool match",
                    search_method=search_method)
                assert len(s_res["hits"]) > 0

    def test_filtering_list_case_tensor(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "_id": "1235", "my_list": ["tag1", "tag2 some"]}],
                tensor_fields=["abc", "other_field"]
            )
        )

        res_exists = tensor_search.search(
            index_name=self.default_text_index, config=self.config, text="", filter="my_list:tag1")

        res_not_exists = tensor_search.search(
            index_name=self.default_text_index, config=self.config, text="", filter="my_list:tag55")

        res_other = tensor_search.search(
            index_name=self.default_text_index, config=self.config, text="", filter="my_string:b")

        # strings in lists are converted into keyword, which aren't filterable on a token basis.
        # Because the list member is "tag2 some" we can only exact match (incl. the space).
        # "tag2" by itself doesn't work, only "(tag2 some)"
        res_should_only_match_keyword_bad = tensor_search.search(
            index_name=self.default_text_index, config=self.config, text="", filter="my_list:tag2")
        res_should_only_match_keyword_good = tensor_search.search(
            index_name=self.default_text_index, config=self.config, text="", filter="my_list:(tag2 some)")

        assert res_exists["hits"][0]["_id"] == "1235"
        assert res_exists["hits"][0]["_highlights"][0] == {"abc": "some text"}
        assert len(res_exists["hits"]) == 1

        assert len(res_not_exists["hits"]) == 0

        assert res_other["hits"][0]["_id"] == "5678"
        assert len(res_other["hits"]) == 1

        assert len(res_should_only_match_keyword_bad["hits"]) == 0
        assert len(res_should_only_match_keyword_good["hits"]) == 1

    def test_filtering_list_case_lexical(self):
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "_id": "1235", "my_list": ["tag1", "tag2 some"]}
                ],
                tensor_fields=["abc", "other_field"]
            )
        )

        test_input = [
            ("my_list:tag1", 1, "1235"),
            ("my_list:tag55", 0, None),
            ("my_string:b", 1, "5678"),
        ]

        for filter_string, expected_hits, expected_id in test_input:
            with self.subTest(
                    f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_id}"):
                res = tensor_search.search(
                    index_name=self.default_text_index, config=self.config, text="some",
                    search_method=SearchMethod.LEXICAL, filter=filter_string
                )
                self.assertEqual(expected_hits, len(res["hits"]))
                self.assertEqual(expected_id, res["hits"][0]["_id"] if expected_id else None)

    #
    def test_filtering_list_case_image(self):

        hippo_img = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=[
                    {"img": hippo_img, "abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"img": hippo_img, "abc": "some text", "other_field": "Close match hehehe", "_id": "1234",
                     "an_int": 2},
                    {"img": hippo_img, "abc": "some text", "_id": "1235", "my_list": ["tag1", "tag2 some"]}],
                tensor_fields=["abc", "other_field", "img"]
            )
        )

        test_parameters = [
            ("my_list:tag1", 1, "1235"),
            ("my_list:tag55", 0, None),
            ("my_string:b", 1, "5678"),
        ]

        for filter_string, expected_hits, expected_id in test_parameters:
            with self.subTest(
                    f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_id}"):
                res = tensor_search.search(
                    index_name=self.default_image_index, config=self.config, text="some",
                    search_method=SearchMethod.TENSOR, filter=filter_string
                )

                self.assertEqual(expected_hits, len(res["hits"]))
                self.assertEqual(expected_id, res["hits"][0]["_id"] if expected_id else None)

    def test_filtering(self):
        # TODO-Li Add support for filter on Bool
        # Add documents first (assuming add_docs_caller is a method to add documents)
        res = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1233", "my_bool": True}
                ],
                tensor_fields=["abc", "other_field"]
            )
        )

        # Define test parameters as tuples (filter_string, expected_hits, expected_id)
        test_parameters = [
            ("my_string:c", 0, None),
            ("an_int:2", 1, "1234"),
            ("my_string:b", 1, "5678"),
            ("my_int_something:5", 0, None),
            ("an_int:[5 TO 30]", 0, None),
            ("an_int:[0 TO 30]", 1, "1234"),
            ("my_bool:true", 1, "1233"),
            ("an_int:[0 TO 30] OR my_bool:true", 2, None),  # Multiple hits, so expected_id is None
            ("(an_int:[0 TO 30] AND an_int:2) AND abc:(some text)", 1, "1234"),
        ]

        for filter_string, expected_hits, expected_id in test_parameters:
            with self.subTest(
                    f"filter_string={filter_string}, expected_hits={expected_hits}, expected_id={expected_id}"):
                res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="some text", result_count=3,
                    filter=filter_string, verbose=0
                )

                self.assertEqual(expected_hits, len(res["hits"]))
                if expected_id:
                    self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_filtering_string_boolean_and_real_boolean_fields(self):
        documents = [
            {"_id": "1", "text_field_1": "true", "text_field_2": "false",
             "bool_field_1": True, "bool_field_2": False, "text_field_3": "search me"},
            {"_id": "2", "text_field_1": "false", "text_field_2": "True",
             "bool_field_1": False, "bool_field_2": True, "text_field_3": "search me"},
        ]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=documents,
                tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
            )
        )

        test_cases = [
            ("text_field_1:true", 1, "1"),
            ("text_field_1:false", 1, "2"),
            ("bool_field_1:true", 1, "1"),
            ("bool_field_1:false", 1, "2"),
            ("text_field_2:true", 1, "2"),
            ("text_field_2:false", 1, "1"),
            ("bool_field_2:true", 1, "2"),
            ("bool_field_2:false", 1, "1"),
            ("bool_field_2:false AND bool_field_1:false", 0, None),
            ("bool_field_2:false AND text_field_1:true", 1, "1"),
        ]
        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            for filter_string, expected_hits, expected_id in test_cases:
                with (self.subTest(
                        f"search_method = {search_method}, filter_string={filter_string}, "
                        f"expected_hits={expected_hits}, expected_id={expected_id}")):
                    res = tensor_search.search(
                        index_name=self.default_text_index, config=self.config, text="search me",
                        search_method=search_method, filter=filter_string
                    )
                    self.assertEqual(expected_hits, len(res["hits"]))
                    if expected_id:
                        self.assertEqual(expected_id, res["hits"][0]["_id"])
                        expected_document = documents[0] if expected_id == "1" else \
                            documents[1] if expected_id == "2" else None
                        self.assertEqual(self.strip_marqo_fields(res["hits"][0], strip_id=False), expected_document)

    def test_filter_spaced_fields(self):
        # Add documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1233", "my_bool": True},
                    {"abc": "some text", "Floaty_Field": 0.548, "_id": "344", "my_bool": True},
                ],
                tensor_fields=["abc", "other_field"]
            )
        )

        # Define test parameters as tuples (filter_string, expected_hits, expected_ids)
        test_parameters = [
            ("other_field:baaadd", 1, ["5678"]),
            ("other_field:(Close match hehehe)", 2, ["1234", "1233"]),
            ("(Floaty_Field:[0 TO 1]) AND (abc:(some text))", 1, ["344"]),
        ]

        for filter_string, expected_hits, expected_ids in test_parameters:
            with self.subTest(f"filter_string={filter_string}, expected_hits={expected_hits}"):
                res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text='',
                    filter=filter_string, verbose=0
                )

                self.assertEqual(expected_hits, len(res["hits"]))
                for expected_id in expected_ids:
                    self.assertIn(expected_id, [hit['_id'] for hit in res['hits']])

    def test_filtering_bad_syntax(self):
        # Adding documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1233", "my_bool": True},
                ],
                tensor_fields=["abc", "other_field"]
            )
        )

        # Define test parameters as tuples (filter_string)
        bad_filter_strings = [
            "(other_field):baaadd",  # Incorrect syntax for field name with space
            "(an_int:[0 TO 30] and an_int:2) AND abc:(some text)",  # and instead of AND here
            "",  # Empty filter string
        ]

        for filter_string in bad_filter_strings:
            with self.subTest(f"filter_string={filter_string}"):
                with self.assertRaises(core_exceptions.FilterStringParsingError):
                    tensor_search.search(
                        config=self.config, index_name=self.default_text_index, text="some text",
                        result_count=3, filter=filter_string, verbose=0
                    )

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
                    "an_int": 1,
                    "a_float": 1.2,
                    "a_bool": True,
                    "some_str": "blah"
                }],
                tensor_fields=["some_str"]
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
            "an_int": 1,
            "a_float": 1.2,
            "a_bool": True,
            "some_str": "blah"
        }]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                tensor_fields=["some_str"]
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
                        "doc_title": "The captain bravely lead her followers into battle."
                                     " She directed her soldiers to and fro.",
                        "field_X": "some text",
                        "field1": "other things", "my_bool": True,
                        "_id": "123456", "a_float": 0.61
                    },
                    {
                        "doc_title": "The captain bravely lead her followers into battle."
                                     " She directed her soldiers to and fro.",
                        "field_X": "some text",
                        "my_bool_2": False,
                        "_id": "233",
                    },
                    {
                        "_id": "other doc", "a_float": 0.66, "bfield": "some text too", "my_int": 5,
                        "fake_int": "234", "fake_float": "1.23", "gapped_field_name": "gap"
                    },
                    {
                        "_id": "123457", "bfield": "true"
                    }
                ],
                tensor_fields=["doc_title", "field_X", "field1"])
        )

        # Define test parameters as tuples (filter_string, expected_hits, expected_id)
        test_parameters = [
            ("(my_bool:true AND a_float:[0.1 TO 0.75]) AND field1:(other things)", 1, "123456"),
            ("my_bool:True", 1, "123456"),
            ("my_bool:tRue", 1, "123456"),
            ("my_bool_2:false", 1, "233"),
            ("my_bool:false", 0, None),  # no hits for bool_field_1=false
            ("my_bool:some_value", 0, None),  # no hits for bool_field_1 not boolean
            ("my_looLoo:1", 0, None),
            ("my_int:5", 1, "other doc"),
            ("my_int:[1 TO 10]", 1, "other doc"),
            ("a_float:0.61", 1, "123456"),
            ("field1:(other things)", 1, "123456"),
            ("fake_int:234", 1, "other doc"),
            ("fake_float:1.23", 1, "other doc"),
            ("gapped_field_name:gap", 1, "other doc"),
            # ("bfield:true", 1, "123457")  # string field with boolean-like value # TODO - This fails due to a bug
        ]

        for filter_string, expected_hits, expected_id in test_parameters:
            with self.subTest(f"filter_string={filter_string}, expected_hits={expected_hits}"):
                res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="some text",
                    result_count=3, filter=filter_string, search_method=SearchMethod.LEXICAL
                )

                if expected_hits == math.inf:  # Handle the special case for "Any" hits
                    self.assertTrue(len(res["hits"]) >= 1)
                else:
                    self.assertEqual(expected_hits, len(res["hits"]))
                    if expected_id:
                        self.assertEqual(expected_id, res["hits"][0]["_id"])

    def test_filter_on_id_and_more(self):
        """Test various filtering scenarios including _id and other conditions"""
        # Adding documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {"abc": "some text", "other_field": "baaadd", "_id": "5678", "status": "active"},
                    {"abc": "some text", "other_field": "Close match hehehe", "_id": "1234", "status": "inactive"},
                    {"abc": "different text", "other_field": "irrelevant", "_id": "9012", "status": "active"}
                ],
                tensor_fields=["abc", "other_field"]
            )
        )

        test_cases = [
            ("filter on id 5678", "_id:5678", 1, ["5678"]),
            ("filter on id 1234", "_id:1234", 1, ["1234"]),
            ("AND filter", "_id:5678 AND status:active", 1, ["5678"]),
            ("OR filter", "_id:5678 OR _id:1234", 2, ["5678", "1234"]),
            ("Complex filter", "_id:5678 OR (abc:some\ text AND status:inactive)", 2, ["5678", "1234"]),
            ("Non-ID field filter", "status:active", 2, ["5678", "9012"]),
            ("No result filter", "_id:0000", 0, [])
        ]

        for name, filter_query, expected_count, expected_ids in test_cases:
            with self.subTest(name=name):
                res = tensor_search.search(
                    config=self.config, index_name=self.default_text_index, text="some text",
                    filter=filter_query
                )
                self.assertEqual(expected_count, len(res["hits"]))
                if expected_ids:
                    self.assertEqual(set(expected_ids), set([hit["_id"] for hit in res["hits"]]))

    def test_attributes_to_retrieve(self):
        docs = [
            {
                "field_1": "Exact match hehehe",
                "field_2": "baaadd",
                "random_field": "res res res",
                "random_lala": "res res res haha",
                "marqomarqo": "check check haha",
            }
        ]

        test_inputs = (
            (["void_field"], {"_id", "_score", "_highlights"}),
            ([], {"_id", "_score", "_highlights"}),
            (["field_1"], {"field_1", "_id", "_score", "_highlights"}),
            (["field_1", "field_2"], {"field_1", "field_2", "_id", "_score", "_highlights"}),
            (["field_1", "random_field"], {"field_1", "random_field", "_id", "_score", "_highlights"}),
            (["field_1", "random_field", "random_lala"],
             {"field_1", "random_field", "random_lala", "_id", "_score", "_highlights"}),
            (["field_1", "random_field", "random_lala", "marqomarqo"],
             {"field_1", "random_field", "random_lala", "marqomarqo", "_id", "_score", "_highlights"}),
            (["field_1", "field_2", "random_field", "random_lala", "marqomarqo"],
             {"field_1", "field_2", "random_field", "random_lala", "marqomarqo", "_id", "_score", "_highlights"}),
            (None, {"field_1", "field_2", "random_field", "random_lala", "marqomarqo", "_id", "_score", "_highlights"}),
        )

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                tensor_fields=["field_1", "field_2"]
            )
        )

        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            for searchable_attributes, expected_fields in test_inputs:
                with self.subTest(
                        f"search_method = {search_method}, searchable_attributes={searchable_attributes}, expected_fields = {expected_fields}"):
                    res = tensor_search.search(
                        config=self.config, index_name=self.default_text_index, text="Exact match hehehe",
                        attributes_to_retrieve=searchable_attributes, search_method=search_method
                    )

                    self.assertEqual(expected_fields, set(res["hits"][0].keys()))

    def test_limit_results(self):
        """"""
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"

        vocab = requests.get(vocab_source).text.splitlines()

        batch_size_list = [50, 50, 28]
        # We add 128 documents to the index wth batch_size 50, 50, 28 to avoid timeout
        for batch_size in batch_size_list:
            res = tensor_search.add_documents(
                config=self.config,
                add_docs_params=AddDocsParams(
                    index_name=self.default_text_index,
                    docs=[{"Title": "a test of" + (" ".join(random.choices(population=vocab, k=2)))}
                          for _ in range(batch_size)],
                    tensor_fields=["Title"]
                )
            )
        self.assertEqual(128, self.monitoring.get_index_stats_by_name(self.default_text_index).
                         number_of_documents)
        search_text = "a test of"

        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            for max_doc in [2, 5, 10, 100]:
                with self.subTest(f"search_method={search_method}, max_doc={max_doc}"):
                    mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}
                    with mock.patch.dict(os.environ, {**os.environ, **mock_environ}):
                        half_search = tensor_search.search(
                            search_method=search_method,
                            config=self.config,
                            index_name=self.default_text_index,
                            text=search_text,
                            result_count=max_doc // 2
                        )
                        self.assertEqual(max_doc // 2, half_search['limit'])
                        self.assertEqual(max_doc // 2, len(half_search['hits']))

                        limit_search = tensor_search.search(
                            search_method=search_method,
                            config=self.config,
                            index_name=self.default_text_index,
                            text=search_text,
                            result_count=max_doc
                        )
                        self.assertEqual(max_doc, limit_search['limit'])
                        self.assertEqual(max_doc, len(limit_search['hits']))

                        with self.assertRaises(errors.IllegalRequestedDocCount):
                            oversized_search = tensor_search.search(
                                search_method=search_method,
                                config=self.config,
                                index_name=self.default_text_index,
                                text=search_text,
                                result_count=max_doc + 1
                            )

                        with self.assertRaises(errors.IllegalRequestedDocCount):
                            very_oversized_search = tensor_search.search(
                                search_method=search_method,
                                config=self.config,
                                index_name=self.default_text_index,
                                text=search_text,
                                result_count=(max_doc + 1) * 2
                            )

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
        """does the URL get returned as the highlight? (it should - because no rerankers are being used)"""
        url_1 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        url_2 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
        docs = [
            {"_id": "123",
             "image_field": url_1,
             "text_field": "irrelevant text"
             },
            {"_id": "789",
             "image_field": url_2},
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
                tensor_fields=["image_field"]
            )
        )
        res = tensor_search.search(
            config=self.config, index_name=self.default_image_index,
            text="A hippo in the water", result_count=3,
        )
        assert len(res['hits']) == 2
        assert {hit['image_field'] for hit in res['hits']} == {url_2, url_1}
        assert {hit['_highlights'][0]['image_field'] for hit in res['hits']} == {url_2, url_1}

    def test_multi_search(self):
        docs = [
            {"field_a": "Doberman, canines, golden retrievers are humanity's best friends",
             "_id": 'dog_doc'},
            {"field_a": "All things poodles! Poodles are great pets",
             "_id": 'poodle_doc'},
            {"field_a": "Construction and scaffolding equipment",
             "_id": 'irrelevant_doc'}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                tensor_fields=["field_a"]
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

                # the poodle doc should be lower ranked than the irrelevant doc
                for hit_position, _ in enumerate(res['hits']):
                    self.assertEqual(expected_ordering[hit_position], res['hits'][hit_position]['_id'])

    def test_multi_search_images(self):
        docs = [
            {
                "loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {
                "loc b": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
                "_id": 'artefact_hippo'
            }
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
                tensor_fields=["loc a", "loc b"]
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
              "artefact": 1.0, "photo realistic": -1,
              },
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
                # the poodle doc should be lower ranked than the irrelevant doc
                for hit_position, _ in enumerate(res['hits']):
                    self.assertEqual(expected_ordering[hit_position], res['hits'][hit_position]['_id'])

    def test_multi_search_images_invalid_queries(self):
        docs = [
            {
                "loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {
                "field_a": "Some text about a weird forest",
                "_id": 'artefact_hippo'
            }
        ]

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
                tensor_fields=["loc", "field_a"]
            )
        )

        invalid_queries = [{}, None, {123: 123}, {'123': None},
                           {"https://marqo_not_real.com/image_1.png": 3}, set()]
        for q in invalid_queries:
            with self.subTest(f"query={q}"):
                with self.assertRaises((ValidationError, errors.InvalidArgError)) as e:
                    tensor_search.search(
                        text=q,
                        index_name=self.default_image_index,
                        result_count=5,
                        config=self.config,
                        search_method=SearchMethod.TENSOR)

    def test_multi_search_images_edge_cases(self):
        docs = [
            {
                "loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {
                "field_a": "Some text about a weird forest",
                "_id": 'artefact_hippo'
            }
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
                tensor_fields=["loc", "field_a"]
            )
        )

        alright_queries = [{"v ": 1.2}, {"d ": 0}, {"vf": -1}]
        for q in alright_queries:
            with self.subTest(f"query={alright_queries}"):
                tensor_search.search(
                    text=q,
                    index_name=self.default_image_index,
                    result_count=5,
                    config=self.config,
                    search_method=SearchMethod.TENSOR)

    def test_multi_search_images_lexical(self):
        """Error if you try this"""
        docs = [
            {"loc": "124", "_id": 'realistic_hippo'},
            {"field_a": "Some text about a weird forest",
             "_id": 'artefact_hippo'}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                tensor_fields=["loc", "field_a"]
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
        """This test is to ensure image search works as expected
        The code paths for image and search have diverged quite a bit
        """
        hippo_image = (
            'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        )
        doc_dict = {
            'realistic_hippo': {"loc": hippo_image,
                                "_id": 'realistic_hippo'},
            'artefact_hippo': {"field_a": "Some text about a weird forest",
                               "_id": 'artefact_hippo'}
        }

        docs = list(doc_dict.values())

        res = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_image_index,
                docs=docs,
                tensor_fields=["loc", "field_a"]
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
                tensor_fields=[]
            )
        )
        lexical_search_result = tensor_search.search(
            text="some text",
            index_name=self.default_text_index,
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
                tensor_fields=["text_field_1", "text_field_2"]
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
            {'long_field_1': 1, '_id': '0', "search_field": "some text"},  # small positive integer
            {'long_field_1': -1, '_id': '1', "search_field": "some text"},  # small negative integer
            # large positive integer that can't be handled by int
            {'long_field_1': 100232142864, '_id': '2', "search_field": "some text"},
            # large negative integer that can't be handled by int
            {'long_field_1': -923217213, '_id': '3', "search_field": "some text"},
            # large positive integer mathematical expression
            {'double_field_1': 10000000000.0, '_id': '4', "search_field": "some text"},
            # large negative integer mathematical expression
            {'double_field_1': -1000000000000.0, '_id': '5', "search_field": "some text"},
            # large positive float
            {'double_field_1': 10000000000.12325, '_id': '6', "search_field": "some text"},
            # large negative float
            {'double_field_1': -9999999999.87675, '_id': '7', "search_field": "some text"}
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=valid_documents,
                tensor_fields=["search_field"]
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

    def test_search_with_content_double_colon(self):
        docs = [
            {"_id": "1", "text_field": "::my_text"} # This should work properly
        ]
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=docs,
                tensor_fields=["text_field"]
            )
        )
        tensor_search_result = tensor_search.search(
            text="some text",
            index_name=self.default_text_index,
            config=self.config,
            search_method=SearchMethod.TENSOR,
        )
        self.assertEqual(1, len(tensor_search_result['hits']))
        self.assertEqual("1", tensor_search_result['hits'][0]['_id'])

    def test_search_returned_documents(self):
        """A test to ensure that the returned are not missing/adding any unexpected fields"""
        full_fields_document = ({
            "_id": "full_fields",
            "text_field": "some text",
            "int_field": 1,
            "float_field": 2.0,
            "bool_field": True,
            "list_field": ["a", "b","c"],
            "string_bool_field": "True",
            "string_int_field": "1",
            "string_float_field": "1.2",
            "string_list_field": "['a', 'b', 'c']"
        }, "full-fields document")

        partial_fields_document = ({
            "_id": "partial_field",
            "text_field": "some text",
            "float_field": 1.0,
            "bool_field": True,
            "list_field": ["a", "b", "c"],
        }, "partial-fields document")

        no_field_documents = ({
            "_id": "no_field",
            "text_field": "some text"
        }, "no-field document")

        for document, msg in [full_fields_document, partial_fields_document, no_field_documents]:
            with self.subTest(msg):
                self.clear_index_by_name(self.default_text_index)
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.default_text_index,
                        docs=[document],
                        tensor_fields=["text_field"]
                    )
                )

                search_result = tensor_search.search(
                    text="some text",
                    index_name=self.default_text_index,
                    config=self.config,
                    search_method=SearchMethod.TENSOR,
                )

                self.assertEqual(1, len(search_result['hits']))
                self.assertEqual(document, self.strip_marqo_fields(search_result['hits'][0], strip_id=False))

    def test_tensor_search_query_can_be_none(self):
        res = tensor_search.search(text=None, config=self.config, index_name=self.default_text_index,
                                   context=SearchContext(
                                       **{"tensor": [{"vector": [1, ] * 384, "weight": 1},
                                                     {"vector": [2, ] * 384, "weight": 2}]}))

        self.assertIn("hits", res)

    def test_lexical_query_can_not_be_none(self):
        context = SearchContext(
            **{"tensor": [{"vector": [1, ] * 384, "weight": 1},
                          {"vector": [2, ] * 384, "weight": 2}]})

        test_case = [
            (None, context, "with context"),
            (None, None, "without context")
        ]

        for query, context, msg in test_case:
            with self.subTest(msg):
                with self.assertRaises(InvalidArgError):
                    res = tensor_search.search(text=None, config=self.config, index_name=self.default_text_index,
                                               search_method=SearchMethod.LEXICAL)
                                                    