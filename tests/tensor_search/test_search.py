import math
import pprint
from unittest import mock

from marqo.tensor_search.enums import TensorField, SearchMethod
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase


class TestVectorSearch(MarqoTestCase):

    def setUp(self) -> None:
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.index_name_3 = "my-test-index-3"
        self._delete_test_indices()

    def _delete_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_delete = [self.index_name_1, self.index_name_2, self.index_name_3]
        else:
            ix_to_delete = indices
        for ix_name in ix_to_delete:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix_name)
            except IndexNotFoundError as s:
                pass

    def test_vector_search_searchable_attributes_non_existent(self):
        """TODO: non existent attrib."""

    def test_each_doc_returned_once(self):
        """TODO: make sure each return only has one doc for each ID,
                - esp if matches are found in multiple fields
        """
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe efgh ", "other field": "baaadd efgh ",
                 "_id": "5678", "finally": "some field efgh "},
                {"abc": "shouldn't really match ", "other field": "Nope.....",
                 "_id": "1234", "finally": "Random text here efgh "},
            ], auto_refresh=True)
        search_res = tensor_search._vector_text_search(
            config=self.config, index_name=self.index_name_1, text=" efgh ",
            return_doc_ids=True, number_of_highlights=2, result_count=10
        )
        assert len(search_res['hits']) == 2

    def test_vector_text_search_validate_result_count(self):
        try:
            tensor_search._vector_text_search(
                config=self.config, index_name=self.index_name_1, result_count=-1, text="some text...")
        except InvalidArgError as e:
            assert "illegal result_count" in str(e)

        try:
            tensor_search._vector_text_search(
                config=self.config, index_name=self.index_name_1,
                result_count=constants.MAX_VECTOR_SEARCH_RESULT_COUNT + 1, text="some text...")
        except InvalidArgError as e:
            assert "illegal result_count" in str(e)

    def test_vector_search_against_empty_index(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        search_res = tensor_search._vector_text_search(
                config=self.config, index_name=self.index_name_1,
                result_count=5, text="some text...")
        assert {'hits': []} == search_res

    def test_vector_search_against_non_existent_index(self):
        try:
            tensor_search._vector_text_search(
                config=self.config, index_name="some-non-existent-index",
                result_count=5, text="some text...")
        except IndexNotFoundError as s:
            pass

    def test_vector_search_long_query_string(self):
        query_text = """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
                    The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
                    """
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"_id": "12345", "Desc": "The Guardian is newspaper, read in the UK and other places around the world"},
                {"_id": "abc12334", "Title": "Grandma Jo's family recipe. ",
                 "Steps": "1. Cook meat. 2: Dice Onions. 3: Serve."},
            ], auto_refresh=True)
        search_res = tensor_search._vector_text_search(
            config=self.config, index_name=self.index_name_1, text=query_text,
            return_doc_ids=True
        )

    def test_vector_search_all_highlights(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe efgh ", "other field": "baaadd efgh ",
                 "_id": "5678", "finally": "some field efgh "},
                {"abc": "random text efgh ", "other field": "Close matc efgh h hehehe",
                 "_id": "1234", "finally": "Random text here efgh "},
            ], auto_refresh=True)
        search_res = tensor_search._vector_text_search(
            config=self.config, index_name=self.index_name_1, text=" efgh ",
            return_doc_ids=True, number_of_highlights=None, simplified_format=False
        )
        for res in search_res['hits']:
            assert len(res["highlights"]) == 3

    def test_vector_search_n_highlights(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe efgh ", "other field": "baaadd efgh ",
                 "_id": "5678", "finally": "some field efgh "},
                {"abc": "random text efgh ", "other field": "Close matc efgh h hehehe",
                 "_id": "1234", "finally": "Random text here efgh "},
            ], auto_refresh=True)
        search_res = tensor_search._vector_text_search(
            config=self.config, index_name=self.index_name_1, text=" efgh ",
            return_doc_ids=True, number_of_highlights=2, simplified_format=False
        )
        for res in search_res['hits']:
            assert len(res["highlights"]) == 2

    def test_vector_search_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
            searchable_attributes=["other field"], return_doc_ids=True
        )
        assert search_res["hits"][0]["_id"] == "1234"
        for res in search_res["hits"]:
            assert list(res["_highlights"].keys()) == ["other field"]

    def test_vector_search_searchable_attributes_multiple(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
            searchable_attributes=["other field", "Cool Field 1"], return_doc_ids=True
        )
        assert search_res["hits"][0]["_id"] == "1234"
        assert search_res["hits"][1]["_id"] == "9000"
        for res in search_res["hits"]:
            assert "abc" not in res["_highlights"]

    def test_tricky_search(self):
        """We ran into bugs with this doc"""
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs = [
                {
                    'text': 'In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium.hey include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements.',
                    'combined': 'In addition to NiS collection fire assay for a five element PGM suite, the samples will undergo research quality analyses for a wide range of elements, including the large ion. , the rare earth elements, high field strength elements, sulphur and selenium.hey include 55 elements of the periodic system: O, Si, Al, Ti, B, C, all the alkali and alkaline-earth metals, the halogens, and many of the rare elements.'
                },
                {
                    "abc": "defgh",
                    "this cat sat": "on the mat"
                }
            ], auto_refresh=True
        )
        res = tensor_search.search(
            text="In addition to NiS collection fire assay for a five element",
            config=self.config, index_name=self.index_name_1)

    def test_search_format(self):
        """Is the result formatted correctly?"""
        q = "Exact match hehehe"
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
            searchable_attributes=["other field", "Cool Field 1"], return_doc_ids=True, result_count=50
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
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="", return_doc_ids=True
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
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        try:
            # too big
            search_res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
                searchable_attributes=["other field", "Cool Field 1"], return_doc_ids=True, result_count=-1
            )
            raise AssertionError
        except InvalidArgError as e:
            assert "result count" in str(e)
        try:
            # too small
            search_res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
                searchable_attributes=["other field", "Cool Field 1"], return_doc_ids=True, result_count=1000000
            )
            raise AssertionError
        except InvalidArgError as e:
            assert "result count" in str(e)
        # should work with 0
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
            searchable_attributes=["other field", "Cool Field 1"], return_doc_ids=True, result_count=0
        )
        assert len(search_res["hits"]) == 0

    def test_highlights_tensor(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)

        tensor_highlights = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text",
            search_method=SearchMethod.TENSOR, highlights=True)
        assert len(tensor_highlights["hits"]) == 2
        for hit in tensor_highlights["hits"]:
            assert "_highlights" in hit

        tensor_no_highlights = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text",
            search_method=SearchMethod.TENSOR, highlights=False)
        assert len(tensor_no_highlights["hits"]) == 2
        for hit in tensor_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_highlights_lexical(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)

        lexical_highlights = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text",
            search_method=SearchMethod.LEXICAL, highlights=True)
        assert len(lexical_highlights["hits"]) == 2
        for hit in lexical_highlights["hits"]:
            assert "_highlights" in hit

        lexical_no_highlights = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text",
            search_method=SearchMethod.LEXICAL, highlights=False)
        assert len(lexical_no_highlights["hits"]) == 2
        for hit in lexical_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_search_lexical_int_field(self):
        """doesn't error out if there is a random int field"""
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_int": 144},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "my_int": 88},
            ], auto_refresh=True)

        s_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="cool match",
            search_method=SearchMethod.LEXICAL)
        pprint.pprint(s_res)
        assert len(s_res["hits"]) > 0

    def test_search_vector_int_field(self):
        """doesn't error out if there is a random int field"""
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_int": 144},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "my_int": 88},
            ], auto_refresh=True)

        s_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="88",
            search_method=SearchMethod.TENSOR)
        pprint.pprint(s_res)
        assert len(s_res["hits"]) > 0

    def test_filtering(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True},
            ], auto_refresh=True)

        res_doesnt_exist = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="my_string:c", verbose=1
        )

        res_exists_int = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="an_int:2", verbose=1
        )

        res_exists_string = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="my_string:b", verbose=1
        )

        res_field_doesnt_exist = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="my_int_something:5", verbose=1
        )

        res_range_doesnt_exist = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="an_int:[5 TO 30]", verbose=1
        )

        res_range_exists = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="an_int:[0 TO 30]", verbose=1
        )

        res_bool = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="my_bool:true", verbose=1
        )

        res_multi = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="an_int:[0 TO 30] OR my_bool:true", verbose=1
        )

        res_complex = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="(an_int:[0 TO 30] and an_int:2) AND abc:(some text)", verbose=1
        )

        assert res_exists_int["hits"][0]["_id"] == "1234"
        assert len(res_exists_int["hits"]) == 1

        assert res_exists_string["hits"][0]["_id"] == "5678"
        assert len(res_exists_string["hits"]) == 1

        assert len(res_field_doesnt_exist["hits"]) == 0
        assert len(res_range_doesnt_exist["hits"]) == 0
        assert len(res_doesnt_exist["hits"]) == 0

        assert res_range_exists["hits"][0]["_id"] == "1234"
        assert len(res_range_exists["hits"]) == 1

        assert res_bool["hits"][0]["_id"] == "1233"
        assert len(res_bool["hits"]) == 1

        assert len(res_multi["hits"]) == 2

        assert len(res_complex["hits"]) == 1

        assert 3 == len(tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=4,
            filter="*:*", verbose=1
        )["hits"])

    def test_filtering_bad_syntax(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True},
            ], auto_refresh=True)
        try:
            res_doesnt_exist = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
                filter="(other field):baaadd", verbose=1
            )
            raise AssertionError
        except InvalidArgError:
            pass

    def test_set_device(self):
        """calling search with a specified device overrides device defined in config"""
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="some text",
                search_method=SearchMethod.TENSOR, highlights=True, device="cuda:123")
            return True

        assert run()
        assert mock_config.search_device == "cpu"
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:123"

    def test_search_other_types_subsearch(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[{
                "an_int": 1,
                "a_float": 1.2,
                "a_bool": True,
                "some_str": "blah"
            }])
        for to_search in [1, 1.2, True, "blah"]:
            assert "hits" in tensor_search._lexical_search(
                text=str(to_search), config=self.config, index_name=self.index_name_1,

            )
            assert "hits" in tensor_search._vector_text_search(
                text=str(to_search), config=self.config, index_name=self.index_name_1
            )

    def test_search_other_types_top_search(self):
        docs = [{
            "an_int": 1,
            "a_float": 1.2,
            "a_bool": True,
            "some_str": "blah"
        }]
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=docs)
        for field, to_search in docs[0].items():
            assert "hits" in tensor_search.search(
                text=str(to_search), config=self.config, index_name=self.index_name_1,
                search_method=SearchMethod.TENSOR, filter=f"{field}:{to_search}"

            )
            assert "hits" in tensor_search.search(
                text=str(to_search), config=self.config, index_name=self.index_name_1,
                search_method=SearchMethod.LEXICAL, filter=f"{field}:{to_search}"
            )

    def test_lexical_filtering(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {
                    "doc title": "The captain bravely lead her followers into battle."
                                 " She directed her soldiers to and fro.",
                    "field X": "some text",
                    "field1": "other things", "my_bool": True,
                    "_id": "123456", "a_float": 0.61
                },
                {
                    "_id": "other doc", "a_float": 0.66, "bfield": "some text too", "my_int":5,
                    "fake_int":"234", "fake_float":"1.23", "gapped field_name": "gap"
                }
            ], auto_refresh=True)

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
            filter="(my_bool:true AND a_float:[0.1 TO 0.75]) AND field1:(other things)",
            search_method=SearchMethod.LEXICAL
        )
        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "123456"

        pairs = [
            ("my_looLoo:1", None), ("", None), ("*:*", math.inf),
             ("my_int:5", "other doc"), ("my_int:[1 TO 10]", "other doc"),
             ("a_float:0.61", "123456"), ("field1:(other things)", "123456"),
             ("fake_int:234", "other doc"), ("fake_float:1.23", "other doc"),
             ("fake_float:[0 TO 2]", "other doc"), ("gapped\ field_name:gap", "other doc")
        ]

        for filter, expected in pairs:
            check_res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
                filter=filter,
                search_method=SearchMethod.LEXICAL
            )
            if expected is None:
                assert 0 == len(check_res["hits"])
            elif expected == math.inf:  # stand-in for "Any"
                assert 2 == len(check_res["hits"])
            else:
                assert 1 == len(check_res["hits"])
                assert expected == check_res["hits"][0]["_id"]

    def test_vector_search_error_if_tensor_search_against_s2search(self):
        mock_config = copy.deepcopy(self.config)
        mock_config.cluster_is_s2search = True

        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {
                    "doc title": "The captain bravely lead her followers into battle."
                                 " She directed her soldiers to and fro.",
                    "field X": "some text",
                    "field1": "other things", "my_bool": True,
                    "_id": "123456", "a_float": 0.61
                },
                {
                    "_id": "other doc", "a_float": 0.66, "bfield": "some text too", "my_int": 5,
                    "fake_int": "234", "fake_float": "1.23", "gapped field_name": "gap"
                }
            ], auto_refresh=True)

        # Lexical filtering should work:
        assert tensor_search.search(
            config=mock_config, text=" ", filter="a_float:[0.62 TO 0.7]", index_name=self.index_name_1,
            search_method=SearchMethod.LEXICAL)["hits"][0]["_id"] == "other doc"
        try:
            # Tensor search errors out:
            tensor_search.search(config=mock_config, text=" ", filter="a_float:[0.5 TO 0.7]",
                                 index_name=self.index_name_1, search_method=SearchMethod.TENSOR)
            raise AssertionError
        except InvalidArgError:
            pass

    def test_attributes_to_retrieve_vector(self):
        docs = {
            "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
                          "abc": "random text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
            attributes_to_retrieve=["other field", "Cool Field 1"], return_doc_ids=True,
            search_method="TENSOR"
        )
        assert len(search_res["hits"]) == 3
        for res in search_res["hits"]:
            assert docs[res["_id"]]["other field"] == res["other field"]
            assert docs[res["_id"]]["Cool Field 1"] == res["Cool Field 1"]
            assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == \
                   {"other field", "Cool Field 1", "_id"}

    def test_attributes_to_retrieve_lexical(self):
        docs = {
            "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
                          "abc": "random text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
            attributes_to_retrieve=["other field", "Cool Field 1"], return_doc_ids=True, search_method="LEXICAL"
        )
        assert len(search_res["hits"]) == 3
        for res in search_res["hits"]:
            assert docs[res["_id"]]["other field"] == res["other field"]
            assert docs[res["_id"]]["Cool Field 1"] == res["Cool Field 1"]
            assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == \
                   {"other field", "Cool Field 1", "_id"}

    def test_attributes_to_retrieve_empty(self):
        docs = {
            "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
                          "abc": "random text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for method in ("LEXICAL", "TENSOR"):
            search_res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
                attributes_to_retrieve=[], return_doc_ids=True, search_method=method
            )
            assert len(search_res["hits"]) == 3
            for res in search_res["hits"]:
                assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == {"_id"}

    def test_attributes_to_retrieve_empty_index(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        assert 0 == tensor_search.get_stats(config=self.config, index_name=self.index_name_1)['numberOfDocuments']
        for to_retrieve in [[], ["some field name"], ["some field name", "wowowow field"]]:
            for method in ("LEXICAL", "TENSOR"):
                search_res = tensor_search.search(
                    config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
                    attributes_to_retrieve=to_retrieve, return_doc_ids=True, search_method=method
                )
                assert len(search_res["hits"]) == 0
                assert search_res['query'] == "Exact match hehehe"

    def test_attributes_to_retrieve_non_existent(self):
        docs = {
            "5678": {"abc": "Exact a match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhata  match", "_id": "rgjknrgnj",
                          "abc": "random a text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat a match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for to_retrieve in [[], ["non existing field name"], ["other field", "non existing field name"]]:
            for method in ("TENSOR", "LEXICAL"):
                search_res = tensor_search.search(
                    config=self.config, index_name=self.index_name_1, text=" a ",
                    attributes_to_retrieve=to_retrieve, return_doc_ids=True, search_method=method
                )
                assert len(search_res["hits"]) == 3
                for res in search_res["hits"]:
                    assert "non existing field name" not in res
                    assert set(k for k in res.keys()
                               if k not in TensorField.__dict__.values() and k != "_id"
                               ).issubset(to_retrieve)

    def test_attributes_to_retrieve_and_searchable_attribs(self):
        docs = {
            "i_1": {"field_1": "a", "other field": "baaadd",
                    "Cool Field 1": "res res res", "_id": "i_1"},
            "i_2": {"field_1": "a", "_id": "i_2",
                    "field_2": "a", "other field": "Close match hehehe"},
            "i_3": {"field_1": " a ", "_id": "i_3", "field_2": "a",
                    "field_3": "a "}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for to_retrieve, to_search, expected_ids, expected_fields in [
            (["field_1"], ["field_3"], ["i_3"], ["field_1"]),
            (["field_3"], ["field_1"], ["i_1", "i_2", "i_3"], ["field_3"]),
            (["field_1", "field_2"], ["field_2", "field_3"], ["i_2", "i_3"], ["field_1", "field_2"]),
        ]:
            for method in ("TENSOR", "LEXICAL"):
                search_res = tensor_search.search(
                    config=self.config, index_name=self.index_name_1, text="a",
                    attributes_to_retrieve=to_retrieve, return_doc_ids=True, search_method=method,
                    searchable_attributes=to_search
                )
                assert len(search_res["hits"]) == len(expected_ids)
                assert set(expected_ids) == {h['_id'] for h in search_res["hits"]}
                for res in search_res["hits"]:
                    relevant_fields = set(expected_fields).intersection(set(docs[res["_id"]].keys()))
                    assert set(k for k in res.keys()
                               if k not in TensorField.__dict__.values() and k != "_id"
                               ) == relevant_fields

    def test_attributes_to_retrieve_non_list(self):
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1,
                                    docs=[{"cool field 111": "this is some content"}],
                                    auto_refresh=True)
        for method in ("TENSOR", "LEXICAL"):
            for bad_attr in ["jknjhc", "", dict(), 1234, 1.245]:
                try:
                    print("bad_attrbad_attrbad_attr",bad_attr)
                    tensor_search.search(
                        config=self.config, index_name=self.index_name_1, text="a",
                        attributes_to_retrieve=bad_attr, return_doc_ids=True, search_method=method,
                    )
                    raise AssertionError
                except (InvalidArgError, InvalidFieldNameError):
                    pass


