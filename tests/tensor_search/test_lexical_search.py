import time
from marqo.tensor_search import enums, backend
from marqo.tensor_search import tensor_search
import copy
from marqo.errors import InvalidArgError, IndexNotFoundError
from tests.marqo_test import MarqoTestCase
import random
import requests
import json
from marqo.tensor_search.models.add_docs_objects import AddDocsParams


class TestLexicalSearch(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}

        self.index_name_1 = "my-test-index-1"
        self.config = copy.deepcopy(self.config)
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

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

    
    def test_lexical_search_empty_text(self):
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some doc 1": "some field 2", "some doc 2": "some other thing"}], auto_refresh=True)
        )
        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="")
        assert len(res["hits"]) == 0
        assert res["hits"] == []

    def test_lexical_search_bad_text_type(self):
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some doc 1": "some field 2", "some doc 2": "some other thing"}], auto_refresh=True))
        bad_args = [None, 1234, 1.0]
        for a in bad_args:
            try:
                res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text=a)
                raise AssertionError
            except InvalidArgError as e:
                assert "type str" in str(e)

    def test_lexical_search_no_index(self):
        try:
            res = tensor_search._lexical_search(config=self.config, index_name="non existent index", text="abcdefg")
        except IndexNotFoundError as s:
            pass

    def test_lexical_search_multiple(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True,
                docs=[d1,
                      {"some doc 1": "some 2", "field abc": "robodog is not a cat", "_id": "unusual id"},
                      d0])
        )
        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="marqo field")
        assert len(res["hits"]) == 2
        assert res["hits"][0]["_id"] == "alpha alpha"
        assert res["hits"][1]["_id"] == "abcdef"
        assert self.strip_marqo_fields(res["hits"][0]) == d0
        assert self.strip_marqo_fields(res["hits"][1]) == d1

    def test_lexical_search_single_searchable_attribs(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d3 = {"TITITLE": "Tony from the way", "field lambda": "some prop field called marqo",
              "_id": "122"}
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field", "_id": "123"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d4, d1 ]))
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d3, d2]))
        res = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="marqo field",
             searchable_attributes=["field lambda"], result_count=3)
        assert len(res["hits"]) == 2
        assert res["hits"][0]["_id"] == "122"
        assert res["hits"][1]["_id"] == "123"
        assert self.strip_marqo_fields(res["hits"][0], strip_id=False) == d3
        assert self.strip_marqo_fields(res["hits"][1]) == d4

    def test_lexical_search_multiple_searchable_attribs(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d3 = {"TITITLE": "Tony from the way", "_id": "122",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field", "_id": "123"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d4, d1]))
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[d3, d2])
        )
        res = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
             searchable_attributes=["field lambda", "FIELD omega"])
        assert len(res["hits"]) == 3
        assert self.strip_marqo_fields(res["hits"][0]) == d1
        assert self.strip_marqo_fields(res["hits"][1]) == d3
        assert self.strip_marqo_fields(res["hits"][2]) == d4

    def test_lexical_search_result_count(self):
        d0 = {
            "some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field",}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat"}
        d3 = {"TITITLE": "Tony from the way",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. Some other text. "
                                                "Trying to reduce often the keywords appear here. SMH "
                                                "Another bunch of words that may mean something. "
                                                "Just a slight mention of a field"}
        d5 = {"some completely irrelevant": "document hehehe"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d4, d1, d3, d2]))
        r1 = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            result_count=2
        )
        assert len(r1["hits"]) == 2
        r2 = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            result_count=1000
        )
        assert len(r2["hits"]) == 4
        r3 = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            result_count=0
        )
        assert len(r3["hits"]) == 0

    def test_search_lexical_param(self):
        """tensor_search.search(search_method='lexical') should behave identically to
            lexical_search() for a given set of args
        """
        d0 = {
            "some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field",}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat"}
        d3 = {"TITITLE": "Tony from the way",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field"}
        d5 = {"some completely irrelevant": "document hehehe"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d4, d1, d3, d2]))
        res_lexical_search = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            searchable_attributes=["field lambda", "FIELD omega"])
        res_search_entry_point = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            searchable_attributes=["field lambda", "FIELD omega"],
            search_method=enums.SearchMethod.LEXICAL)
        res_search_entry_point_no_processing_time = res_search_entry_point.copy()
        del res_search_entry_point_no_processing_time ['processingTimeMs']
        del res_search_entry_point_no_processing_time ['query']
        del res_search_entry_point_no_processing_time ['limit']
        del res_search_entry_point_no_processing_time ['offset']
        assert len(res_lexical_search['hits']) > 0
        assert res_search_entry_point_no_processing_time == res_lexical_search


    def test_lexical_search_overwriting_doc(self):
        """can we overwrite doc and do a search on the latest doc?"""
        a_consistent_id = "my id is awesome"
        d0 = {
            "_id": a_consistent_id,
            "some doc 1": "some  2",
            "the big field": "very unlikely theory. is pretty awesom, in the",
            "boring field": "4 grey boring walls. "
        }
        d1 = {
            "_id": a_consistent_id,
            "some doc 1": "some FIELD 2",
            "the big field": "just your average doc...",
            "Cool field": "Marqo is the best!"
        }
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d0]))
        assert [] == tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field")["hits"]
        grey_query = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="4 grey boring walls")
        assert len (grey_query["hits"]) == 1
        assert grey_query["hits"][0]["_id"] == a_consistent_id
        # update doc so it does indeed get returned
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d1]))
        cool_query = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field")
        assert a_consistent_id == cool_query["hits"][0]["_id"]
        assert len(cool_query["hits"]) == 1
        assert [] == tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="4 grey boring walls")["hits"]

    def test_lexical_search_filter(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field", "Lucy":"Travis"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d3 = {"TITITLE": "Tony from the way", "field lambda": "some prop field called marqo",
              "_id": "122"}
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field", "day": 190,
              "_id": "123"}
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d4, d1 ]))
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, auto_refresh=True,
                docs=[d3, d2]))
        res = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="marqo field",
             filter_string="title:Marqo OR (Lucy:Travis AND day:>50)"
            , result_count=3)
        assert len(res["hits"]) == 2
        assert res["hits"][0]["_id"] == "123" or res["hits"][1]["_id"] == "123"
        assert res["hits"][0]["_id"] == "abcdef" or res["hits"][1]["_id"] == "abcdef"

    def test_lexical_search_empty_searchable_attribs(self):
        """Empty searchable attribs searches all fields"""
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "extravagant very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "extravagant robodog is not a cat", "_id": "Jupyter_12"}

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d1, d2])
        )
        res = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="extravagant",
             searchable_attributes=[], result_count=3)
        assert len(res["hits"]) == 2
        assert (res["hits"][0]["_id"] == "alpha alpha") or (res["hits"][0]["_id"] == "Jupyter_12")
        assert (res["hits"][0]["_id"] != "abcdef") and (res["hits"][0]["_id"] != "abcdef")

    def test_lexical_search_double_quotes(self):
        # 2-tuples of input text, and required terms expected to be in the results.
        docs = [
            {
                "Field 1": "gender is male. gang is cyberpunk. accessory is none.",
                "Field 2": "",
                "Field 3": "",
                "_id": "0"
            },
            {
                "Field 1": "gender is male. gang is cyberpunk. accessory is necklace.",
                "Field 2": "",
                "Field 3": "",
                "_id": "1"
            },
            {
                "Field 1": "gender is male. gang is random. accessory is none.",
                "Field 2": "",
                "Field 3": "",
                "_id": "2"
            },
            {
                "Field 1": "gender is male. gang is random. accessory is necklace.",
                "Field 2": "",
                "Field 3": "",
                "_id": "3"
            },
            {
                "Field 1": "gender is female. gang is cyberpunk. accessory is none.",
                "Field 2": "",
                "Field 3": "",
                "_id": "4"
            },
            {
                "Field 1": "gender is female. gang is cyberpunk. accessory is necklace.",
                "Field 2": "",
                "Field 3": "accessory is none.",
                "_id": "5"
            },
            {
                "Field 1": "gender is female. gang is random. accessory is none.",
                "Field 2": "",
                "Field 3": "",
                "_id": "6"
            },
            {
                "Field 1": "gender is female. gang is random. accessory is necklace.",
                "Field 2": "accessory is none.",
                "Field 3": "",
                "_id": "7"
            },
        ]
        fields = ["Field 1", "Field 2", "Field 3"]

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=docs, auto_refresh=False)
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)

        cases = [
            {"input": '"gender is female"', "required_terms": ["gender is female"]},
            {"input": '"gender is female" "random"', "required_terms": ["gender is female", "random"]},
            {"input": 'male cyberpunk none "accessory is necklace"', 
                "required_terms": ["accessory is necklace"],
                "first_n_results_ordered": ['5', '1', '3', '7']},
            {"input": '"cyberpunk1234" necklace', "no_results": True},
            {"input": 'cyberpunk1234 necklace',
                "first_n_results_unordered": ['7', '1', '3', '5']},
            {"input": '"accessory is none"', "required_terms": ["accessory is none"]}, # Multi-field testing

            # Escaped quotes
            {"input": '\\"fake term\\" not required, this should yield results male',
                "first_n_results_unordered": ['3', '0', '2', '1']},
            {"input": '"fake term" is required, this should yield no results', "no_results": True},

            # Syntax errors (should not return errors)
            {"input": '"gender is fe"male male"',
                "first_n_results_unordered": ['3', '0', '2', '1']},
            {"input": '"""', "no_results": True},
            {"input": '"term1 " term2 "', "no_results": True},
            {"input": '"AND OR &*) ((', "no_results": True}
        ]

        for case in cases:
            res = tensor_search._lexical_search(
                config=self.config, index_name=self.index_name_1, text=case['input'],
                 searchable_attributes=fields, result_count=8)

            id_only_hits = [hit["_id"] for hit in res["hits"]]

            if "required_terms" in case:
                for hit in res["hits"]:
                    for term in case["required_terms"]:
                        term_found = False

                        for field in fields:
                            if term in hit[field]:
                                term_found = True
                                break
                        
                        assert term_found
            
            if "first_n_results_unordered" in case:
                n = len(case["first_n_results_unordered"])
                assert set(id_only_hits[:n]) == set(case["first_n_results_unordered"]) 
            
            if "first_n_results_ordered" in case:
                n = len(case["first_n_results_ordered"])
                assert id_only_hits[:n] == case["first_n_results_ordered"]
            
            if "no_results" in case:
                assert len(id_only_hits) == 0

    def test_lexical_search_list(self):
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[
                    {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "_id": "1235",  "my_list": ["tag1", "tag2 some"]},
                    {"abc": "some text", "_id": "1001", "my_cool_list": ["b_1", "b2"], "fun list": ['truk', 'car']},
                ], auto_refresh=True, non_tensor_fields=["my_list", "fun list", "my_cool_list"]))
        base_search_args = {
            'index_name': self.index_name_1, "config": self.config,
            "search_method": enums.SearchMethod.LEXICAL
        }
        res_exists = tensor_search.search(**{'text': "tag1", **base_search_args})
        assert len(res_exists['hits']) == 1
        assert res_exists['hits'][0]['_id'] == '1235'

        res_not_exists = tensor_search.search(**{'text': "tag55", **base_search_args})
        assert len(res_not_exists['hits']) == 0

        res_filtered_other = tensor_search.search(
            **{'text': "tag1", 'filter': 'abc:(some text)', **base_search_args})
        # it will actually return the other docs, as they match the filter, but with scores of 0
        assert res_filtered_other['hits'][0]['_id'] == '1235'

        res_filtered_same = tensor_search.search(
            **{'text': "tag1", 'filter': 'my_list:tag2', **base_search_args})
        assert len(res_filtered_same['hits']) == 1
        assert res_filtered_same['hits'][0]['_id'] == '1235'

        res_filtered_other_list = tensor_search.search(
            **{'text': "b_1", 'filter': 'fun\ list:truk', **base_search_args})
        assert len(res_filtered_other_list['hits']) == 1
        assert res_filtered_other_list['hits'][0]['_id'] == '1001'

    def test_lexical_search_list_searchable_attr(self):
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[
                    {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                    {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                    {"abc": "some text", "_id": "1235",  "my_list": ["tag1", "tag2 some"]},
                    {"abc": "some text", "_id": "1001", "my_cool_list": ["b_1", "b2"], "fun list": ['truk', 'car']},
                ], auto_refresh=True, non_tensor_fields=["my_list", "fun list", "my_cool_list"])
        )
        base_search_args = {
            'index_name': self.index_name_1, "config": self.config,
            "search_method": enums.SearchMethod.LEXICAL, 'text': "tag1"
        }
        res_exists = tensor_search.search(
            **{**base_search_args, "searchable_attributes": ["my_list"]})
        assert len(res_exists['hits']) == 1
        assert res_exists['hits'][0]['_id'] == '1235'

        res_wrong_attr = tensor_search.search(
            **{**base_search_args, "searchable_attributes": ["abc"]})
        assert len(res_wrong_attr['hits']) == 0

    def test_lexical_search_filter_with_dot(self):
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[
                {"content": "a man on a horse", "filename" : "Important_File_1.pdf", "_id":"123"},
                {"content": "the horse is eating grass", "filename": "Important_File_2.pdf", "_id": "456"},
                {"content": "what is the document", "filename": "Important_File_3.pdf", "_id": "789"},

            ], auto_refresh=True)
        )

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1,
                                            text="horse",  searchable_attributes=["content"],
                                            filter_string="filename: \"Important_File_1.pdf\"", result_count=8)

        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "123"

        res = tensor_search._vector_text_search(config=self.config, index_name=self.index_name_1,
                                            query="horse",  searchable_attributes=["content"],
                                            filter_string="filename: Important_File_1.pdf", result_count=8)

        assert len(res["hits"]) == 1
        assert res["hits"][0]["_id"] == "123"

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1,
                                            text="horse",  searchable_attributes=["content"],
                                            filter_string="filename: Important_File_1.pdf", result_count=8)

        assert len(res["hits"]) == 3, "this is a bug at the moment. the filter is not applied. " \
                                      "fix it will introduce the error."