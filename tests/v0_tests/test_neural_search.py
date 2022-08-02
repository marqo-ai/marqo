import copy

import marqo
from marqo.client import Client
from marqo.errors import MarqoApiError
import unittest
import pprint


class TestAddDocuments(unittest.TestCase):

    def setUp(self) -> None:
        self.client = Client(url='https://localhost:9200', main_user="admin",main_password="admin")
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    @staticmethod
    def strip_marqo_fields(doc, strip_id=True):
        """Strips Marqo fields from a returned doc to get the original doc"""
        copied = copy.deepcopy(doc)

        strip_fields = ["_highlights", "_score"]
        if strip_id:
            strip_fields += ["_id"]

        for to_strip in strip_fields:
            del copied[to_strip]

        return copied

    def test_search_single(self):
        """Searches an index of a single doc.
        Checks the basic functionality and response structure"""
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
            "Title": "This is a title about some doc. ",
            "Description": """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
            The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
            """
        }
        add_doc_res = self.client.index(self.index_name_1).add_documents([d1])
        search_res = self.client.index(self.index_name_1).search(
            "title about some doc")
        assert len(search_res["hits"]) == 1
        assert self.strip_marqo_fields(search_res["hits"][0]) == d1
        assert len(search_res["hits"][0]["_highlights"]) > 0
        assert ("Title" in search_res["hits"][0]["_highlights"]) or ("Description" in search_res["hits"][0]["_highlights"])

    def test_search_empty_index(self):
        self.client.create_index(index_name=self.index_name_1)
        search_res = self.client.index(self.index_name_1).search(
            "title about some doc")
        assert len(search_res["hits"]) == 0

    def test_search_multi(self):
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
                "doc title": "Cool Document 1",
                "field 1": "some extra info",
                "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
            }
        d2 = {
                "doc title": "Just Your Average Doc",
                "field X": "this is a solid doc",
                "_id": "123456"
        }
        res = self.client.index(self.index_name_1).add_documents([
            d1, d2
        ])
        search_res = self.client.index(self.index_name_1).search(
            "this is a solid doc")
        assert d2 == self.strip_marqo_fields(search_res['hits'][0], strip_id=False)
        assert search_res['hits'][0]['_highlights']["field X"] == "this is a solid doc"

    def test_select_lexical(self):
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
            "doc title": "Very heavy, dense metallic lead.",
            "field 1": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "doc title": "The captain bravely lead her followers into battle."
                         " She directed her soldiers to and fro.",
            "field X": "this is a solid doc",
            "_id": "123456"
        }
        res = self.client.index(self.index_name_1).add_documents([
            d1, d2
        ])

        # Ensure that vector search works
        search_res = self.client.index(self.index_name_1).search(
            "Examples of leadership")
        assert d2 == self.strip_marqo_fields(search_res["hits"][0], strip_id=False)
        assert search_res["hits"][0]['_highlights']["doc title"].startswith("The captain bravely lead her followers")

        # try it with lexical search:
        #    can't find the above with synonym
        assert len(self.client.index(self.index_name_1).search(
            "Examples of leadership", search_method=marqo.SearchMethods.LEXICAL)["hits"]) == 0
        #    but can look for a word
        assert self.client.index(self.index_name_1).search(
            '"captain"')["hits"][0]["_id"] == "123456"
