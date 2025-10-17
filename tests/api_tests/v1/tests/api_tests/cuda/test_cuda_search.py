import copy
import uuid

import marqo
import pytest
from marqo import enums
from marqo.client import Client

from tests.marqo_test import MarqoTestCase


@pytest.mark.cuda_test
class TestCudaStructuredSearch(MarqoTestCase):
    text_index_name = "api_test_structured_index_text" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "api_test_structured_image_index_image" + str(uuid.uuid4()).replace('-', '')
    filter_test_index_name = "api_test_structured_filter_index_filter" + str(uuid.uuid4()).replace('-', '')

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
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.filter_test_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "field_a", "type": "text", "features": ["filter"]},
                    {"name": "field_b", "type": "text", "features": ["filter"]},
                    {"name": "str_for_filtering", "type": "text", "features": ["filter"]},
                    {"name": "int_for_filtering", "type": "int", "features": ["filter"]},
                ],
                "tensorFields": ["field_a", "field_b"],
            },
            {
                "indexName": cls.image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_content", "type": "image_pointer"},
                ],
                "tensorFields": ["title", "image_content"],
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name, cls.filter_test_index_name]

    def setUp(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

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

    def test_search_single_doc(self):
        """Searches an index of a single doc.
        Checks the basic functionality and response structure"""
        d1 = {
            "title": "This is a title about some doc. ",
            "content": """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
            The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
            """
        }
        add_doc_res = self.client.index(self.text_index_name).add_documents([d1], device="cuda")
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc", device="cuda")
        self.assertEqual(1, len(search_res["hits"]))
        self.assertEqual(d1, self.strip_marqo_fields(search_res["hits"][0]))
        assert len(search_res["hits"][0]["_highlights"]) > 0
        assert ("title" in search_res["hits"][0]["_highlights"][0]) or ("content" in search_res["hits"][0]["_highlights"][0])

    def test_search_empty_index(self):
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc", device="cuda")
        assert len(search_res["hits"]) == 0

    def test_search_multi_docs(self):
        d1 = {
            "title": "Cool Document 1",
            "content": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "title": "Just Your Average Doc",
            "content": "this is a solid doc",
            "_id": "123456"
        }
        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], device="cuda")
        search_res = self.client.index(self.text_index_name).search(
            "this is a solid doc", device="cuda")
        assert d2 == self.strip_marqo_fields(search_res['hits'][0], strip_id=False)
        assert search_res['hits'][0]['_highlights'][0]["content"] == "this is a solid doc"

    def test_select_lexical(self):
        d1 = {
            "title": "Very heavy, dense metallic lead.",
            "content": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "title": "The captain bravely lead her followers into battle."
                     " She directed her soldiers to and fro.",
            "content": "this is a solid doc",
            "_id": "123456"
        }
        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], device="cuda")

        # Ensure that vector search works
        search_res = self.client.index(self.text_index_name).search(
            "Examples of leadership", search_method=enums.SearchMethods.TENSOR, device="cuda")
        assert d2 == self.strip_marqo_fields(search_res["hits"][0], strip_id=False)
        assert search_res["hits"][0]['_highlights'][0]["title"].startswith("The captain bravely lead her followers")

        # try it with lexical search:
        #    can't find the above with synonym
        assert len(self.client.index(self.text_index_name).search(
            "Examples of leadership", device="cuda", search_method=marqo.SearchMethods.LEXICAL)["hits"]) == 0
        #    but can look for a word
        assert self.client.index(self.text_index_name).search(
            "captain", device="cuda", search_method=marqo.SearchMethods.LEXICAL)["hits"][0]["_id"] == "123456"

    def test_filter_string_and_searchable_attributes(self):
        docs = [
            {
                "_id": "0",  # content in field_a
                "field_a": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 0,
            },
            {
                "_id": "1",  # content in field_b
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 0,
            },
            {
                "_id": "2",  # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 1,
            },
            {
                "_id": "3",  # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 1,
            }
        ]
        res = self.client.index(self.filter_test_index_name, ).add_documents(docs, device="cuda")

        test_cases = [
            {  # filter string only (str)
                "query": "random content",
                "filter_string": "str_for_filtering:apple",
                "expected": ["0", "2"]
            },
            {  # filter string only (int)
                "query": "random content",
                "filter_string": "int_for_filtering:0",
                "expected": ["0", "1"]
            },
            {  # filter string only (str and int)
                "query": "random content",
                "filter_string": "str_for_filtering:banana AND int_for_filtering:1",
                "expected": ["3"]
            },
        ]

        for case in test_cases:
            query = case["query"]
            filter_string = case.get("filter_string", "")
            expected = case["expected"]

            with self.subTest(query=query, filter_string=filter_string, expected=expected):
                search_res = self.client.index(self.filter_test_index_name, ).search(
                    query,
                    filter_string=filter_string, device="cuda"
                )
                actual_ids = set([hit["_id"] for hit in search_res["hits"]])
                self.assertEqual(len(search_res["hits"]), len(expected),
                                 f"Failed count check for query '{query}' with filter '{filter_string}'.")
                self.assertEqual(actual_ids, set(expected),
                                 f"Failed ID match for query '{query}' with filter '{filter_string}'. Expected {expected}, got {actual_ids}.")

    def test_multi_queries(self):
        docs = [
            {
                "content": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {"content": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]

        self.client.index(index_name=self.image_index_name).add_documents(
            documents=docs
        )

        queries_expected_ordering = [
            ({"Nature photography": 2.0, "Artefact": -2}, ['realistic_hippo', 'artefact_hippo']),
            ({"Nature photography": -1.0, "Artefact": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png": -1.0,
              "blah": 1.0}, ['realistic_hippo', 'artefact_hippo']),
            ({"https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png": 2.0,
              "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": -1.0},
             ['artefact_hippo', 'realistic_hippo']),
        ]
        for query, expected_ordering in queries_expected_ordering:
            res = self.client.index(index_name=self.image_index_name).search(
                q=query,
                search_method="TENSOR", device="cuda")
            # the poodle doc should be lower ranked than the irrelevant doc
            for hit_position, _ in enumerate(res['hits']):
                self.assertEqual(expected_ordering[hit_position], res['hits'][hit_position]['_id'])

    def test_custom_search_results(self):
        self.client.index(index_name=self.image_index_name).add_documents(
            [
                {
                    "title": "A comparison of the best pets",
                    "content": "Animals",
                    "_id": "d1"
                },
                {
                    "title": "The history of dogs",
                    "content": "A history of household pets",
                    "_id": "d2"
                }
            ]
        )

        query = {
            "What are the best pets": 1
        }
        context = {"tensor": [
            {"vector": [1, ] * 512, "weight": 0},
            {"vector": [2, ] * 512, "weight": 0}]
        }

        original_res = self.client.index(self.image_index_name).search(q=query, device="cuda")
        custom_res = self.client.index(self.image_index_name).search(q=query, context=context, device="cuda")
        original_score = original_res["hits"][0]["_score"]
        custom_score = custom_res["hits"][0]["_score"]
        self.assertEqual(custom_score, original_score)


@pytest.mark.cuda_test

class TestCudaUnstructuredSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        try:
            cls.delete_indexes(["api_test_unstructured_index", "api_test_unstructured_image_index"])
        except Exception:
            pass

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')
        cls.image_index_name = "api_test_unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.image_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

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

    def test_search_single_doc(self):
        """Searches an index of a single doc.
        Checks the basic functionality and response structure"""
        d1 = {
            "Title": "This is a title about some doc. ",
            "Description": """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
            The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
            """
        }
        add_doc_res = self.client.index(self.text_index_name).add_documents([d1],
                                                                            tensor_fields=["Title", "Description"],
                                                                            device="cuda")
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc", device="cuda")
        assert len(search_res["hits"]) == 1
        assert self.strip_marqo_fields(search_res["hits"][0]) == d1
        assert len(search_res["hits"][0]["_highlights"]) > 0
        assert ("Title" in search_res["hits"][0]["_highlights"][0]) or (
                    "Description" in search_res["hits"][0]["_highlights"][0])

    def test_search_empty_index(self):
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc", device="cuda")
        assert len(search_res["hits"]) == 0

    def test_search_multi_docs(self):
        d1 = {
            "doc_title": "Cool Document 1",
            "field_1": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "doc_title": "Just Your Average Doc",
            "field_X": "this is a solid doc",
            "_id": "123456"
        }
        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], tensor_fields=["doc_title", "field_1", "field_X"], device="cuda")
        search_res = self.client.index(self.text_index_name).search(
            "this is a solid doc")
        assert d2 == self.strip_marqo_fields(search_res['hits'][0], strip_id=False)
        assert search_res['hits'][0]['_highlights'][0]["field_X"] == "this is a solid doc"

    def test_select_lexical(self):
        d1 = {
            "doc_title": "Very heavy, dense metallic lead.",
            "field_1": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "doc_title": "The captain bravely lead her followers into battle."
                         " She directed her soldiers to and fro.",
            "field_X": "this is a solid doc",
            "_id": "123456"
        }
        res = self.client.index(self.text_index_name).add_documents([
            d1, d2
        ], tensor_fields=["doc_title", "field_1", "field_X"])

        # Ensure that vector search works
        search_res = self.client.index(self.text_index_name).search(
            "Examples of leadership", search_method=enums.SearchMethods.TENSOR, device="cuda")
        assert d2 == self.strip_marqo_fields(search_res["hits"][0], strip_id=False)
        assert search_res["hits"][0]['_highlights'][0]["doc_title"].startswith("The captain bravely lead her followers")

        # try it with lexical search:
        #    can't find the above with synonym
        assert len(self.client.index(self.text_index_name).search(
            "Examples of leadership", device="cuda", search_method=marqo.SearchMethods.LEXICAL)["hits"]) == 0
        #    but can look for a word
        assert self.client.index(self.text_index_name).search(
            "captain", device="cuda", search_method=marqo.SearchMethods.LEXICAL)["hits"][0]["_id"] == "123456"

    def test_filter_string_and_searchable_attributes(self):
        docs = [
            {
                "_id": "0",  # content in field_a
                "field_a": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 0,
            },
            {
                "_id": "1",  # content in field_b
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 0,
            },
            {
                "_id": "2",  # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 1,
            },
            {
                "_id": "3",  # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 1,
            }
        ]
        res = self.client.index(self.text_index_name).add_documents(docs,
                                                                    tensor_fields=["field_a", "field_b"], device="cuda")

        test_cases = [
            {  # filter string only (str)
                "query": "random content",
                "filter_string": "str_for_filtering:apple",
                "expected": ["0", "2"]
            },
            {  # filter string only (int)
                "query": "random content",
                "filter_string": "int_for_filtering:0",
                "expected": ["0", "1"]
            },
            {  # filter string only (str and int)
                "query": "random content",
                "filter_string": "str_for_filtering:banana AND int_for_filtering:1",
                "expected": ["3"]
            },
        ]

        for case in test_cases:
            query = case["query"]
            filter_string = case.get("filter_string", "")
            expected = case["expected"]

            with self.subTest(query=query, filter_string=filter_string, expected=expected):
                search_res = self.client.index(self.text_index_name).search(
                    query,
                    device = "cuda",
                    filter_string=filter_string,
                )
                actual_ids = set([hit["_id"] for hit in search_res["hits"]])
                self.assertEqual(len(search_res["hits"]), len(expected),
                                 f"Failed count check for query '{query}' with filter '{filter_string}'.")
                self.assertEqual(actual_ids, set(expected),
                                 f"Failed ID match for query '{query}' with filter '{filter_string}'. Expected {expected}, got {actual_ids}.")

    def test_escaped_non_tensor_field(self):
        """We need to make sure non tensor field escaping works properly.

        We test to ensure Marqo doesn't match to the non tensor field
        """
        docs = [{
            "dont_tensorise_Me": "Dog",
            "tensorise_me": "quarterly earnings report"
        }]
        self.client.index(index_name=self.text_index_name).add_documents(
            docs, tensor_fields=["tensorise_me"], device="cuda"
        )
        search_res = self.client.index(index_name=self.text_index_name).search("Dog", device="cuda")
        assert list(search_res['hits'][0]['_highlights'][0].keys()) == ['tensorise_me']

    def test_multi_queries(self):
        docs = [
            {
                "loc a": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {"loc b": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            'index_defaults': {
                'model': "ViT-B/16",
                'treat_urls_and_pointers_as_images': True
            }
        }

        self.client.index(index_name=self.text_index_name).add_documents(
            documents=docs, tensor_fields=["loc a", "loc b"], device="cuda"
        )

        queries_expected_ordering = [
            ({"Nature photography": 2.0, "Artefact": -2}, ['realistic_hippo', 'artefact_hippo']),
            ({"Nature photography": -1.0, "Artefact": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png": -1.0,
              "blah": 1.0}, ['realistic_hippo', 'artefact_hippo']),
            ({"https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png": 2.0,
              "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": -1.0},
             ['artefact_hippo', 'realistic_hippo']),
        ]
        for query, expected_ordering in queries_expected_ordering:
            res = self.client.index(index_name=self.text_index_name).search(
                q=query,
                search_method="TENSOR", device="cuda")
            # the poodle doc should be lower ranked than the irrelevant doc
            for hit_position, _ in enumerate(res['hits']):
                assert res['hits'][hit_position]['_id'] == expected_ordering[hit_position]

    def test_custom_search_results(self):
        self.client.index(index_name=self.image_index_name).add_documents(
            [
                {
                    "Title": "A comparison of the best pets",
                    "Description": "Animals",
                    "_id": "d1"
                },
                {
                    "Title": "The history of dogs",
                    "Description": "A history of household pets",
                    "_id": "d2"
                }
            ], tensor_fields=["Title", "Description"], device="cuda"
        )

        query = {
            "What are the best pets": 1
        }
        context = {"tensor": [
            {"vector": [1, ] * 512, "weight": 0},
            {"vector": [2, ] * 512, "weight": 0}]
        }

        original_res = self.client.index(self.image_index_name).search(q=query, device="cuda")
        custom_res = self.client.index(self.image_index_name).search(q=query, context=context, device="cuda")
        original_score = original_res["hits"][0]["_score"]
        custom_score = custom_res["hits"][0]["_score"]
        self.assertEqual(custom_score, original_score)