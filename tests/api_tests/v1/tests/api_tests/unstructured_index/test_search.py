import copy
import uuid
from unittest import mock

import marqo
from marqo import enums
from marqo.client import Client
from marqo.enums import SearchMethods
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestUnstructuredSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        try:
            cls.delete_indexes(["api_test_unstructured_index", "api_test_unstructured_image_index"])
        except Exception:
            pass

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')
        cls.text_index_2_name = "api_test_unstructured_index_2_" + str(uuid.uuid4()).replace('-', '')
        cls.image_index_name = "api_test_unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.text_index_2_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name, cls.text_index_2_name, cls.image_index_name]

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
            "title": "This is a title about some doc. ",
            "description": """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
            The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
            """
        }
        add_doc_res = self.client.index(self.text_index_name).add_documents([d1], tensor_fields=["title", "description"])
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc")
        assert len(search_res["hits"]) == 1
        assert self.strip_marqo_fields(search_res["hits"][0]) == d1
        assert len(search_res["hits"][0]["_highlights"]) > 0
        assert ("title" in search_res["hits"][0]["_highlights"][0]) or ("description" in search_res["hits"][0]["_highlights"][0])

    def test_search_empty_index(self):
        search_res = self.client.index(self.text_index_name).search(
            "title about some doc")
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
        ], tensor_fields=["doc_title", "field_1", "field_X"])
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
            "Examples of leadership", search_method=enums.SearchMethods.TENSOR)
        assert d2 == self.strip_marqo_fields(search_res["hits"][0], strip_id=False)
        assert search_res["hits"][0]['_highlights'][0]["doc_title"].startswith("The captain bravely lead her followers")

        # try it with lexical search:
        #    can't find the above with synonym
        assert len(self.client.index(self.text_index_name).search(
            "Examples of leadership", search_method=marqo.SearchMethods.LEXICAL)["hits"]) == 0
        #    but can look for a word
        assert self.client.index(self.text_index_name).search(
            "captain", search_method=marqo.SearchMethods.LEXICAL)["hits"][0]["_id"] == "123456"
        
    def test_search_with_no_device(self):
        """use default as defined in config unless overridden"""
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()
        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.text_index_name).search(q="my search term")
            temp_client.index(self.text_index_name).search(q="my search term", device="cuda:2")
            return True
        assert run()
        # no device in path when device is not set
        args, kwargs0 = mock__post.call_args_list[0]
        assert "device" not in kwargs0["path"]
        # device in path if it is set
        args, kwargs1 = mock__post.call_args_list[1]
        assert "device=cuda2" in kwargs1["path"]

    def test_filter_string_and_searchable_attributes(self):
        docs = [
            {
                "_id": "0",                     # content in field_a
                "field_a": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 0,
            },
            {
                "_id": "1",                     # content in field_b
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 0,
            },
            {
                "_id": "2",                     # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "apple",
                "int_for_filtering": 1,
            },
            {
                "_id": "3",                     # content in both
                "field_a": "random content",
                "field_b": "random content",
                "str_for_filtering": "banana",
                "int_for_filtering": 1,
            }
        ]
        res = self.client.index(self.text_index_name).add_documents(docs, tensor_fields=["field_a", "field_b"])

        test_cases = [
            {   # filter string only (str)
                "query": "random content",
                "filter_string": "str_for_filtering:apple",
                "expected": ["0", "2"]
            },
            {   # filter string only (int)
                "query": "random content",
                "filter_string": "int_for_filtering:0",
                "expected": ["0", "1"]
            },
            {   # filter string only (str and int)
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
            docs, tensor_fields=["tensorise_me"]
        )
        search_res = self.client.index(index_name=self.text_index_name).search("Dog")
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
            documents=docs, tensor_fields=["loc a", "loc b"]
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
                search_method="TENSOR")
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
            ], tensor_fields=["Title", "Description"]
        )
        
        query = {
            "What are the best pets": 1
        }
        context = {"tensor": [
            {"vector": [1, ] * 512, "weight": 0},
            {"vector": [2, ] * 512, "weight": 0}]
        }

        original_res = self.client.index(self.image_index_name).search(q=query)
        custom_res = self.client.index(self.image_index_name).search(q=query, context=context)
        original_score = original_res["hits"][0]["_score"]
        custom_score = custom_res["hits"][0]["_score"]
        self.assertEqual(custom_score, original_score)

    def test_filter_on_large_integer_and_float(self):
        valid_documents = [
            {'long_field_1': 1, '_id': '0', "field_a": "some text"},  # small positive integer
            {'long_field_1': -1, '_id': '1', "field_a": "some text"},  # small negative integer
            # large positive integer that can't be handled by int
            {'long_field_1': 1002321422323, '_id': '2', "field_a": "some text"},
            # large negative integer that can't be handled by int
            {'long_field_1': -9232172132345, '_id': '3', "field_a": "some text"},
            # large positive integer mathematical expression
            {'double_field_1': 10000000000.0, '_id': '4', "field_a": "some text"},
            # large negative integer mathematical expression
            {'double_field_1': -1000000000000.0, '_id': '5', "field_a": "some text"},
            # large positive float
            {'double_field_1': 10000000000.12325, '_id': '6', "field_a": "some text"},
            # large negative float
            {'double_field_1': -9999999999.87675, '_id': '7', "field_a": "some text"},
        ]

        self.client.index(self.text_index_name).add_documents(
                documents=valid_documents, tensor_fields=[]
        )

        self.assertEqual(len(valid_documents),
                         self.client.index(self.text_index_name).get_stats()["numberOfDocuments"])

        for document in valid_documents:
            for search_method in [SearchMethods.LEXICAL, SearchMethods.TENSOR]:
                numeric_field = list(document.keys())[0]
                numeric_value = document[numeric_field] if isinstance(document[numeric_field], (int, float)) \
                    else document[numeric_field][0]
                filter_string = f"{numeric_field}:{numeric_value}"
                expected_document_ids = document["_id"]
                with self.subTest(f"filter_string = {filter_string}, "
                                  f"expected_document_ids = {expected_document_ids}, "
                                  f"search_method = {search_method}"):
                    res = self.client.index(self.text_index_name).search(
                        q="some text",
                        filter_string=filter_string, search_method=SearchMethods.LEXICAL
                    )
                    self.assertEqual(1, len(res["hits"]))
                    self.assertEqual(expected_document_ids, res["hits"][0]["_id"])

    def test_filter_on_id(self):
        """A test to check that filtering on _id works"""
        docs = [
            {
                "title": "Cool Document 1",
                "content": "some extra info",
                "_id": "e197e580-039"
            },
            {
                "title": "Just Your Average Doc",
                "content": "this is a solid doc",
                "_id": "123456"
            }
        ]
        res = self.client.index(self.text_index_name).add_documents(docs, tensor_fields=["title", "content"])
        test_case = [
            ("_id:e197e580-039", ["e197e580-039"], "single _id filter"),
            ("_id:e197e580-039 OR _id:123456", ["e197e580-039", "123456"], "multiple _id filter with OR"),
            ("_id:e197e580-039 AND _id:123456", [], "multiple _id filter with AND"),
            ("_id:e197e580-039 AND title:(Cool Document 1)", ["e197e580-039"],
             "multiple _id filter with AND and title filter"),
        ]
        for search_method in ["TENSOR", "LEXICAL"]:
            for filter_string, expected, msg in test_case:
                with self.subTest(f"{search_method} - {msg}"):
                    search_res = self.client.index(self.text_index_name).search(q = "title", filter_string=filter_string)
                    actual_ids = set([hit["_id"] for hit in search_res["hits"]])
                    self.assertEqual(len(search_res["hits"]), len(expected),
                                     f"Failed count check for filter '{filter_string}'.")
                    self.assertEqual(actual_ids, set(expected), f"Failed ID match for filter '{filter_string}'")

    def test_searchable_attributes(self):
        docs_batch_1 = [
            {
                "title": "Cool Document 1",
                "content": "some extra info",
                "_id": "1"
            },
            {
                "title": "Just Your Average Doc",
                "content": "this is a solid doc",
                "_id": "2"
            }
        ]
        self.client.index(self.text_index_2_name).add_documents(docs_batch_1, tensor_fields=["title", "content"])

        docs_batch_2 = [
            {
                "desc": "Cool Document 2",
                "content": "some extra info",
                "_id": "3"
            },
            {
                "desc": "Just Your Average Doc 2",
                "content": "this is a solid doc",
                "_id": "4"
            }
        ]
        self.client.index(self.text_index_2_name).add_documents(docs_batch_2, tensor_fields=["desc", "content"])

        # Tensor search for title fields should only return the first 2 docs
        search_res = self.client.index(self.text_index_2_name).search(q="Cool", search_method=SearchMethods.TENSOR,
                                                                      searchable_attributes=["title"])
        self.assertEqual(len(search_res["hits"]), 2)
        self.assertEqual(search_res["hits"][0]["_id"], "1")
        self.assertEqual(search_res["hits"][1]["_id"], "2")

        # Lexical search for desc field should only return the matching doc 3
        search_res = self.client.index(self.text_index_2_name).search(q="Cool", search_method=SearchMethods.LEXICAL,
                                                                      searchable_attributes=["desc"])
        self.assertEqual(len(search_res["hits"]), 1)
        self.assertEqual(search_res["hits"][0]["_id"], "3")

        # Hybrid search on content fields should return matching docs from both batches
        search_res = self.client.index(self.text_index_2_name).search(
            q="Solid", search_method="HYBRID",
            hybrid_parameters={
                  "retrievalMethod": "disjunction",
                  "rankingMethod": "rrf",
                  "alpha": 0.5,
                  "searchableAttributesLexical": ["content"],
                  "searchableAttributesTensor": ["title"]
            }
        )
        # lexical returns 2 and 4, tensor returns 1 and 2, after rrf ranking, we return 3 results (limit defaults to 10 in Marqo)
        # 1 and 4 should have equal RRF score (interchangeable)
        self.assertEqual(len(search_res["hits"]), 3)
        self.assertEqual(search_res["hits"][0]["_id"], "2")
        self.assertIn(search_res["hits"][1]["_id"], ["1", "4"])
        self.assertIn(search_res["hits"][2]["_id"], ["1", "4"])

        # in batch 3, we reindex doc 1 but remove title as a tensor field
        docs_batch_3 = [
            {
                "title": "Cool Document 1",
                "content": "some extra info",
                "_id": "1"
            },
        ]
        self.client.index(self.text_index_2_name).add_documents(docs_batch_3, tensor_fields=["content"])
        # Now we should only able to see doc 2 in the result when tensor search on title
        search_res = self.client.index(self.text_index_2_name).search(q="Cool", search_method=SearchMethods.TENSOR,
                                                                      searchable_attributes=["title"])
        self.assertEqual(len(search_res["hits"]), 1)
        self.assertEqual(search_res["hits"][0]["_id"], "2")

        # But Lexical search on title can still find doc 1
        search_res = self.client.index(self.text_index_2_name).search(q="Cool", search_method=SearchMethods.LEXICAL,
                                                                      searchable_attributes=["title"])
        self.assertEqual(len(search_res["hits"]), 1)
        self.assertEqual(search_res["hits"][0]["_id"], "1")
