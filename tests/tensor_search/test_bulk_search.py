from unittest import mock
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars, IndexSettingsField
from marqo.errors import (
    IndexNotFoundError, InvalidArgError, InvalidFieldNameError, IllegalRequestedDocCount, BadRequestError
)
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase


class TestBulkSearch(MarqoTestCase):

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

    def test_bulk_vector_text_search_searchable_attributes_non_existent(self):
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
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ")],
            result_count=10
        )
        assert len(search_res) == 1
        assert len(search_res[0]['hits']) == 2

    def test_bulk_vector_text_search_against_empty_index(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ")],
            result_count=10
        )
        assert len(search_res) == 0

    def test_bulk_vector_text_search_against_non_existent_index(self):
        try:
            tensor_search._bulk_vector_text_search(
                config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ")],
                result_count=10
            )
            raise AssertionError
        except IndexNotFoundError:
            pass

    def test_bulk_vector_text_search_long_query_string(self):
        query_text = """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
                    The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
                    """
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"_id": "12345", "Desc": "The Guardian is newspaper, read in the UK and other places around the world"},
                {"_id": "abc12334", "Title": "Grandma Jo's family recipe. ",
                 "Steps": "1. Cook meat. 2: Dice Onions. 3: Serve."},
            ], auto_refresh=True)
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=query_text)],

        )
        assert len(search_res) == 1
        assert len(search_res[0]['hits']) == 2

    def test_bulk_vector_text_search_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", searchableAttributes=["other field"])]),
            marqo_config=self.config,
        )
        assert len(search_res['result']) == 1
        assert search_res['result'][0]["hits"][0]["_id"] == "1234"
        for res in search_res['result'][0]["hits"]:
            assert list(res["_highlights"].keys()) == ["other field"]

    def test_bulk_vector_text_search_searchable_attributes_multiple(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(
                index=self.index_name_1, q="Exact match hehehe", searchableAttributes=["other field", "Cool Field 1"]
            )]),
            marqo_config=self.config,
        )
        assert len(search_res['result']) == 1
        ids = [search_res['result'][0]["hits"][0]["_id"], search_res['result'][0]["hits"][1]["_id"]]
        assert "1234" in ids
        assert "9000" in ids
        assert ids[0] != ids[1]
        for res in search_res['result'][0]["hits"]:
            assert "abc" not in res["_highlights"]

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
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(
                index=self.index_name_1, q=q, searchableAttributes=["other field", "Cool Field 1"], limit=50
            )]),
            marqo_config=self.config,
        )
        assert len(search_res["result"]) == 1
        print(search_res)
        assert "processingTimeMs" in search_res
        assert search_res["processingTimeMs"] > 0
        assert isinstance(search_res["processingTimeMs"], int)

        assert "query" in search_res['result'][0]
        assert search_res['result'][0]["query"] == q

        assert "limit" in search_res['result'][0]
        assert search_res['result'][0]["limit"] == 50

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
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=-1)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        try:
            # too small
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=1000000)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        try:
            # should not work with 0
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=0)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        # should work with 1:
        search_results = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=1)], ))
        assert len(search_results["result"]) >= 1
        assert len(search_results["result"][0]['hits']) >= 1

    def test_highlights_tensor(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)

        results = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some text", showHighlights=True)]), marqo_config=self.config
        )
        assert len(results["result"]) == 1
        tensor_highlights = results["result"][0]
        assert len(tensor_highlights["hits"]) == 2
        for hit in tensor_highlights["hits"]:
            assert "_highlights" in hit

        results = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some text", showHighlights=False)]), marqo_config=self.config
        )
        assert len(results["result"]) == 1
        tensor_no_highlights = results["result"][0]
        assert len(tensor_no_highlights["hits"]) == 2
        for hit in tensor_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_search_vector_int_field(self):
        """doesn't error out if there is a random int field"""
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_int": 144},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "my_int": 88},
            ], auto_refresh=True)

        results = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="88")],
            config=self.config)
        assert len(results) == 1
        s_res = results[0]
        assert len(s_res["hits"]) > 0

    def test_filtering_list_case_tensor(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "_id": "1235",  "my_list": ["tag1", "tag2 some"]}
            ], auto_refresh=True, non_tensor_fields=["my_list"])

        res_exists = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag1")],
            config=self.config)
        res_not_exists = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag55")],
            config=self.config)
        res_other = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_string:b")],
            config=self.config)

        # strings in lists are converted into keyword, which aren't filterable on a token basis.
        # Because the list member is "tag2 some" we can only exact match (incl. the space).
        # "tag2" by itself doesn't work, only "(tag2 some)"
        res_should_only_match_keyword_bad = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag2")],
            config=self.config)
        res_should_only_match_keyword_good = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:(tag2 some)")],
            config=self.config)
        assert res_exists[0]["hits"][0]["_id"] == "1235"
        assert res_exists[0]["hits"][0]["_highlights"] == {"abc": "some text"}
        assert len(res_exists[0]["hits"]) == 1

        assert len(res_not_exists[0]["hits"]) == 0

        assert res_other[0]["hits"][0]["_id"] == "5678"
        assert len(res_other[0]["hits"]) == 1

        assert len(res_should_only_match_keyword_bad[0]["hits"]) == 0
        assert len(res_should_only_match_keyword_good[0]["hits"]) == 1

    def test_filtering_list_case_image(self):
        settings = {"index_defaults": {"treat_urls_and_pointers_as_images": True, "model": "ViT-B/32"}}
        tensor_search.create_vector_index(index_name=self.index_name_1, index_settings=settings, config=self.config)
        hippo_img = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"img": hippo_img, "abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"img": hippo_img, "abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"img": hippo_img, "abc": "some text", "_id": "1235", "my_list": ["tag1", "tag2 some"]}
            ], auto_refresh=True, non_tensor_fields=["my_list"])

        res_img_2_imgs = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q=hippo_img, filter="my_list:tag1")],
            config=self.config
        )
        assert len(res_img_2_imgs) == 1
        res_img_2_img = res_img_2_imgs[0]
        assert res_img_2_img["hits"][0]["_id"] == "1235"
        assert len(res_img_2_img["hits"]) == 1
        assert res_img_2_img["hits"][0]["_highlights"] == {"img": hippo_img}

        res_img_2_img_none = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q=hippo_img, filter="my_list:not_exist")],
            config=self.config
        )
        assert len(res_img_2_img_none) == 1
        assert len(res_img_2_img_none[0]["hits"]) == 0

        res_txt_2_imgs = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter="my_list:tag1")],
            config=self.config
        )
        assert len(res_txt_2_imgs) == 1
        res_txt_2_img = res_txt_2_imgs[0]
        assert res_txt_2_img["hits"][0]["_id"] == "1235"
        assert res_txt_2_img["hits"][0]["_highlights"] == {"abc": "some text"}
        assert len(res_txt_2_img["hits"]) == 1

        res_txt_2_imgs = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter="my_list:not_exist")],
            config=self.config
        )
        assert len(res_txt_2_imgs) == 1
        assert len(res_txt_2_imgs[0]["hits"]) == 0

    def test_filtering(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True}
            ], auto_refresh=True)

        filter_strings = ["my_string:c", "an_int:2", "my_string:b", "my_int_something:5", "an_int:[5 TO 30]", "an_int:[0 TO 30]", "my_bool:true", "an_int:[0 TO 30] OR my_bool:true", "(an_int:[0 TO 30] and an_int:2) AND abc:(some text)"]
        expected_ids = [[], ["1234"], ["5678"], [],[], ["1234"], ["1233"], ["1233", "1234"], ["1234"]]
        for i in range(len(filter_strings)):
            result = tensor_search._bulk_vector_text_search(
                queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter=filter_strings[i])],
                config=self.config
            )
            assert len(result) == 1
            assert len(result[0]["hits"]) == len(expected_ids[i])
            if len(expected_ids[i]) > 0:
                for j in range(len(result[0]["hits"])):
                    assert result[0]["hits"][j]["_id"] in expected_ids[i]

    def test_filter_spaced_fields(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True},
                {"abc": "some text", "Floaty Field": 0.548, "_id": "344", "my_bool": True},
            ], auto_refresh=True)

        filter_to_ids = {
            "other\ field:baaadd": ["5678"],
            "other\ field:(Close match hehehe)": ['1234', '1233'],
            "(Floaty\ Field:[0 TO 1]) AND (abc:(some text))": ["344"]
        }
        for f, expected in filter_to_ids.items():
            result = tensor_search._bulk_vector_text_search(
                queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter=f)],
                config=self.config
            )
            assert len(result) == 1
            assert len(result[0]["hits"]) == len(expected)
            if len(expected) > 0:
                for j in range(len(result[0]["hits"])):
                    assert result[0]["hits"][j]["_id"] in expected

    def test_set_device(self):
        """calling search with a specified device overrides device defined in config"""
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some")]), device="cuda:123")
            return True

        assert run()
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:123"
        assert mock_config.search_device == "cpu"

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
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(index=self.index_name_1, q=str(to_search)
                 )])
            )
            assert "hits" in results["result"][0]

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
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(index=self.index_name_1, q=str(to_search), filter=f"{field}:{to_search}")])
            )
            assert "hits" in results["result"][0]
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1, q=str(to_search), search_method=SearchMethod.LEXICAL,
                        filter=f"{field}:{to_search}")])
            )
            assert "hits" in results["result"][0]

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
        results = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q="Exact match hehehe",
                    attributesToRetrieve=["other field", "Cool Field 1"]
                )], ))

        assert len(results['result']) == 1
        search_res = results['result'][0]
        assert len(search_res["hits"]) == 3
        for res in search_res["hits"]:
            assert docs[res["_id"]]["other field"] == res["other field"]
            assert docs[res["_id"]]["Cool Field 1"] == res["Cool Field 1"]
            assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == \
                   {"other field", "Cool Field 1", "_id"}

    # def test_attributes_to_retrieve_empty(self):
    #     docs = {
    #         "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
    #                  "Cool Field 1": "res res res", "_id": "5678"},
    #         "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
    #                       "abc": "random text", "other field": "Close match hehehe"},
    #         "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
    #     }
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
    #     for method in ("LEXICAL", "TENSOR"):
    #         search_res = tensor_search.search(
    #             config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
    #             attributes_to_retrieve=[], search_method=method
    #         )
    #         assert len(search_res["hits"]) == 3
    #         for res in search_res["hits"]:
    #             assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == {"_id"}
    #
    # def test_attributes_to_retrieve_empty_index(self):
    #     tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
    #     assert 0 == tensor_search.get_stats(config=self.config, index_name=self.index_name_1)['numberOfDocuments']
    #     for to_retrieve in [[], ["some field name"], ["some field name", "wowowow field"]]:
    #         for method in ("LEXICAL", "TENSOR"):
    #             search_res = tensor_search.search(
    #                 config=self.config, index_name=self.index_name_1, text="Exact match hehehe",
    #                 attributes_to_retrieve=to_retrieve, search_method=method
    #             )
    #             assert len(search_res["hits"]) == 0
    #             assert search_res['query'] == "Exact match hehehe"
    #
    # def test_attributes_to_retrieve_non_existent(self):
    #     docs = {
    #         "5678": {"abc": "Exact a match hehehe", "other field": "baaadd",
    #                  "Cool Field 1": "res res res", "_id": "5678"},
    #         "rgjknrgnj": {"Cool Field 1": "somewhata  match", "_id": "rgjknrgnj",
    #                       "abc": "random a text", "other field": "Close match hehehe"},
    #         "9000": {"Cool Field 1": "somewhat a match", "_id": "9000", "other field": "weewowow"}
    #     }
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
    #     for to_retrieve in [[], ["non existing field name"], ["other field", "non existing field name"]]:
    #         for method in ("TENSOR", "LEXICAL"):
    #             search_res = tensor_search.search(
    #                 config=self.config, index_name=self.index_name_1, text=" a ",
    #                 attributes_to_retrieve=to_retrieve, search_method=method
    #             )
    #             assert len(search_res["hits"]) == 3
    #             for res in search_res["hits"]:
    #                 assert "non existing field name" not in res
    #                 assert set(k for k in res.keys()
    #                            if k not in TensorField.__dict__.values() and k != "_id"
    #                            ).issubset(to_retrieve)
    #
    # def test_attributes_to_retrieve_and_searchable_attribs(self):
    #     docs = {
    #         "i_1": {"field_1": "a", "other field": "baaadd",
    #                 "Cool Field 1": "res res res", "_id": "i_1"},
    #         "i_2": {"field_1": "a", "_id": "i_2",
    #                 "field_2": "a", "other field": "Close match hehehe"},
    #         "i_3": {"field_1": " a ", "_id": "i_3", "field_2": "a",
    #                 "field_3": "a "}
    #     }
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
    #     for to_retrieve, to_search, expected_ids, expected_fields in [
    #         (["field_1"], ["field_3"], ["i_3"], ["field_1"]),
    #         (["field_3"], ["field_1"], ["i_1", "i_2", "i_3"], ["field_3"]),
    #         (["field_1", "field_2"], ["field_2", "field_3"], ["i_2", "i_3"], ["field_1", "field_2"]),
    #     ]:
    #         for method in ("TENSOR", "LEXICAL"):
    #             search_res = tensor_search.search(
    #                 config=self.config, index_name=self.index_name_1, text="a",
    #                 attributes_to_retrieve=to_retrieve, search_method=method,
    #                 searchable_attributes=to_search
    #             )
    #             assert len(search_res["hits"]) == len(expected_ids)
    #             assert set(expected_ids) == {h['_id'] for h in search_res["hits"]}
    #             for res in search_res["hits"]:
    #                 relevant_fields = set(expected_fields).intersection(set(docs[res["_id"]].keys()))
    #                 assert set(k for k in res.keys()
    #                            if k not in TensorField.__dict__.values() and k != "_id"
    #                            ) == relevant_fields
    #
    # def test_attributes_to_retrieve_non_list(self):
    #     tensor_search.add_documents(config=self.config, index_name=self.index_name_1,
    #                                 docs=[{"cool field 111": "this is some content"}],
    #                                 auto_refresh=True)
    #     for method in ("TENSOR", "LEXICAL"):
    #         for bad_attr in ["jknjhc", "", dict(), 1234, 1.245]:
    #             try:
    #                 tensor_search.search(
    #                     config=self.config, index_name=self.index_name_1, text="a",
    #                     attributes_to_retrieve=bad_attr, search_method=method,
    #                 )
    #                 raise AssertionError
    #             except (InvalidArgError, InvalidFieldNameError):
    #                 pass
    #
    # def test_limit_results(self):
    #     """"""
    #     vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    #
    #     vocab = requests.get(vocab_source).text.splitlines()
    #
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25)))}
    #               for _ in range(2000)], auto_refresh=False
    #     )
    #     tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         for max_doc in [0, 1, 2, 5, 10, 100, 1000]:
    #
    #             _environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}
    #
    #             @mock.patch("os.environ", mock_environ)
    #             def run():
    #                 half_search = tensor_search.search(search_method=search_method,
    #                     config=self.config, index_name=self.index_name_1, text='a', result_count=max_doc//2)
    #                 assert half_search['limit'] == max_doc//2
    #                 assert len(half_search['hits']) == max_doc//2
    #                 limit_search = tensor_search.search(search_method=search_method,
    #                     config=self.config, index_name=self.index_name_1, text='a', result_count=max_doc)
    #                 assert limit_search['limit'] == max_doc
    #                 assert len(limit_search['hits']) == max_doc
    #                 try:
    #                     oversized_search = tensor_search.search(search_method=search_method,
    #                         config=self.config, index_name=self.index_name_1, text='a', result_count=max_doc + 1)
    #                 except IllegalRequestedDocCount:
    #                     pass
    #                 try:
    #                     very_oversized_search = tensor_search.search(search_method=search_method,
    #                         config=self.config, index_name=self.index_name_1, text='a', result_count=(max_doc + 1) * 2)
    #                 except IllegalRequestedDocCount:
    #                     pass
    #                 return True
    #         assert run()
    #
    # def test_limit_results_none(self):
    #     """if env var isn't set or is None"""
    #     vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    #
    #     vocab = requests.get(vocab_source).text.splitlines()
    #
    #     tensor_search.add_documents_orchestrator(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25)))}
    #               for _ in range(700)], auto_refresh=False, processes=4, batch_size=50
    #     )
    #     tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    #
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         for mock_environ in [dict(), {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: None},
    #                              {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: ''}]:
    #             @mock.patch("os.environ", mock_environ)
    #             def run():
    #                 lim = 500
    #                 half_search = tensor_search.search(
    #                     search_method=search_method,
    #                     config=self.config, index_name=self.index_name_1, text='a', result_count=lim)
    #                 assert half_search['limit'] == lim
    #                 assert len(half_search['hits']) == lim
    #                 return True
    #
    #             assert run()
    #
    # def test_pagination_single_field(self):
    #     vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    #
    #     vocab = requests.get(vocab_source).text.splitlines()
    #     num_docs = 2000
    #
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25))),
    #                 "_id": str(i)
    #                 }
    #               for i in range(num_docs)], auto_refresh=False
    #     )
    #     tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    #
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         for doc_count in [2000]:
    #             # Query full results
    #             full_search_results = tensor_search.search(
    #                                     search_method=search_method,
    #                                     config=self.config,
    #                                     index_name=self.index_name_1,
    #                                     text='a',
    #                                     result_count=doc_count)
    #
    #             for page_size in [5, 10, 100, 1000, 2000]:
    #                 paginated_search_results = {"hits": []}
    #
    #                 for page_num in range(math.ceil(num_docs / page_size)):
    #                     lim = page_size
    #                     off = page_num * page_size
    #                     page_res = tensor_search.search(
    #                                     search_method=search_method,
    #                                     config=self.config,
    #                                     index_name=self.index_name_1,
    #                                     text='a',
    #                                     result_count=lim, offset=off)
    #
    #                     paginated_search_results["hits"].extend(page_res["hits"])
    #
    #                 # Compare paginated to full results (length only for now)
    #                 assert len(full_search_results["hits"]) == len(paginated_search_results["hits"])
    #
    #                 # TODO: re-add this assert when KNN incosistency bug is fixed
    #                 # assert full_search_results["hits"] == paginated_search_results["hits"]
    #
    # def test_pagination_break_limitations(self):
    #     tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
    #     # Negative offset
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         for lim in [1, 10, 1000]:
    #             for off in [-1, -10, -1000]:
    #                 try:
    #                     tensor_search.search(text=" ",
    #                                         index_name=self.index_name_1,
    #                                         config=self.config,
    #                                         result_count=lim,
    #                                         offset=off,
    #                                         search_method=search_method)
    #                     raise AssertionError
    #                 except IllegalRequestedDocCount:
    #                     pass
    #
    #     # Negative limit
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         for lim in [0, -1, -10, -1000]:
    #             for off in [1, 10, 1000]:
    #                 try:
    #                     tensor_search.search(text=" ",
    #                                         index_name=self.index_name_1,
    #                                         config=self.config,
    #                                         result_count=lim,
    #                                         offset=off,
    #                                         search_method=search_method)
    #                     raise AssertionError
    #                 except IllegalRequestedDocCount:
    #                     pass
    #
    #     # Going over 10,000 for offset + limit
    #     mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: "10000"}
    #     @mock.patch("os.environ", mock_environ)
    #     def run():
    #         for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #             try:
    #                 tensor_search.search(search_method=search_method,
    #                                     config=self.config, index_name=self.index_name_1, text=' ',
    #                                     result_count=10000,
    #                                     offset=1)
    #                 raise AssertionError
    #             except IllegalRequestedDocCount:
    #                 pass
    #
    #         return True
    #
    #     assert run()
    #
    # def test_pagination_multi_field_error(self):
    #     # Try pagination with 0, 2, and 3 fields
    #     # To be removed when multi-field pagination is added.
    #     docs = [
    #         {
    #             "field_a": 0,
    #             "field_b": 0,
    #             "field_c": 0
    #         },
    #         {
    #             "field_a": 1,
    #             "field_b": 1,
    #             "field_c": 1
    #         }
    #     ]
    #
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=False
    #     )
    #
    #     tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    #
    #     for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
    #         try:
    #             tensor_search.search(text=" ",
    #                                 index_name=self.index_name_1,
    #                                 config=self.config,
    #                                 offset=1,
    #                                 searchable_attributes=["field_a", "field_b"],
    #                                 search_method=search_method)
    #             raise AssertionError
    #         except InvalidArgError:
    #             pass
    #
    #         try:
    #             tensor_search.search(text=" ",
    #                                 index_name=self.index_name_1,
    #                                 config=self.config,
    #                                 offset=1,
    #                                 search_method=search_method)
    #             raise AssertionError
    #         except InvalidArgError:
    #             pass
    #
    #         try:
    #             tensor_search.search(text=" ",
    #                                 index_name=self.index_name_1,
    #                                 config=self.config,
    #                                 offset=1,
    #                                 searchable_attributes=[],
    #                                 search_method=search_method)
    #             raise AssertionError
    #         except InvalidArgError:
    #             pass
    #
    # def test_image_search_highlights(self):
    #     """does the URL get returned as the highlight? (it should - because no rerankers are being used)"""
    #     settings = {
    #         "index_defaults": {
    #             "treat_urls_and_pointers_as_images": True,
    #             "model": "ViT-B/32",
    #         }}
    #     tensor_search.create_vector_index(
    #         index_name=self.index_name_1, index_settings=settings, config=self.config
    #     )
    #     url_1 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
    #     url_2 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
    #     docs = [
    #         {"_id": "123",
    #          "image_field": url_1,
    #          "text_field": "some words here"
    #          },
    #         {"_id": "789",
    #          "image_field": url_2},
    #     ]
    #     tensor_search.add_documents(
    #         config=self.config, auto_refresh=True, index_name=self.index_name_1, docs=docs
    #     )
    #     res = tensor_search.search(
    #         config=self.config, index_name=self.index_name_1, text="some text", result_count=3,
    #         searchable_attributes=['image_field']
    #     )
    #     assert len(res['hits']) == 2
    #     assert {hit['image_field'] for hit in res['hits']} == {url_2, url_1}
    #     assert {hit['_highlights']['image_field'] for hit in res['hits']} == {url_2, url_1}
    #
    # def test_multi_search(self):
    #     docs = [
    #         {"field_a": "Doberman, canines, golden retrievers are humanity's best friends",
    #          "_id": 'dog_doc'},
    #         {"field_a": "All things poodles! Poodles are great pets",
    #          "_id": 'poodle_doc'},
    #         {"field_a": "Construction and scaffolding equipment",
    #          "_id": 'irrelevant_doc'}
    #     ]
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=True
    #     )
    #     queries_expected_ordering = [
    #         ({"Dogs": 2.0, "Poodles": -2}, ['dog_doc', 'irrelevant_doc', 'poodle_doc']),
    #         ("dogs", ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
    #         ({"dogs": 1}, ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
    #         ({"Dogs": -2.0, "Poodles": 2}, ['poodle_doc', 'irrelevant_doc', 'dog_doc']),
    #     ]
    #     for query, expected_ordering in queries_expected_ordering:
    #         res = tensor_search.search(
    #             text=query,
    #             index_name=self.index_name_1,
    #             result_count=5,
    #             config=self.config,
    #             search_method=SearchMethod.TENSOR, )
    #
    #         # the poodle doc should be lower ranked than the irrelevant doc
    #         for hit_position, _ in enumerate(res['hits']):
    #             assert res['hits'][hit_position]['_id'] == expected_ordering[hit_position]
    #
    # def test_multi_search_images(self):
    #     docs = [
    #         {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    #          "_id": 'realistic_hippo'},
    #         {"loc b": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
    #          "_id": 'artefact_hippo'}
    #     ]
    #     image_index_config = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: True
    #         }
    #     }
    #     tensor_search.create_vector_index(
    #         config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=True
    #     )
    #     queries_expected_ordering = [
    #         ({"Nature photography": 2.0, "Artefact": -2}, ['realistic_hippo', 'artefact_hippo']),
    #         ({"Nature photography": -1.0, "Artefact": 1.0}, ['artefact_hippo', 'realistic_hippo']),
    #         ({"Nature photography": -1.5, "Artefact": 1.0, "hippo": 1.0}, ['artefact_hippo', 'realistic_hippo']),
    #         ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": -1.0,
    #           "blah": 1.0}, ['realistic_hippo', 'artefact_hippo']),
    #         ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
    #           "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0},
    #          ['artefact_hippo', 'realistic_hippo']),
    #         ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
    #           "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
    #           "artefact": 1.0, "photo realistic": -1,
    #           },
    #          ['artefact_hippo', 'realistic_hippo']),
    #     ]
    #     for query, expected_ordering in queries_expected_ordering:
    #         res = tensor_search.search(
    #             text=query,
    #             index_name=self.index_name_1,
    #             result_count=5,
    #             config=self.config,
    #             search_method=SearchMethod.TENSOR)
    #         # the poodle doc should be lower ranked than the irrelevant doc
    #         for hit_position, _ in enumerate(res['hits']):
    #             assert res['hits'][hit_position]['_id'] == expected_ordering[hit_position]
    #
    # def test_multi_search_images_edge_cases(self):
    #     docs = [
    #         {"loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    #          "_id": 'realistic_hippo'},
    #         {"field_a": "Some text about a weird forest",
    #          "_id": 'artefact_hippo'}
    #     ]
    #     image_index_config = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: True
    #         }
    #     }
    #     tensor_search.create_vector_index(
    #         config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=True
    #     )
    #     invalid_queries = [{}, None, {123: 123}, {'123': None},
    #                        {"https://marqo_not_real.com/image_1.png": 3}, set()]
    #     for q in invalid_queries:
    #         try:
    #             tensor_search.search(
    #                 text=q,
    #                 index_name=self.index_name_1,
    #                 result_count=5,
    #                 config=self.config,
    #                 search_method=SearchMethod.TENSOR)
    #             raise AssertionError
    #         except (InvalidArgError, BadRequestError) as e:
    #             pass
    #
    # def test_multi_search_images_ok_edge_cases(self):
    #     docs = [
    #         {"loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    #          "_id": 'realistic_hippo'},
    #         {"field_a": "Some text about a weird forest",
    #          "_id": 'artefact_hippo'}
    #     ]
    #     image_index_config = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: True
    #         }
    #     }
    #     tensor_search.create_vector_index(
    #         config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=True
    #     )
    #     alright_queries = [{"v ": 1.2}, {"d ": 0}, {"vf": -1}]
    #     for q in alright_queries:
    #         tensor_search.search(
    #             text=q,
    #             index_name=self.index_name_1,
    #             result_count=5,
    #             config=self.config,
    #             search_method=SearchMethod.TENSOR)
    #
    # def test_image_search(self):
    #     """This test is to ensure image search works as expected
    #     The code paths for image and search have diverged quite a bit
    #     """
    #     hippo_image = (
    #         'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
    #     )
    #     doc_dict = {
    #         'realistic_hippo': {"loc": hippo_image,
    #          "_id": 'realistic_hippo'},
    #         'artefact_hippo': {"field_a": "Some text about a weird forest",
    #          "_id": 'artefact_hippo'}
    #     }
    #     docs = list(doc_dict.values())
    #     image_index_config = {
    #         IndexSettingsField.index_defaults: {
    #             IndexSettingsField.model: "ViT-B/16",
    #             IndexSettingsField.treat_urls_and_pointers_as_images: True
    #         }
    #     }
    #     tensor_search.create_vector_index(
    #         config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
    #     tensor_search.add_documents(
    #         config=self.config, index_name=self.index_name_1,
    #         docs=docs, auto_refresh=True
    #     )
    #     res = tensor_search.search(
    #         text=hippo_image,
    #         index_name=self.index_name_1,
    #         result_count=5,
    #         config=self.config,
    #         search_method=SearchMethod.TENSOR)
    #     assert len(res['hits']) == 2
    #     for hit in res['hits']:
    #         original_doc = doc_dict[hit['_id']]
    #         assert len(hit['_highlights']) == 1
    #         highlight_field = list(hit['_highlights'].keys())[0]
    #         assert highlight_field in original_doc
    #         assert hit[highlight_field] == original_doc[highlight_field]
