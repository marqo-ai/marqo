import math
import os
import random
import unittest
from unittest import mock

import requests

from marqo.api.exceptions import (
    IllegalRequestedDocCount
)
from marqo.core.models.marqo_index import FieldType, FieldFeature, IndexType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search, utils
from marqo.tensor_search.enums import SearchMethod, EnvVars
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from tests.utils.transition import add_docs_caller


class TestPagination(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        index_request_structured = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(
                    name='desc',
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch]
                )
            ],
            tensor_fields=['title']
        )
        index_request_unstructured = cls.unstructured_marqo_index_request()

        cls.indexes = cls.create_indexes([
            index_request_structured,
            index_request_unstructured
        ])

        cls.index_structured = cls.indexes[0]
        cls.index_unstructured = cls.indexes[1]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_document, search, bulk_search need this env var
        # Ensure other os.environ patches in indiv tests do not erase this one.
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

    def test_pagination_single_field(self):
        num_docs = 400
        batch_size = 100

        for index in [self.index_structured, self.index_unstructured]:
            for _ in range(0, num_docs, batch_size):
                r = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(index_name=index.name,
                                                  docs=[{"title": 'my title', 'desc': 'my title'} for i in
                                                        range(batch_size)],
                                                  device="cpu",
                                                  tensor_fields=['title'] if index.type == IndexType.Unstructured
                                                  else None
                                                  )
                )
                self.assertFalse(r['errors'], "Errors in add documents call")

            for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
                full_search_results = tensor_search.search(
                    search_method=search_method,
                    config=self.config,
                    index_name=index.name,
                    text='my title',
                    result_count=400)

                for page_size in [5, 10, 100, 200]:
                    with self.subTest(f'Index: {index.type}, Search method: {search_method}, Page size: {page_size}'):
                        paginated_search_results = {"hits": []}

                        for page_num in range(math.ceil(num_docs / page_size)):
                            lim = page_size
                            off = page_num * page_size
                            page_res = tensor_search.search(
                                search_method=search_method,
                                config=self.config,
                                index_name=index.name,
                                text='my title',
                                result_count=lim, offset=off)

                            paginated_search_results["hits"].extend(page_res["hits"])

                        # Compare paginated to full results (length only for now)
                        self.assertEqual(len(full_search_results["hits"]), len(paginated_search_results["hits"]))

    def test_pagination_high_limit_offset(self):
        """
        Test pagination with max device limit and offset (1000 and 10,000)
        """
        num_docs = 12000
        batch_size = 100

        max_limit = 1000
        max_offset = 10000

        self.assertTrue(num_docs >= max_limit, "Test requires num_docs >= max_limit")

        original_read_var = utils.read_env_vars_and_defaults

        def read_var(var):
            if var == EnvVars.MARQO_MAX_RETRIEVABLE_DOCS:
                return num_docs
            return original_read_var(var)

        # Patch EnvVars.MARQO_MAX_RETRIEVABLE_DOCS so we can test max offset
        with mock.patch.object(utils, 'read_env_vars_and_defaults', new=read_var):
            for index in [self.index_structured, self.index_unstructured]:
                for _ in range(0, num_docs, batch_size):
                    r = tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(index_name=index.name,
                                                      docs=[{"title": 'my title', 'desc': 'my title'} for i in
                                                            range(batch_size)],
                                                      device="cpu",
                                                      tensor_fields=['title'] if index.type == IndexType.Unstructured
                                                      else None,
                                                      )
                    )
                    self.assertFalse(r['errors'], "Errors in add documents call")

                for search_method in [SearchMethod.TENSOR, SearchMethod.LEXICAL]:
                    offsets_covered = set()
                    for page_size in [max_limit]:
                        with self.subTest(
                                f'Index: {index.type}, Search method: {search_method}, Page size: {page_size}'):
                            paginated_search_results = {"hits": []}

                            pages = math.ceil((max_offset + max_limit) / page_size)
                            for page_num in range(pages):
                                lim = page_size
                                off = page_num * page_size
                                offsets_covered.add(off)
                                page_res = tensor_search.search(
                                    search_method=search_method,
                                    config=self.config,
                                    index_name=index.name,
                                    text='my title',
                                    result_count=lim, offset=off,
                                    # Approximate search retrieved ~8400 docs, under investigation
                                    approximate=False if search_method == SearchMethod.TENSOR else None
                                )

                                paginated_search_results["hits"].extend(page_res["hits"])

                            # Compare paginated to full results (length only for now)
                            expected_count = (pages - 1) * page_size + page_size
                            self.assertEqual(expected_count, len(paginated_search_results["hits"]))

                    self.assertTrue(max_offset in offsets_covered, "Max offset not covered. Check test parameters")

    def test_pagination_limt_exceeded_error(self):
        """
        Verify InvalidArgs error is raised when limit is exceeded
        """
        self.fail()

    def test_pagination_offset_exceeded_error(self):
        """
        Verify InvalidArgs error is raised when offset is exceeded
        """
        self.fail()

    @unittest.skip
    def test_pagination_multi_field(self):
        # Execute pagination with 3 fields
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"

        vocab = requests.get(vocab_source).text.splitlines()
        num_docs = 1000

        # Recreate index with random model
        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings={"index_defaults": {"model": "random"}})

        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[{"field_1": "a " + (" ".join(random.choices(population=vocab, k=5))),
                   "field_2": "a " + (" ".join(random.choices(population=vocab, k=5))),
                   "field_3": "a " + (" ".join(random.choices(population=vocab, k=5))),
                   "_id": str(i)
                   } for i in range(num_docs)
                  ], auto_refresh=False
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)

        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for doc_count in [1000]:
                # Query full results
                full_search_results = tensor_search.search(
                    search_method=search_method,
                    config=self.config,
                    index_name=self.index_name_1,
                    text='a',
                    result_count=doc_count)

                for page_size in [5, 10, 100, 1000]:
                    paginated_search_results = {"hits": []}

                    for page_num in range(math.ceil(num_docs / page_size)):
                        lim = page_size
                        off = page_num * page_size
                        page_res = tensor_search.search(
                            search_method=search_method,
                            config=self.config,
                            index_name=self.index_name_1,
                            text='a',
                            result_count=lim, offset=off)

                        paginated_search_results["hits"].extend(page_res["hits"])

                    # Compare paginated to full results (length only for now)
                    assert len(full_search_results["hits"]) == len(paginated_search_results["hits"])

                    # TODO: re-add this assert when KNN incosistency bug is fixed
                    # assert full_search_results["hits"] == paginated_search_results["hits"]

    @unittest.skip
    def test_pagination_break_limitations(self):
        # Negative offset
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for lim in [1, 10, 1000]:
                for off in [-1, -10, -1000]:
                    try:
                        tensor_search.search(text=" ",
                                             index_name=self.index_name_1,
                                             config=self.config,
                                             result_count=lim,
                                             offset=off,
                                             search_method=search_method)
                        raise AssertionError
                    except IllegalRequestedDocCount:
                        pass

        # Negative limit
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for lim in [0, -1, -10, -1000]:
                for off in [1, 10, 1000]:
                    try:
                        tensor_search.search(text=" ",
                                             index_name=self.index_name_1,
                                             config=self.config,
                                             result_count=lim,
                                             offset=off,
                                             search_method=search_method)
                        raise AssertionError
                    except IllegalRequestedDocCount:
                        pass

        # Going over 10,000 for offset + limit
        mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: "10000"}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
                try:
                    tensor_search.search(search_method=search_method,
                                         config=self.config, index_name=self.index_name_1, text=' ',
                                         result_count=10000,
                                         offset=1)
                    raise AssertionError
                except IllegalRequestedDocCount:
                    pass

            return True

        assert run()

    @unittest.skip
    def test_pagination_empty_searchable_attributes(self):
        # Result should be empty whether paginated or not.
        docs = [
            {
                "field_a": 0,
                "field_b": 0,
                "field_c": 0
            },
            {
                "field_a": 1,
                "field_b": 1,
                "field_c": 1
            }
        ]

        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=False
        )

        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="some text",
            searchable_attributes=[], search_method="TENSOR", offset=1
        )
        assert res["hits"] == []

    @unittest.skip
    def test_lexical_search_pagination_empty_searchable_attribs(self):
        """Empty searchable attribs returns empty results (Even paginated)"""
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "extravagant very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "extravagant robodog is not a cat", "_id": "Jupyter_12"}

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d1, d2], device="cpu")
        )
        res = tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="extravagant",
            searchable_attributes=[], result_count=3, offset=1)
        assert res["hits"] == []
