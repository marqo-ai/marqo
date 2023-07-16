import math
import os
import sys 
from tests.utils.transition import add_docs_caller
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from unittest import mock
from marqo.s2_inference.s2_inference import vectorise, get_model_properties_from_registry
import numpy as np
from marqo.tensor_search import utils
import typing
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars, IndexSettingsField, MlModel
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount, BadRequestError, InternalError
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase
import requests
import random

class TestPagination(MarqoTestCase):

    def setUp(self) -> None:
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.index_name_3 = "my-test-index-3"
        self._delete_test_indices()
        self._create_test_indices()

        # Any tests that call add_document, search, bulk_search need this env var
        # Ensure other os.environ patches in indiv tests do not erase this one.
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

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

    def _create_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_create = [self.index_name_1, self.index_name_2, self.index_name_3]
        else:
            ix_to_create = indices
        for ix_name in ix_to_create:
            tensor_search.create_vector_index(config=self.config, index_name=ix_name)
    
    def test_pagination_single_field(self):
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"

        vocab = requests.get(vocab_source).text.splitlines()
        num_docs = 1000
        
        # Recreate index with random model
        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings={"index_defaults": {"model": "random"}})

        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=10))),
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

    def test_pagination_multi_field(self):
        # Execute pagination with 3 fields
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"

        vocab = requests.get(vocab_source).text.splitlines()
        num_docs = 1000
        
        # Recreate index with random model
        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings={"index_defaults": {"model": "random"}})

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