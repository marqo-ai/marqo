import copy
import os
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
import functools
import json
import math
import pprint
from unittest import mock
from marqo.s2_inference import types
import PIL
import marqo.tensor_search.utils as marqo_utils
import numpy as np
import requests
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from marqo.tensor_search import enums
from marqo.errors import IndexNotFoundError, InvalidArgError, BadRequestError, InternalError
from marqo.tensor_search import tensor_search, index_meta_cache, backend
from tests.marqo_test import MarqoTestCase
import time
from marqo.tensor_search import add_docs
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity

class TestDefaultDevice(MarqoTestCase):

    """
        Assumptions:
        1. once CUDA is available on startup, it will always be available, or else Marqo is broken
        2. CPU is always available
        3. MARQO_BEST_AVAILABLE_DEVICE is set once on startup and never changed
        4. 
    """
    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"

        self.mock_bulk_vector_text_search_results = \
        [
            {'hits': [
                {
                    'abc': 'Exact match hehehe', 
                    'other field': 'baaadd', 
                    '_id': 'id1-first', 
                    '_highlights': {
                        'abc': 'Exact match hehehe'
                    }, 
                    '_score': 0.8317631
                }, 
                {
                    'abc': 'random text', 
                    'other field': 'Close match hehehe', 
                    '_id': 'id1-second', 
                    '_highlights': {
                        'other field': 'Close match hehehe'
                    }, 
                    '_score': 0.82157063
                }
            ]}, 
            {'hits': [
                {
                    'abc': 'Exact match hehehe', 
                    'other field': 'baaadd', 
                    '_id': 'id1-first', 
                    '_highlights': {
                        'abc': 'Exact match hehehe'
                    }, 
                    '_score': 0.83613795
                }, 
                {
                    'abc': 'random text', 
                    'other field': 'Close match hehehe', 
                    '_id': 'id1-second', 
                    '_highlights': {
                        'other field': 'Close match hehehe'
                    }, 
                    '_score': 0.82666266
                }
            ]}
        ]

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

    def tearDown(self) -> None:
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_add_docs_orchestrator_defaults_to_best_device(self):
        """
            when no device is set,
            add docs orchestrator should call add_documents / _batch_request
            with env var MARQO_BEST_AVAILABLE_DEVICE
        """
        test_cases = [
            ("cpu", {}, ["marqo.tensor_search.tensor_search.add_documents"]),   # normal
            ("cpu", {"batch_size": 2}, ["marqo.tensor_search.tensor_search._batch_request"]),    # server batched
            
            ("cuda", {}, ["marqo.tensor_search.tensor_search.add_documents"]),   # normal
            ("cuda", {"batch_size": 2}, ["marqo.tensor_search.tensor_search._batch_request"]),    # server batched
        ]
        for best_available_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Call orchestrator
                tensor_search.add_documents_orchestrator(
                    config=self.config,
                    add_docs_params=AddDocsParams(index_name=self.index_name_1, 
                                    docs=[{"Title": "blah"} for i in range(5)], 
                                    auto_refresh=True,
                                    # no device set, so should default to best
                                ),
                    **extra_params
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    if "add_docs_params" in mocked_method.call_args[1]:
                        assert mocked_method.call_args[1]["add_docs_params"].device == best_available_device
                    else:
                        assert mocked_method.call_args[1]["device"] == best_available_device
                
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()

                return True

            assert run()

    @mock.patch("os.environ", dict())
    def test_add_docs_orchestrator_fails_with_no_default(self):
        """
            If no best available device is set, this function should raise internal error.
        """
        self.assertNotIn("MARQO_BEST_AVAILABLE_DEVICE", os.environ)
        # Call orchestrator
        try:
            tensor_search.add_documents_orchestrator(
                config=self.config,
                add_docs_params=AddDocsParams(index_name=self.index_name_1, 
                                docs=[{"Title": "blah"} for i in range(5)], 
                                auto_refresh=True,
                                # no device set, so should default to best
                            ),
            )
            raise AssertionError
        except InternalError:
            pass
        
    def test_add_docs_orchestrator_uses_set_device(self):
        """
            when device is explicitly set,
            add docs orchestrator should call add_documents / _batch_request
            with set device, ignoring MARQO_BEST_AVAILABLE_DEVICE
        """
        test_cases = [
            ("cpu", "cuda", {}, ["marqo.tensor_search.tensor_search.add_documents"]),   # normal
            ("cpu", "cuda", {"batch_size": 2}, ["marqo.tensor_search.tensor_search._batch_request"]),    # server batched
            
            ("cuda", "cpu", {}, ["marqo.tensor_search.tensor_search.add_documents"]),   # normal
            ("cuda", "cuda", {"batch_size": 2}, ["marqo.tensor_search.tensor_search._batch_request"]),    # server batched
        ]
        for best_available_device, explicitly_set_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Call orchestrator
                tensor_search.add_documents_orchestrator(
                    config=self.config,
                    add_docs_params=AddDocsParams(index_name=self.index_name_1, 
                                    docs=[{"Title": "blah"} for i in range(5)], 
                                    auto_refresh=True,
                                    device=explicitly_set_device
                                ),
                    **extra_params
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    if "add_docs_params" in mocked_method.call_args[1]:
                        assert mocked_method.call_args[1]["add_docs_params"].device == explicitly_set_device
                    else:
                        assert mocked_method.call_args[1]["device"] == explicitly_set_device
                
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()
                return True

            assert run()
    
    def test_search_defaults_to_best_device(self):
        """
            when no device is set,
            search should call vector text search and reranker
            with env var MARQO_BEST_AVAILABLE_DEVICE
        """
        test_cases = [
            ("cpu", {}, ["marqo.tensor_search.tensor_search._vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
            ("cuda", {}, ["marqo.tensor_search.tensor_search._vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
        ]

        for best_available_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Add docs
                tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
                    auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[{"test": "blah"}])
                )

                # Call search
                tensor_search.search(
                    config=self.config, 
                    index_name=self.index_name_1, 
                    text="random search lol",
                    reranker="owl/ViT-B/32",
                    searchable_attributes=["test"],
                    # no device set, so should use default
                    **extra_params
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    assert mocked_method.call_args[1]["device"] == best_available_device
                
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()
                return True
            assert run()
    
    def test_search_uses_set_device(self):
        """
            when device is explicitly set,
            search should call vector text search and reranker
            with explicitly set device
        """
        test_cases = [
            ("cpu", "cuda", {}, ["marqo.tensor_search.tensor_search._vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
            ("cuda", "cpu", {}, ["marqo.tensor_search.tensor_search._vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
        ]

        for best_available_device, explicitly_set_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Add docs
                tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
                    auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[{"test": "blah"}])
                )

                # Call search
                tensor_search.search(
                    config=self.config, 
                    index_name=self.index_name_1, 
                    text="random search lol",
                    reranker="owl/ViT-B/32",
                    searchable_attributes=["test"],
                    device=explicitly_set_device,
                    **extra_params
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    assert mocked_method.call_args[1]["device"] == explicitly_set_device
                
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()
                return True
            assert run()
    
    @mock.patch("os.environ", dict())
    def test_search_fails_with_no_default(self):
        """
            If no best available device is set, this function should raise internal error.
        """
        self.assertNotIn("MARQO_BEST_AVAILABLE_DEVICE", os.environ)
        # Add docs
        tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
            auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[{"test": "blah"}])
        )

        try:
            # Call search
            tensor_search.search(
                config=self.config, 
                index_name=self.index_name_1, 
                text="random search lol",
                reranker="owl/ViT-B/32",
                searchable_attributes=["test"],
            )
            raise AssertionError
        except InternalError:
            pass

    def test_bulk_search_defaults_to_best_device(self):
        """
            when no device is set,
            bulk search should call bulk vector text search and reranker
            with env var MARQO_BEST_AVAILABLE_DEVICE
        """
        test_cases = [
            ("cpu", {}, ["marqo.tensor_search.tensor_search._bulk_vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
            ("cuda", {}, ["marqo.tensor_search.tensor_search._bulk_vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
        ]
        for best_available_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Mock bulk vector test search results
                for method, mock_obj in zip(called_methods, mocks):
                    if method == "marqo.tensor_search.tensor_search._bulk_vector_text_search":
                        mock_obj.return_value = self.mock_bulk_vector_text_search_results

                # Add docs
                tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
                    auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[
                        {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                        {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"}
                    ])
                )

                # Call bulk search
                tensor_search.bulk_search(
                    marqo_config=self.config, 
                    query=BulkSearchQuery(
                        queries=[
                            BulkSearchQueryEntity(
                                index=self.index_name_1, 
                                reRanker="owl/ViT-B/32",
                                q="match", 
                                searchableAttributes=["abc", "other field"],
                            ),
                            BulkSearchQueryEntity(
                                index=self.index_name_1, 
                                reRanker="owl/ViT-B/32",
                                q="match 2", 
                                searchableAttributes=["abc", "other field"],
                            )
                        ],
                        # no device set, so should use default
                        **extra_params
                    )
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    assert mocked_method.call_args[1]["device"] == best_available_device
                
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()
                return True
            assert run()
    
    def test_bulk_search_uses_set_device(self):
        """
            when device is explicitly set,
            bulk search should call bulk vector text search and reranker
            with explicitly set device
        """
        test_cases = [
            ("cpu", "cuda", {}, ["marqo.tensor_search.tensor_search._bulk_vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
            ("cuda", "cpu", {}, ["marqo.tensor_search.tensor_search._bulk_vector_text_search", "marqo.s2_inference.reranking.rerank.rerank_search_results"]),
        ]

        for best_available_device, explicitly_set_device, extra_params, called_methods in test_cases:
            @mock.patch.dict(os.environ, {**os.environ, **{"MARQO_BEST_AVAILABLE_DEVICE": best_available_device}})
            def run():
                # Mock inner methods
                # Create and start a patcher for each method
                patchers = [mock.patch(method) for method in called_methods]
                mocks = [patcher.start() for patcher in patchers]

                # Mock bulk vector test search results
                for method, mock_obj in zip(called_methods, mocks):
                    if method == "marqo.tensor_search.tensor_search._bulk_vector_text_search":
                        mock_obj.return_value = self.mock_bulk_vector_text_search_results

                # Add docs
                tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
                    auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[
                        {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                        {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"}
                    ])
                )

                # Call bulk search
                tensor_search.bulk_search(
                    marqo_config=self.config, 
                    device=explicitly_set_device,
                    query=BulkSearchQuery(
                        queries=[
                            BulkSearchQueryEntity(
                                index=self.index_name_1, 
                                reRanker="owl/ViT-B/32",
                                q="match", 
                                searchableAttributes=["abc", "other field"],
                            ),
                            BulkSearchQueryEntity(
                                index=self.index_name_1, 
                                reRanker="owl/ViT-B/32",
                                q="match 2", 
                                searchableAttributes=["abc", "other field"],
                            )
                        ],
                        # no device set, so should use default
                        **extra_params
                    )
                )
                # Confirm lower level functions were called with default device
                for mocked_method in mocks:
                    assert mocked_method.call_args[1]["device"] == explicitly_set_device
                # Stop all the patchers (important, if not stopped, will leak into next tests)
                for patcher in patchers:
                    patcher.stop()
                
                return True
            assert run()

    @mock.patch("os.environ", dict())
    def test_bulk_search_fails_with_no_default(self):
        """
            If no best available device is set, this function should raise internal error.
        """
        self.assertNotIn("MARQO_BEST_AVAILABLE_DEVICE", os.environ)
        # Add docs
        tensor_search.add_documents(config=self.config, add_docs_params = AddDocsParams(
            auto_refresh=True, device="cpu", index_name=self.index_name_1, docs=[{"test": "blah"}])
        )

        try:
            # Call bulk search
            tensor_search.bulk_search(
                marqo_config=self.config, 
                query=BulkSearchQuery(
                    queries=[
                        BulkSearchQueryEntity(
                            index=self.index_name_1, 
                            reRanker="owl/ViT-B/32",
                            q="match", 
                            searchableAttributes=["abc", "other field"],
                        ),
                        BulkSearchQueryEntity(
                            index=self.index_name_1, 
                            reRanker="owl/ViT-B/32",
                            q="match 2", 
                            searchableAttributes=["abc", "other field"],
                        )
                    ],
                    # no device set, so should use default
                )
            )
            raise AssertionError
        except InternalError:
            pass