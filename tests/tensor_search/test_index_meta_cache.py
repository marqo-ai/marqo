import copy
import os
import datetime
import threading
import time
import unittest

import requests
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search import index_meta_cache
from marqo.config import Config
from marqo.api.exceptions import IndexNotFoundError
from marqo.tensor_search import utils
from marqo.tensor_search.enums import TensorField, SearchMethod
from tests.marqo_test import MarqoTestCase
from unittest import mock
from marqo.api import exceptions, configs


@unittest.skip
class TestIndexMetaCache(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.index_name_3 = "my-test-index-3" # for tests where index must be created as part of the test
        self.config = Config(self.authorized_url)
        self._delete_testing_indices()
        self._create_test_indices()

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2, self.index_name_3]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def _create_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_create = [self.index_name_1, self.index_name_2]
        else:
            ix_to_create = indices
        for ix_name in ix_to_create:
            tensor_search.create_vector_index(config=self.config, index_name=ix_name)

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

    def test_search_works_on_cache_clear(self):
        try:
            # reset the cache, too:
            index_meta_cache.empty_cache()
            # there needs to be an error because the index doesn't exist yet
            tensor_search.search(config=self.config, text="some text", index_name=self.index_name_3)
        except IndexNotFoundError as s:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_3)
        # no error, because there is an index, and the cache is updated:
        tensor_search.search(config=self.config, text="some text", index_name=self.index_name_3)
        # emptying the cache:
        index_meta_cache.empty_cache()
        # no error is thrown because the index is search, and the cache is updated
        tensor_search.search(config=self.config, text="some text", index_name=self.index_name_3)
        assert self.index_name_3 in index_meta_cache.get_cache()

    def test_add_new_fields_preserves_index_cache(self):
        add_doc_res_1 = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True, device="cpu")
        )
        add_doc_res_2 = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"cool field": "yep yep", "haha": "heheh"}],
                auto_refresh=True, device="cpu"
            )
        )
        index_info_t0 = index_meta_cache.get_cache()[self.index_name_1]
        # reset cache:
        index_meta_cache.empty_cache()
        add_doc_res_3 = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"newer field": "ndewr content",
                                                                     "goblin": "paradise"}],
                auto_refresh=True, device="cpu"
            )
        )

        # None of these vector fields should exist
        for field in ["newer field", "goblin", "cool field", "abc", "haha"]:
            assert utils.generate_vector_name(field) \
                   not in index_meta_cache.get_cache()[self.index_name_1].properties[TensorField.chunks]["properties"]
        
        # Only 1 vector field should exist
        assert TensorField.marqo_knn_field \
            in index_meta_cache.get_cache()[self.index_name_1].properties[TensorField.chunks]["properties"]

    def test_delete_removes_index_from_cache(self):
        """note the implicit index creation"""
        add_doc_res_1 = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True, device="cpu"
            )
        )
        add_doc_res_2 = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"abc": "def"}], auto_refresh=True, device="cpu"
            )
        )
        assert self.index_name_1 in index_meta_cache.get_cache()
        tensor_search.delete_index(index_name=self.index_name_1, config=self.config)
        assert self.index_name_1 not in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()

    def test_create_index_updates_cache(self):
        index_meta_cache.empty_cache()
        tensor_search.create_vector_index(index_name=self.index_name_3, config=self.config)
        assert TensorField.field_name \
               in index_meta_cache.index_info_cache[self.index_name_3].properties[TensorField.chunks]["properties"]

    def test_lexical_search_caching(self):
        d0 = {
            "d-one": "marqo", "_id": "abc1234",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field",
        }
        d1 = {"some doc 1": "some 2 marqo", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d2 = {"exclude me": "marqo"}
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[d0, d1, d2], device="cpu")
        )
        # reset cache
        index_meta_cache.empty_cache()
        search_res =tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo",
            searchable_attributes=["some doc 1", "d-one"])
        assert len(search_res['hits']) == 2
        assert d1 in [self.strip_marqo_fields(res) for res in search_res['hits']]
        assert d0 in [self.strip_marqo_fields(res) for res in search_res['hits']]

    def test_get_documents_caching(self):
        d0 = {
            "d-one": "marqo", "_id": "abc1234",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field",
        }
        d1 = {"some doc 1": "some 2 marqo", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d2 = {"exclude me": "marqo"}
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True,
                docs=[d0, d1, d2 ], device="cpu")
        )
        # reset cache
        index_meta_cache.empty_cache()
        search_res = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1, document_id="Jupyter_12")
        assert d1 == search_res

    def test_empty_cache(self):
        assert len(index_meta_cache.get_cache()) > 0
        index_meta_cache.empty_cache()
        assert len(index_meta_cache.get_cache()) == 0

    def _simulate_externally_added_docs(self, index_name, docs, check_only_in_external_cache: str = None):
        """helper function to simulate another client adding docs

        Args:
            docs: list of docs to add with add_documents()
            check_only_in_external_cache: a string to check ends up in the simulated
                external cache, but not in the local cache (it should be a field in
                'docs' but not previously indexed).
        """
        if check_only_in_external_cache is not None:
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties[TensorField.chunks]["properties"]
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties

        # save the state of the cache:
        cache_t0 = copy.deepcopy(index_meta_cache.get_cache())
        # mock external party indexing something:
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=index_name,
                docs=docs, auto_refresh=True, device="cpu"))

        if check_only_in_external_cache is not None:
            assert (
                    check_only_in_external_cache
                    in index_meta_cache.get_cache()[index_name].properties[TensorField.chunks]["properties"]
                   ) or (
                    check_only_in_external_cache
                    in index_meta_cache.get_cache()[index_name].properties
            )
        # set cache to t0 state:
        index_meta_cache.index_info_cache = copy.deepcopy(cache_t0)
        # search refreshes index cache every 2 seconds
        time.sleep(2.1)
        if check_only_in_external_cache is not None:
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties[TensorField.chunks]["properties"]
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties

    def test_search_lexical_externally_created_field(self):
        """ search (search_method=SearchMethod.lexical)
        after the first cache hit is empty, it should be updated.
        """
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
             search_method=SearchMethod.LEXICAL)
        assert len(result["hits"]) == 0
        # REFRESH INTERVAL IS 2 seconds
        time.sleep(4)
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
             search_method=SearchMethod.LEXICAL)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_search_vectors_externally_created_field(self):
        """ search (search_method=SearchMethod.chunk_embeddings)
        """
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
             search_method=SearchMethod.TENSOR)
        # With single KNN Field, correct result appears even when field is not in cache!
        assert "1234" in [h["_id"] for h in result["hits"]]
        assert len([h["_id"] for h in result["hits"]]) > 0
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
             search_method=SearchMethod.TENSOR)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_search_vectors_externally_created_field_attributes(self):
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()[self.index_name_1].properties
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.TENSOR)
        # With single KNN Field, correct result appears even when field is not in cache!
        assert result["hits"][0]["_id"] == "1234"

    def test_search_lexical_externally_created_field_attributes(self):
        """lexical search doesn't need an up-to-date cache to work"""
        index_meta_cache.empty_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_3)
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        self._simulate_externally_added_docs(
            self.index_name_3, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()[self.index_name_1].properties
        result = tensor_search.search(
            index_name=self.index_name_3, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.LEXICAL)
        assert result["hits"][0]["_id"] == "1234"
        result_2 = tensor_search.search(
            index_name=self.index_name_3, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.LEXICAL)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_vector_search_non_existent_field(self):
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        assert "brand new field" not in index_meta_cache.get_cache()[self.index_name_1].properties
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.TENSOR)
        assert result['hits'] == []

    def test_lexical_search_non_existent_field(self):
        """"""
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        assert "brand new field" not in index_meta_cache.get_cache()[self.index_name_1].properties
        # no error:
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="sftstsbtdts",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.LEXICAL)

    def test_cache_update_on_search(self):
        """
        The cache should update after search
        With single KNN field, doc should be found even if field is not in cache
        """
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                docs=[{"some field": "Plane 1"}], auto_refresh=True, device="cpu"))
        time.sleep(2.5)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()[self.index_name_1].properties
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.TENSOR)
        assert result["hits"][0]["_id"] == "1234"
        time.sleep(0.5)
        if self.config.cluster_is_remote:
            # Allow extra time if using a remote cluster
            time.sleep(3)
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
             search_method=SearchMethod.TENSOR)
        assert "brand new field" in index_meta_cache.get_cache()[self.index_name_1].properties
        assert result_2["hits"][0]["_id"] == "1234"

    def test_populate_cache(self):
        index_meta_cache.empty_cache()
        assert len(index_meta_cache.get_cache()) == 0
        index_meta_cache.populate_cache(config=self.config)
        assert self.index_name_1 in index_meta_cache.get_cache()

    def test_default_settings_appears_after_ix_creation(self):
        index_meta_cache.empty_cache()
        assert self.index_name_1 not in index_meta_cache.get_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_3)
        ix_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_3)
        assert ix_info.index_settings == configs.get_default_index_settings()

    def test_index_settings_after_cache_refresh(self):
        expected_index_settings = configs.get_default_index_settings()
        # Create index with some random model
        expected_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.model] = "open_clip/RN50/openai"
        index_meta_cache.empty_cache()
        assert self.index_name_3 not in index_meta_cache.get_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_3, index_settings={
                IndexSettingsField.index_defaults: {IndexSettingsField.model: "open_clip/RN50/openai"}}
        )
        ix_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_3)
        assert ix_info.index_settings == expected_index_settings

        index_meta_cache.empty_cache()
        assert self.index_name_3 not in index_meta_cache.get_cache()

        index_meta_cache.refresh_index(config=self.config, index_name=self.index_name_3)
        ix_refreshed_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_3)
        assert ix_refreshed_info.index_settings == expected_index_settings

    def test_index_refresh_on_interval_multi_threaded(self):
        """ This test involves spinning up 5 threads or so. these threads
            try to refresh the cache every 0.1 seconds. Despite this, the
            last_refresh_time ensures we only actually push out a mappings
            request once per second.
            Because checking the last_refresh_time isn't threadsafe, this
            test may occasionally fail. Enabling log output, and allowing
            more log output increases risk of test failure. However, most
            the time it should pass.

        """
        mock_get = mock.MagicMock()
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get})
        @mock.patch('requests.get', mock_get)
        def run():
            N_seconds = 3
            REFRESH_INTERVAL_SECONDS = 1
            start_time = datetime.datetime.now()
            num_threads = 5
            total_loops = [0] * num_threads
            sleep_time = 0.1

            def threaded_while(thread_num, loop_record):
                thread_loops = 0
                while datetime.datetime.now() - start_time < datetime.timedelta(seconds=N_seconds):
                    cache_update_thread = threading.Thread(
                        target=index_meta_cache.refresh_index_info_on_interval,
                        args=(self.config, self.index_name_1, REFRESH_INTERVAL_SECONDS))
                    cache_update_thread.start()
                    time.sleep(sleep_time)
                    thread_loops += 1
                loop_record[thread_num] = thread_loops

            threads = [threading.Thread(target=threaded_while, args=(i, total_loops)) for i in range(num_threads)]
            for th in threads:
                th.start()

            for th in threads:
                th.join()
            estimated_loops = round((N_seconds/sleep_time) * num_threads)
            assert sum(total_loops) in range(estimated_loops - num_threads, estimated_loops + 1)
            time.sleep(0.5)  # let remaining thread complete, if needed

            assert mock_get.call_count == N_seconds
            return True
        assert run()

    def test_index_refresh_on_interval_multi_threaded_no_index(self):
        """ If we encounter NonTensorIndexError/ IndexNotExists error
        while refreshing the index info, it is considered a successful
        refresh and the refresh happens on the intervals as usual.

        isn't threadsafe, this test may occasionally fail.

        """
        mock_get = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: '{"a":"b"}'

        # mock_get.return_value = mock_response
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get})
        @mock.patch('requests.get', mock_get)
        def run(error):
            def use_error(*args, **kwargs):
                raise error('')
            mock_get.side_effect = use_error

            N_seconds = 3
            REFRESH_INTERVAL_SECONDS = 1
            start_time = datetime.datetime.now()
            num_threads = 5
            total_loops = [0] * num_threads
            sleep_time = 0.1

            def threaded_while(thread_num, loop_record):
                thread_loops = 0
                while datetime.datetime.now() - start_time < datetime.timedelta(seconds=N_seconds):
                    cache_update_thread = threading.Thread(
                        target=index_meta_cache.refresh_index_info_on_interval,
                        args=(self.config, self.index_name_1, REFRESH_INTERVAL_SECONDS))
                    cache_update_thread.start()
                    time.sleep(sleep_time)
                    thread_loops += 1
                loop_record[thread_num] = thread_loops

            threads = [threading.Thread(target=threaded_while, args=(i, total_loops)) for i in range(num_threads)]
            for th in threads:
                th.start()

            for th in threads:
                th.join()
            estimated_loops = round((N_seconds/sleep_time) * num_threads)
            assert sum(total_loops) in range(estimated_loops - num_threads, estimated_loops + 1)
            time.sleep(0.5)  # let remaining thread complete, if needed
            assert mock_get.call_count == N_seconds
            return True
        assert run(error=exceptions.NonTensorIndexError)
        mock_get.reset_mock()
        assert run(error=exceptions.IndexNotFoundError)

    def test_index_refresh_on_interval_multi_threaded_errors(self):
        """ If we encounter any error besides
        NonTensorIndexError/ IndexNotExists we this is considered a
        failed refresh, which doesn't prevent other threads from
        trying to update it.

        This is not threadsafe and may occassionally fail

        """
        mock_get = mock.MagicMock()
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: '{"a":"b"}'

        # mock_get.return_value = mock_response
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get})
        @mock.patch('requests.get', mock_get)
        def run(error):
            def use_error(*args, **kwargs):
                raise error('')

            mock_get.side_effect = use_error

            N_seconds = 3
            REFRESH_INTERVAL_SECONDS = 1
            start_time = datetime.datetime.now()
            num_threads = 5
            total_loops = [0] * num_threads
            sleep_time = 0.1

            def threaded_while(thread_num, loop_record):
                thread_loops = 0
                while datetime.datetime.now() - start_time < datetime.timedelta(seconds=N_seconds):
                    cache_update_thread = threading.Thread(
                        target=index_meta_cache.refresh_index_info_on_interval,
                        args=(self.config, self.index_name_1, REFRESH_INTERVAL_SECONDS))
                    cache_update_thread.start()
                    time.sleep(sleep_time)
                    thread_loops += 1
                loop_record[thread_num] = thread_loops

            threads = [threading.Thread(target=threaded_while, args=(i, total_loops)) for i in range(num_threads)]
            for th in threads:
                th.start()

            for th in threads:
                th.join()
            estimated_loops = round((N_seconds / sleep_time) * num_threads)
            assert sum(total_loops) in range(estimated_loops - num_threads, estimated_loops + 1)
            time.sleep(0.5)  # let remaining thread complete, if needed
            # because we get these failures we set the last_refresh_time back to the original
            # allowing other threads to refresh index_info
            assert mock_get.call_count in range(estimated_loops - num_threads, estimated_loops + 1)
            return True

        assert run(error=ValueError)
        mock_get.reset_mock()
        assert run(error=requests.ConnectionError)

    def test_search_index_refresh_on_interval_multi_threaded(self):
        """ Same as test_index_refresh_on_interval_multi_threaded() ,
        but using the search endpoint.

        The same caveat applies: Because checking the last_refresh_time
        isn't threadsafe, this test may occasionally fail.
        """

        mock_get = mock.Mock()
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: '{"a":"b"}'
        mock_get.return_value = mock_response

        # we need to search it once, to get something in the cache, otherwise
        # the threads will see an empty cache and try to fill it
        try:
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, docs=[{"hi": "hello"}],
                    auto_refresh=False, device="cpu"))
        except IndexNotFoundError:
            pass
        @mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get})
        @mock.patch('marqo._httprequests.requests.get', mock_get)
        def run():

            # requests.get('23456')
            N_seconds = 4
            # the following is hard coded in search()
            REFRESH_INTERVAL_SECONDS = 2
            start_time = time.perf_counter_ns()
            num_threads = 5
            total_loops = [0] * num_threads
            sleep_time = 0.1

            def threaded_while(thread_num, loop_record):
                thread_loops = 0
                while time.perf_counter_ns() - start_time < (N_seconds * 1e9):
                    cache_update_thread = threading.Thread(
                        target=tensor_search.search,
                        kwargs={"config": self.config, "index_name": self.index_name_1, "text": "hello" })
                    cache_update_thread.start()
                    time.sleep(sleep_time)
                    thread_loops += 1
                loop_record[thread_num] = thread_loops
            threads = [threading.Thread(target=threaded_while, args=(i, total_loops)) for i in range(num_threads)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            estimated_loops = round((N_seconds/sleep_time) * num_threads)
            assert sum(total_loops) in range(estimated_loops - (2 * num_threads), estimated_loops + 1)
            time.sleep(0.5)  # let remaining thread complete, if needed
            mappings_call_count = len([c for c in mock_get.mock_calls if '_mapping' in str(c)])
            # for the refresh interal hardcoded in search(), which is 2 seconds, we expect a total
            # of only 2 calls to the mappings endpoint, even though there are a lot more search requests
            assert mappings_call_count == round(N_seconds/REFRESH_INTERVAL_SECONDS)
            return True
        assert run()

    def test_add_documents_to_unknown_index(self):
        """This happens when: halfway through the add_documents process, another thread deletes the index.
        When the add_documents process completes, it attempts to update mappings, but when it tries to get
        the existing info, it is no longer there.
        """
        # we need to rename this prevent infinite recursion inside mock_validate_doc
        from marqo.tensor_search.validation import validate_doc as og_validate_doc

        def mock_validate_doc(*args, **kwargs):
            # we want to slow this down slightly, so that the other thread can manipulate the index meta cache
            # validate_doc is between the initial get_index call and
            time.sleep(0.1)
            return og_validate_doc(*args, **kwargs)

        def clear_cache():
            """This will sleep briefly, allowing add_dcuments to start. When it runs it should be between
            both index_info calls.
            """
            time.sleep(0.1)
            index_meta_cache.empty_cache()

        @mock.patch("marqo.tensor_search.validation.validate_doc", mock_validate_doc)
        def run():
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_3,
                                              index_settings={"index_defaults": {"model": "random"}})
            clear_cache_thread = threading.Thread(target=clear_cache)
            clear_cache_thread.start()
            self.add_documents(
                config=self.config,
                add_docs_params=AddDocsParams(
                    **{
                        "index_name": self.index_name_1, "auto_refresh": True, "device":"cpu",
                        "docs": [
                            {"Title": "Blah"}, {"Title": "blah2"},
                            {"Title": "Blah3"}, {"Title": "Blah4"}]
                })
            )
            return True
        assert run()

    def test_add_documents_to_non_existent_index(self):
        """Same as test_add_documents_to_unknown_index but the other thread deletes the index.
        Instead of a 500 error, it should be an "index not found" error
        """
        # we need to rename this prevent infinite recursion inside mock_validate_doc
        from marqo.tensor_search.validation import validate_doc as og_validate_doc

        def mock_validate_doc(*args, **kwargs):
            # we want to slow this down slightly, so that the other thread can manipulate the index meta cache
            # validate_doc is between the initial get_index call and
            time.sleep(0.1)
            return og_validate_doc(*args, **kwargs)

        def delete_index():
            """This will sleep briefly, allowing add_documents to start. When it runs it should be between
            both index_info calls.
            """
            time.sleep(0.1)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_3)

        @mock.patch("marqo.tensor_search.validation.validate_doc", mock_validate_doc)
        def run():
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_3,
                                              index_settings={"index_defaults": {"model": "random"}})
            import unittest
            from typing import Dict, Any, List
            from unittest.mock import patch

            import pytest

            from marqo.core.constants import MARQO_DOC_ID
            from marqo.core.exceptions import DuplicateDocumentError, AddDocumentsError, MarqoDocumentParsingError, \
                InternalError
            from marqo.core.inference.tensor_fields_container import TensorFieldsContainer
            from marqo.core.models.add_docs_params import AddDocsParams, BatchVectorisationMode
            from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem
            from marqo.core.models.marqo_index import FieldType
            from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import \
                UnstructuredAddDocumentsHandler
            from marqo.core.vespa_index.add_documents_handler import AddDocumentsResponseCollector, AddDocumentsHandler
            from marqo.s2_inference import s2_inference
            from marqo.s2_inference.errors import S2InferenceError
            from marqo.vespa.models import VespaDocument, FeedBatchResponse, FeedBatchDocumentResponse
            from marqo.vespa.models.get_document_response import Document, GetBatchResponse, GetBatchDocumentResponse
            from tests.marqo_test import MarqoTestCase
            from tests.marqo_test import TestAudioUrls, TestVideoUrls, TestImageUrls

            @pytest.mark.unittest
            class TestAddDocumentHandler(MarqoTestCase):
                class DummyAddDocumentsHandler(AddDocumentsHandler):
                    """
                    We create a dummy implementation of the AddDocumentsHandler to verify the main workflow
                    """

                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.handled_fields = []
                        self.handled_multimodal_fields = []
                        self.existing_vespa_docs = []
                        self.to_vespa_doc_call_count = 0

                    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
                        return TensorFieldsContainer(self.add_docs_params.tensor_fields, [], {}, True)

                    def _handle_field(self, marqo_doc, field_name, field_content) -> None:
                        doc_id = marqo_doc[MARQO_DOC_ID]
                        marqo_doc[field_name] = field_content
                        self.tensor_fields_container.collect(doc_id, field_name, field_content, FieldType.Text)
                        self.handled_fields.append((doc_id, field_name))

                    def _handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
                        doc_id = marqo_doc[MARQO_DOC_ID]
                        self.handled_multimodal_fields.append(doc_id)

                    def _populate_existing_tensors(self, existing_vespa_docs: List[Document]) -> None:
                        self.existing_vespa_docs = existing_vespa_docs

                    def _to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
                        self.to_vespa_doc_call_count += 1
                        return VespaDocument(id=marqo_doc[MARQO_DOC_ID], fields={})

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                @patch('marqo.vespa.vespa_client.VespaClient.get_batch')
                def test_add_documents_main_workflow_happy_path(self, mock_get_batch, mock_feed_batch):
                    mock_get_batch.side_effect = [GetBatchResponse(errors=True, responses=[
                        GetBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1',
                                                 document=Document(id='id:index1:index1:1', fields={'marqo__id': '1'}),
                                                 status=200),
                        GetBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=404),
                        GetBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=404)
                    ])]
                    mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
                        FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
                        FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=200),
                        FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=200),
                    ])]

                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1',
                            tensor_fields=['field1'],
                            use_existing_tensors=True,
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '2', 'field1': 'hello again', 'field2': 3.0, 'field4': ['abcd']},
                                {'_id': '3', 'field2': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                            ])
                    )

                    response = handler.add_documents()

                    self.assertFalse(response.errors)
                    self.assertEqual('index1', response.index_name)
                    self.assertEqual(3, len(response.items))
                    for i in range(3):
                        self.assertEqual(str(i + 1), response.items[i].id)
                        self.assertEqual(200, response.items[i].status)

                    # verify the workflow call the abstract methods
                    self.assertEqual({
                        ('1', 'field1'), ('1', 'field2'), ('1', 'field3'),
                        ('2', 'field1'), ('2', 'field2'), ('2', 'field4'),
                        ('3', 'field2'), ('3', 'field5')
                    }, set(handler.handled_fields))

                    self.assertEqual({'3', '2', '1'}, set(handler.handled_multimodal_fields))

                    self.assertEqual([Document(id='id:index1:index1:1', fields={'marqo__id': '1'})],
                                     handler.existing_vespa_docs)  # only the doc with 200 status code is passed to the method

                    self.assertEqual(3, handler.to_vespa_doc_call_count)

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                def test_add_documents_should_skip_duplicate_documents(self, mock_feed_batch):
                    mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
                        FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
                    ])]
                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1'],
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '1', 'field4': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                            ])
                    )

                    self.assertFalse(handler.add_documents().errors)
                    self.assertEqual({
                        ('1', 'field4'), ('1', 'field5'),  # the second doc with the same id overrides the first one
                    }, set(handler.handled_fields))
                    self.assertEqual(1, handler.to_vespa_doc_call_count)

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                def test_add_documents_should_skip_duplicate_documents_even_when_the_latter_one_errors_out(self,
                                                                                                           mock_feed_batch):
                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1'],
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '1', 'field4': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                            ])
                    )

                    # override the handle field method to raise an error when handling field5
                    def handle_field_raise_error(self, marqo_doc, field_name, _) -> None:
                        if field_name == 'field5':
                            raise AddDocumentsError('some error')
                        self.handled_fields.append((marqo_doc[MARQO_DOC_ID], field_name))

                    handler._handle_field = handle_field_raise_error.__get__(handler)

                    response = handler.add_documents()
                    self.assertTrue(response.errors)
                    self.assertTrue(1, len(response.items))
                    self.assertEqual('some error', response.items[0].message)

                    self.assertEqual([('1', 'field4')], handler.handled_fields)
                    self.assertEqual(0, handler.to_vespa_doc_call_count)

                    self.assertEqual(1, mock_feed_batch.call_count)
                    self.assertEqual(([], 'index1'), mock_feed_batch.call_args_list[0][0])  # no vespa docs to persist

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                def test_add_documents_should_handle_various_errors(self, mock_feed_batch):
                    mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
                        FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=400,
                                                  message='Could not parse field field1'),
                        FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=429,
                                                  message='vespa error2'),
                        FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=507,
                                                  message='vespa error3'),
                    ])]

                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1'],
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '2', 'field1': 'hello again'},
                                {'_id': '3', 'field1': 'hello world'},
                                {'bad_field': 'bad_content'},  # error out when converting to vespa doc
                                {'_id': [5], 'field4': ['de']},  # doc with invalid id
                                {'field4': ['de'], 'field5': 'a very large string object' * 10000},  # doc too large
                                {},  # empty doc
                                [2.0] * 32  # doc is not a dict
                            ])
                    )

                    def to_vespa_doc_throw_error(_, marqo_doc: Dict[str, Any]) -> VespaDocument:
                        if marqo_doc.get('bad_field') == 'bad_content':
                            raise MarqoDocumentParsingError('MarqoDocumentParsingError')
                        return VespaDocument(id=marqo_doc[MARQO_DOC_ID], fields={})

                    handler._to_vespa_doc = to_vespa_doc_throw_error.__get__(handler)

                    response = handler.add_documents()
                    self.assertTrue(response.errors)

                    self.assertEqual([
                        MarqoAddDocumentsItem(status=400, id='1',
                                              message='The document contains invalid characters in the fields. Original error: Could not parse field field1 ',
                                              error='The document contains invalid characters in the fields. Original error: Could not parse field field1 ',
                                              code='vespa_error'),
                        MarqoAddDocumentsItem(status=429, id='2',
                                              message='Marqo vector store receives too many requests. Please try again later',
                                              error='Marqo vector store receives too many requests. Please try again later',
                                              code='vespa_error'),
                        MarqoAddDocumentsItem(status=400, id='3',
                                              message='Marqo vector store is out of memory or disk space',
                                              error='Marqo vector store is out of memory or disk space',
                                              code='vespa_error'),
                        MarqoAddDocumentsItem(status=400, id='', message='MarqoDocumentParsingError',
                                              error='MarqoDocumentParsingError', code='invalid_argument'),
                        MarqoAddDocumentsItem(status=400, id='',
                                              message='Document _id must be a string type! Received _id [5] of type `list`',
                                              error='Document _id must be a string type! Received _id [5] of type `list`',
                                              code='invalid_document_id'),
                        MarqoAddDocumentsItem(status=400, id='',
                                              message='Document with length `260032` exceeds the allowed document size limit of [100000].',
                                              error='Document with length `260032` exceeds the allowed document size limit of [100000].',
                                              code='doc_too_large'),
                        MarqoAddDocumentsItem(status=400, id='', message="Can't index an empty dict.",
                                              error="Can't index an empty dict.", code='invalid_argument'),
                        MarqoAddDocumentsItem(status=400, id='', message='Docs must be dicts',
                                              error='Docs must be dicts',
                                              code='invalid_argument')
                    ], response.items)

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                @patch('marqo.s2_inference.s2_inference.vectorise', wraps=s2_inference.vectorise)
                def test_add_documents_should_vectorise_tensor_fields_using_different_strategies(self, mock_vectorise,
                                                                                                 _):
                    for batch_mode, expected_vectorise_call_count, expected_call_args in [
                        (BatchVectorisationMode.PER_FIELD, 3, [['hello'], ['hello world'], ['ok']]),
                        (BatchVectorisationMode.PER_DOCUMENT, 2, [['hello'], ['hello world', 'ok']]),
                        (BatchVectorisationMode.PER_BATCH, 1, [['hello world', 'ok', 'hello']]),
                    ]:
                        with self.subTest(batch_mode=batch_mode):
                            handler = self.DummyAddDocumentsHandler(
                                vespa_client=self.vespa_client,
                                marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                                add_docs_params=AddDocsParams(
                                    index_name='index1', tensor_fields=['field1', 'field4'],
                                    batch_vectorisation_mode=batch_mode,
                                    docs=[
                                        {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                        {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                                    ])
                            )

                            mock_vectorise.reset_mock()

                            handler.add_documents()
                            self.assertEqual(expected_vectorise_call_count, mock_vectorise.call_count)
                            # please note that assertCountEqual compares two list ignoring order
                            self.assertCountEqual(expected_call_args,
                                                  [args.kwargs['content'] for args in mock_vectorise.call_args_list])

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                @patch('marqo.s2_inference.s2_inference.vectorise')
                def test_add_documents_should_fail_a_doc_using_vectorise_per_field_strategy(self, mock_vectorise,
                                                                                            mock_feed_batch):
                    mock_vectorise.side_effect = [S2InferenceError('vectorise error'), [[1.0, 2.0]]]
                    mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
                        FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
                    ])]
                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1', 'field4'],
                            batch_vectorisation_mode=BatchVectorisationMode.PER_FIELD,
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                            ])
                    )

                    response = handler.add_documents()
                    self.assertEqual(2, mock_vectorise.call_count)
                    self.assertTrue(response.errors)
                    self.assertEqual(200, response.items[0].status)
                    self.assertEqual(400, response.items[1].status)
                    self.assertEqual('vectorise error', response.items[1].message)

                @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
                @patch('marqo.s2_inference.s2_inference.vectorise')
                def test_add_documents_should_fail_a_doc_using_vectorise_per_doc_strategy(self, mock_vectorise,
                                                                                          mock_feed_batch):
                    mock_vectorise.side_effect = [S2InferenceError('vectorise error'), [[1.0, 2.0]]]
                    mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
                        FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
                    ])]
                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1', 'field4'],
                            batch_vectorisation_mode=BatchVectorisationMode.PER_DOCUMENT,
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                            ])
                    )

                    response = handler.add_documents()
                    self.assertEqual(2, mock_vectorise.call_count)
                    self.assertTrue(response.errors)
                    self.assertEqual(200, response.items[0].status)
                    self.assertEqual(400, response.items[1].status)
                    self.assertEqual('vectorise error', response.items[1].message)

                @patch('marqo.s2_inference.s2_inference.vectorise')
                def test_add_documents_should_fail_a_batch_using_vectorise_per_doc_strategy(self, mock_vectorise):
                    mock_vectorise.side_effect = [S2InferenceError('vectorise error')]

                    handler = self.DummyAddDocumentsHandler(
                        vespa_client=self.vespa_client,
                        marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=['field1', 'field4'],
                            batch_vectorisation_mode=BatchVectorisationMode.PER_BATCH,
                            docs=[
                                {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                                {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                            ])
                    )

                    with self.assertRaisesStrict(InternalError) as context:
                        handler.add_documents()

                    self.assertEqual('Encountered problem when vectorising batch of documents. Reason: vectorise error',
                                     str(context.exception))

                def test_unstructured_add_documents_handler_infer_modality_logic_image_false_and_media_false(self):
                    """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
                    both treat_urls_and_pointers_as_images and treat_urls_and_pointers_as_media are False."""
                    unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
                        marqo_index=self.unstructured_marqo_index(
                            'index1', 'index1',
                            treat_urls_and_pointers_as_images=False,
                            treat_urls_and_pointers_as_media=False
                        ),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
                        ),
                        vespa_client=self.vespa_client
                    )
                    test_cases = [
                        (TestAudioUrls.AUDIO1.value, "audio url should be treated as text"),
                        (TestVideoUrls.VIDEO1.value, "video url should be treated as text"),
                        (TestImageUrls.IMAGE1.value, "image url should be treated as text"),
                    ]
                    for url, test_case in test_cases:
                        with self.subTest(msg=test_case):
                            with (patch(
                                    "marqo.core.unstructured_vespa_index.unstructured_add_document_handler.infer_modality") as
                                  mock_infer_modality):
                                self.assertEqual(FieldType.Text,
                                                 unstructured_add_documents_handler._infer_field_type(url))
                            mock_infer_modality.assert_not_called()

                def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_false(self):
                    """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
                    treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=False."""
                    unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
                        marqo_index=self.unstructured_marqo_index(
                            'index1', 'index1',
                            treat_urls_and_pointers_as_images=True,
                            treat_urls_and_pointers_as_media=False
                        ),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
                        ),
                        vespa_client=self.vespa_client
                    )
                    test_cases = [
                        (TestAudioUrls.AUDIO1.value, "audio url should be treated as text", FieldType.Text),
                        (TestVideoUrls.VIDEO1.value, "video url should be treated as text", FieldType.Text),
                        (TestImageUrls.IMAGE1.value, "image url should be treated as image", FieldType.ImagePointer),
                    ]

                    for url, test_case, expected_field_type in test_cases:
                        with self.subTest(msg=test_case):
                            self.assertEqual(expected_field_type,
                                             unstructured_add_documents_handler._infer_field_type(url))

                def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_true(self):
                    """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
                    treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=True."""
                    unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
                        marqo_index=self.unstructured_marqo_index(
                            'index1', 'index1',
                            treat_urls_and_pointers_as_images=True,
                            treat_urls_and_pointers_as_media=True
                        ),
                        add_docs_params=AddDocsParams(
                            index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
                        ),
                        vespa_client=self.vespa_client
                    )
                    test_cases = [
                        (TestAudioUrls.AUDIO1.value, "audio url should be treated as audio", FieldType.AudioPointer),
                        (TestVideoUrls.VIDEO1.value, "video url should be treated as video", FieldType.VideoPointer),
                        (TestImageUrls.IMAGE1.value, "image url should be treated as image", FieldType.ImagePointer),
                    ]

                    for url, test_case, expected_field_type in test_cases:
                        with self.subTest(msg=test_case):
                            self.assertEqual(expected_field_type,
                                             unstructured_add_documents_handler._infer_field_type(url))

            @pytest.mark.unittest
            class TestAddDocumentsResponseCollector(unittest.TestCase):

                def test_should_collect_marqo_docs(self):
                    collector = AddDocumentsResponseCollector()
                    marqo_doc1 = {'_id': 'doc_id1'}
                    marqo_doc2 = {'_id': 'doc_id2'}

                    collector.collect_marqo_doc(1, marqo_doc1, 'doc_id1')
                    collector.collect_marqo_doc(2, marqo_doc2, None)

                    self.assertEqual(marqo_doc1, collector.marqo_docs['doc_id1'])
                    self.assertEqual(marqo_doc2, collector.marqo_docs['doc_id2'])
                    self.assertEqual(1, collector.marqo_doc_loc_map['doc_id1'])
                    self.assertEqual(2, collector.marqo_doc_loc_map['doc_id2'])
                    self.assertTrue(collector.visited('doc_id1'))
                    self.assertFalse(collector.visited('doc_id2'))
                    self.assertEqual({'doc_id1'}, collector.valid_original_ids())

                def test_collect_error_response_should_skip_duplicate_document_error(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_error_response('doc_id1', DuplicateDocumentError('duplicate'))
                    self.assertFalse(collector.errors)
                    self.assertEqual([], collector.responses)

                def test_collect_error_response_should_capture_add_document_error_with_default_values(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'), loc=1)
                    self.assertTrue(collector.errors)
                    loc, add_doc_item = collector.responses[0]
                    self.assertEqual(1, loc)
                    self.assertEqual('doc_id1', add_doc_item.id)
                    self.assertEqual('error message', add_doc_item.message)
                    self.assertEqual('error message', add_doc_item.error)
                    self.assertEqual(400, add_doc_item.status)
                    self.assertEqual('invalid_argument', add_doc_item.code)

                def test_collect_error_response_should_capture_add_document_error_with_custom_values(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message 2',
                                                                                  error_code='err_code',
                                                                                  status_code=403), loc=1)
                    self.assertTrue(collector.errors)
                    loc, add_doc_item = collector.responses[0]
                    self.assertEqual(1, loc)
                    self.assertEqual('doc_id1', add_doc_item.id)
                    self.assertEqual('error message 2', add_doc_item.message)
                    self.assertEqual('error message 2', add_doc_item.error)
                    self.assertEqual(403, add_doc_item.status)
                    self.assertEqual('err_code', add_doc_item.code)

                def test_collect_error_response_should_infer_loc_if_not_provided(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
                    loc, _ = collector.responses[0]
                    self.assertEqual(5, loc)

                def test_collect_marqo_error_response_should_set_loc_to_none_if_not_provided(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
                    loc, _ = collector.responses[0]
                    self.assertEqual(None, loc)

                def test_collect_marqo_error_response_should_remove_the_collected_marqo_doc(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
                    self.assertIn('doc_id1', collector.marqo_docs)

                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
                    self.assertNotIn('doc_id1', collector.marqo_docs)

                def test_collect_marqo_error_response_should_set_loc_to_none_if_doc_id_is_not_available(self):
                    """
                    This is possible due to persisting doc to Vespa do not always return doc_id when error is thrown.
                    """
                    collector = AddDocumentsResponseCollector()
                    collector.collect_error_response(None, AddDocumentsError('error message'))
                    loc, _ = collector.responses[0]
                    self.assertEqual(None, loc)

                def test_collect_marqo_error_response_should_set_id_as_empty_if_original_id_is_none(self):
                    """
                    If _id is not provided in the request, we will generate a random one. And this information should not be
                    returned to customer if this doc is not persisted. So we set the id in the error response to empty string
                    """
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, None)
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
                    _, add_document_item = collector.responses[0]
                    self.assertEqual('', add_document_item.id)

                def test_collect_marqo_error_response_should_set_doc_visited_if_original_id_is_present(self):
                    """
                    When dealing with duplicates, we only consider the last doc with that id, even it's not valid
                    """
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
                    collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
                    self.assertTrue(collector.visited('doc_id1'))

                def test_collect_successful_response_should_add_200_as_status_code(self):
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
                    collector.collect_successful_response('doc_id1')
                    loc, add_doc_item = collector.responses[0]
                    self.assertEqual(5, loc)
                    self.assertEqual('doc_id1', add_doc_item.id)
                    self.assertEqual(200, add_doc_item.status)
                    self.assertIsNone(add_doc_item.error)
                    self.assertIsNone(add_doc_item.message)
                    self.assertFalse(collector.errors)

                @patch('marqo.core.vespa_index.add_documents_handler.timer')
                def test_collect_final_responses(self, mock_timer):
                    mock_timer.side_effect = [1.0, 2.0]
                    collector = AddDocumentsResponseCollector()
                    collector.collect_marqo_doc(1, {'_id': 'doc_id1'}, 'doc_id1')
                    collector.collect_marqo_doc(2, {'_id': 'gen_doc_id2'}, None)
                    collector.collect_marqo_doc(3, {'_id': 'doc_id3'}, None)
                    collector.collect_error_response('doc_id4', AddDocumentsError('error message 4'), loc=4)
                    collector.collect_error_response(None, AddDocumentsError('error message 1'))
                    collector.collect_error_response('gen_doc_id2', AddDocumentsError('error message 2'))
                    collector.collect_successful_response('doc_id3')

                    # location should be reversed again in the response to revert the operation when we handle the batch of docs
                    response = collector.to_add_doc_responses(index_name='index')
                    self.assertTrue(response.errors)
                    self.assertEqual('index', response.index_name)
                    self.assertEqual(1000, response.processingTimeMs)

                    self.assertEqual(4, len(response.items))
                    self.assertEqual('doc_id4', response.items[0].id)  # doc_id4 is the original doc_id
                    self.assertEqual('error message 4', response.items[0].message)
                    self.assertEqual('doc_id3', response.items[1].id)  # doc_id3 should be returned since it's persisted
                    self.assertEqual('',
                                     response.items[2].id)  # gen_doc_id2 is generated, should not be returned for error
                    self.assertEqual('error message 2', response.items[2].message)
                    self.assertEqual('',
                                     response.items[3].id)  # doc_id1 error message does not contain id, this came last
                    self.assertEqual('error message 1', response.items[3].message)

            clear_cache_thread = threading.Thread(target=delete_index)
            clear_cache_thread.start()
            try:
                self.add_documents(
                    **{"config": self.config},
                    add_docs_params=AddDocsParams(
                        **{
                            "index_name": self.index_name_3, "auto_refresh": True,  "device":"cpu",
                            "docs": [
                                {"Title": "Blah"}, {"Title": "blah2"},
                                {"Title": "Blah3"}, {"Title": "Blah4"}]
                        })
                )
                raise AssertionError
            except exceptions.IndexNotFoundError:
                pass
            return True

        assert run()
