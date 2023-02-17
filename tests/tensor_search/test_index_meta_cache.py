import copy
import datetime
import pprint
import threading
import time
import unittest
from marqo.tensor_search.models.index_info import IndexInfo
from marqo.tensor_search.models import index_info
from marqo.tensor_search import tensor_search
from marqo.tensor_search import index_meta_cache
from marqo.config import Config
from marqo.errors import MarqoError, MarqoApiError, IndexNotFoundError
from marqo.tensor_search import utils
from marqo.tensor_search.enums import TensorField, SearchMethod, IndexSettingsField
from marqo.tensor_search import configs
from tests.marqo_test import MarqoTestCase
from unittest import mock


class TestIndexMetaCache(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.config = Config(self.authorized_url)
        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

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
            tensor_search.search(config=self.config, text="some text", index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        # no error, because there is an index, and the cache is updated:
        tensor_search.search(config=self.config, text="some text", index_name=self.index_name_1)
        # emptying the cache:
        index_meta_cache.empty_cache()
        # no error is thrown because the index is search, and the cache is updated
        tensor_search.search(config=self.config, text="some text", index_name=self.index_name_1)
        assert self.index_name_1 in index_meta_cache.get_cache()

    def test_add_new_fields_preserves_index_cache(self):
        add_doc_res_1 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        add_doc_res_2 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"cool field": "yep yep", "haha": "heheh"}],
            auto_refresh=True
        )
        index_info_t0 = index_meta_cache.get_cache()[self.index_name_1]
        # reset cache:
        index_meta_cache.empty_cache()
        add_doc_res_3 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"newer field": "ndewr content",
                                                                     "goblin": "paradise"}],
            auto_refresh=True
        )
        for field in ["newer field", "goblin", "cool field", "abc", "haha"]:
            assert utils.generate_vector_name(field) \
                   in index_meta_cache.get_cache()[self.index_name_1].properties[TensorField.chunks]["properties"]

    def test_delete_removes_index_from_cache(self):
        """note the implicit index creation"""
        add_doc_res_1 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[{"abc": "def"}], auto_refresh=True
        )
        add_doc_res_2 = tensor_search.add_documents(
            config=self.config, index_name=self.index_name_2, docs=[{"abc": "def"}], auto_refresh=True
        )
        assert self.index_name_1 in index_meta_cache.get_cache()
        tensor_search.delete_index(index_name=self.index_name_1, config=self.config)
        assert self.index_name_1 not in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()

    def test_create_index_updates_cache(self):
        index_meta_cache.empty_cache()
        tensor_search.create_vector_index(index_name=self.index_name_1, config=self.config)
        assert TensorField.field_name \
               in index_meta_cache.index_info_cache[self.index_name_1].properties[TensorField.chunks]["properties"]

    def test_lexical_search_caching(self):
        d0 = {
            "d-one": "marqo", "_id": "abc1234",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field",
        }
        d1 = {"some doc 1": "some 2 marqo", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d2 = {"exclude me": "marqo"}
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d1, d2])
        # reset cache
        index_meta_cache.empty_cache()
        search_res =tensor_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo",
            searchable_attributes=["some doc 1", "d-one"], return_doc_ids=True)
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
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d1, d2 ])
        # reset cache
        index_meta_cache.empty_cache()
        search_res = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1, document_id="Jupyter_12")
        assert d1 == search_res

    def test_empty_cache(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
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
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True)

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
        time.sleep(1)
        if check_only_in_external_cache is not None:
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties[TensorField.chunks]["properties"]
            assert check_only_in_external_cache not in \
                   index_meta_cache.get_cache()[index_name].properties

    def test_search_lexical_externally_created_field(self):
        """ search (search_method=SearchMethod.lexical)
        after the first cache hit is empty, it should be updated.
        """
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            return_doc_ids=True, search_method=SearchMethod.LEXICAL)
        assert len(result["hits"]) == 0
        time.sleep(1)
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            return_doc_ids=True, search_method=SearchMethod.LEXICAL)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_search_vectors_externally_created_field(self):
        """ search (search_method=SearchMethod.chunk_embeddings)
        """
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert "1234" not in [h["_id"] for h in result["hits"]]
        assert len([h["_id"] for h in result["hits"]]) > 0
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_search_vectors_externally_created_field_attributes(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert result['hits'] == []

    def test_search_lexical_externally_created_field_attributes(self):
        """lexical search doesn't need an up-to-date cache to work"""
        index_meta_cache.empty_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.LEXICAL)
        assert result["hits"][0]["_id"] == "1234"
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.LEXICAL)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_vector_search_non_existent_field(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        assert "brand new field" not in index_meta_cache.get_cache()
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert result['hits'] == []

    def test_lexical_search_non_existent_field(self):
        """"""
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        assert "brand new field" not in index_meta_cache.get_cache()
        # no error:
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="sftstsbtdts",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.LEXICAL)

    def test_search_vectors_externally_created_field_attributes_cache_update(self):
        """The cache should update after getting no hits at first"""
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some field": "Plane 1"}], auto_refresh=True)
        self._simulate_externally_added_docs(
            self.index_name_1, [{"brand new field": "a line of text", "_id": "1234"}], "brand new field")
        assert "brand new field" not in index_meta_cache.get_cache()
        result = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert result['hits'] == []
        time.sleep(0.5)
        if self.config.cluster_is_remote:
            # Allow extra time if using a remote cluster
            time.sleep(3)
        result_2 = tensor_search.search(
            index_name=self.index_name_1, config=self.config, text="a line of text",
            searchable_attributes=["brand new field"],
            return_doc_ids=True, search_method=SearchMethod.TENSOR)
        assert result_2["hits"][0]["_id"] == "1234"

    def test_populate_cache(self):
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        index_meta_cache.empty_cache()
        assert len(index_meta_cache.get_cache()) == 0
        index_meta_cache.populate_cache(config=self.config)
        assert self.index_name_1 in index_meta_cache.get_cache()

    def test_default_settings_appears_after_ix_creation(self):
        index_meta_cache.empty_cache()
        assert self.index_name_1 not in index_meta_cache.get_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        ix_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_1)
        assert ix_info.index_settings == configs.get_default_index_settings()

    def test_index_settings_after_cache_refresh(self):
        expected_index_settings = configs.get_default_index_settings()
        expected_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.model] = "special_model_1"

        index_meta_cache.empty_cache()
        assert self.index_name_1 not in index_meta_cache.get_cache()
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings={
                IndexSettingsField.index_defaults: {IndexSettingsField.model: "special_model_1"}}
        )
        ix_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_1)
        assert ix_info.index_settings == expected_index_settings

        index_meta_cache.empty_cache()
        assert self.index_name_1 not in index_meta_cache.get_cache()

        index_meta_cache.refresh_index(config=self.config, index_name=self.index_name_1)
        ix_refreshed_info = index_meta_cache.get_index_info(config=self.config, index_name=self.index_name_1)
        assert ix_refreshed_info.index_settings == expected_index_settings

    def test_index_refresh_on_interval_multi_threaded(self):
        """ This test involves spinning up 5 threads or so. these threads
            try to refresh the cache every 0.1 seconds. Despite this, the
            last_refresh_time ensures we only actually push out a mappings
            request once per second.
            Because checking the last_refresh_time isn't threadsafe, this
            test may occasionally fail. However, most the time it should pass.

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
            assert sum(total_loops) in range(estimated_loops - num_threads, estimated_loops)
            time.sleep(0.5)  # let remaining thread complete, if needed

            assert mock_get.call_count == N_seconds
            return True
        assert run()


