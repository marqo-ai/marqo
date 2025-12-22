import queue
import threading
import time
import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestConcurrencyRequestsBlock(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "type": "unstructured",
            }
        ])

        res = cls.client.index(cls.index_name).add_documents(
            [{"test_1": "what is best to wear on the moon?"},
             {"test_2": "what is best to wear on the moon?"}],
            tensor_fields=["test_1", "test_2"], device="cpu"
        )
        cls.indexes_to_delete = [cls.index_name]

    def setUp(self) -> None:
        self.device = "cpu"

    def tearDown(self) -> None:
        pass

    def normal_search(self, index_name, q):
        # A function will be called in threading
        try:
            res = self.client.index(index_name).search("what is best to wear on the moon?", device=self.device)
            if len(res["hits"]) == 2:
                q.put("normal search success")
            else:
                q.put(AssertionError)
        except Exception as e:
            q.put(e)

    def racing_search(self, index_name, q):
        # A function will be called in threading
        try:
            res = self.client.index(index_name).search("what is best to wear on the moon?", device=self.device)
            q.put(AssertionError)
        except MarqoWebError as e:
            if "Another model load/unload operation is in progress. Please try again later " in str(e):
                q.put("racing search get blocked with correct error")
            else:
                q.put(e)

    def test_concurrent_search_with_cache(self):
        # Search once to make sure the model is in cache
        res = self.client.index(self.index_name).search("what is best to wear on the moon?", device=self.device)

        normal_search_queue = queue.Queue()
        threads = []
        for i in range(2):
            t = threading.Thread(target=self.normal_search, args=(self.index_name, normal_search_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert normal_search_queue.qsize() == 2
        while not normal_search_queue.empty():
            assert normal_search_queue.get() == "normal search success"

    def test_concurrent_search_without_cache(self):
        # Remove all the cached models
        super().removeAllModels()

        normal_search_queue = queue.Queue()
        racing_search_queue = queue.Queue()
        threads = []
        main_thread = threading.Thread(target=self.normal_search, args=(self.index_name, normal_search_queue))
        main_thread.start()
        time.sleep(0.2)

        for i in range(2):
            t = threading.Thread(target=self.racing_search, args=(self.index_name, racing_search_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        main_thread.join()

        assert normal_search_queue.qsize() == 1
        while not normal_search_queue.empty():
            assert normal_search_queue.get() == "normal search success"

        assert racing_search_queue.qsize() == 2
        while not racing_search_queue.empty():
            assert racing_search_queue.get() == "racing search get blocked with correct error"
