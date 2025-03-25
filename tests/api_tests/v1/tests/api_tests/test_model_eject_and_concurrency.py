import queue
import threading
import time
import unittest
import uuid

import pytest
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


@pytest.mark.cuda_test
class TestModelEject(MarqoTestCase):
    '''Although the test is running in cpu, we restrict it to cuda environments due to its intensive usage of memory.'''

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.device = "cpu"
        cls.index_model_object = {
            "test_0": 'open_clip/ViT-B-32/laion400m_e31',
            "test_1": 'open_clip/ViT-B-32/laion400m_e32',
            "test_2": 'open_clip/convnext_base_w/laion2b_s13b_b82k',
            "test_3": 'open_clip/ViT-B-16-plus-240/laion400m_e32',
            "test_4": 'open_clip/RN50x4/openai',
            "test_5": 'open_clip/RN101-quickgelu/yfcc15m',
            "test_6": 'open_clip/ViT-B-32/laion2b_e16',
            "test_7": 'open_clip/ViT-B-32-quickgelu/laion400m_e31',
            "test_8": 'open_clip/ViT-B-16-plus-240/laion400m_e31',
            "test_9": 'open_clip/ViT-L-14/laion2b_s32b_b82k',
            "test_10": "hf/all-MiniLM-L6-v1",
            "test_11": "hf/all-MiniLM-L6-v2",
            "test_12": 'open_clip/ViT-B-16/laion400m_e32',
            "test_13": "hf/all_datasets_v3_MiniLM-L12",
            "test_14": 'open_clip/ViT-B-32/laion2b_e16',
            "test_15": 'open_clip/RN101/yfcc15m',
            "test_16": 'open_clip/convnext_base/laion400m_s13b_b51k',
            "test_17": 'open_clip/convnext_base_w/laion2b_s13b_b82k',
            "test_18": 'open_clip/ViT-B-32/laion2b_s34b_b79k',
            "test_19": 'open_clip/ViT-B-16-plus-240/laion400m_e31',
            "test_20": 'open_clip/ViT-L-14/laion400m_e31',
            "test_21": 'open_clip/ViT-L-14/laion2b_s32b_b82k',
            "test_22": 'open_clip/ViT-B-16/laion400m_e32',
        }

        cls.create_indexes([
            {
                "indexName": index_name,
                "model": model,
                "type": "unstructured",
            } for index_name, model in cls.index_model_object.items()
        ])

        cls.indexes_to_delete = list(cls.index_model_object)

    def test_sequentially_search(self):
        """Iterate through each index and loading each model. We expect to not run out of space as previously
         loaded models are ejected to make space for newer ones.

        If the Marqo does through this test, it indicates that a problem with model cache ejection.

        Running this without a sleep between each call sometimes kills Marqo. This is probably because
        we don't have much control over the garbage collection of dereferenced objects in Python,
        resulting in an Out Of Memory crash.

        Because rapidly loading different models is a niche usecase, we want to relax the strictness of
        the test (by adding a sleep) rather than making the ejections stricter (for example, by locking
        the available models dict).
        """

        # this downloads the models if they aren't already downloaded
        for index_name in list(self.index_model_object):
            self.client.index(index_name).search(q='What is the best outfit to wear on the moon?', device=self.device)
            time.sleep(5)

        # this loads the models from disk to memory
        for index_name in list(self.index_model_object):
            self.client.index(index_name).search(q='What is the best outfit to wear on the moon?', device=self.device)
            time.sleep(5)
        return True


class TestConcurrencyRequestsBlock(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "model": "open_clip/ViT-B-32/laion400m_e31",
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
            if "Request rejected, as this request attempted to update the model cache," in str(e):
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
