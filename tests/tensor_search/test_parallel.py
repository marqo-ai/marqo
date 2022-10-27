from marqo.errors import IndexNotFoundError
import unittest
import copy
from marqo.tensor_search import parallel
import torch
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search


class TestAddDocumentsPara(MarqoTestCase):
    """
    This test generates SSL warnings when running against a local Marqo because
    parallel.py turns on logging.
    """

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
    
    def test_get_device_ids(self) -> None:
        assert parallel.get_gpu_count('cpu') == 0

        assert parallel.get_gpu_count('cuda') == torch.cuda.device_count()

        # TODO need a gpu test

    def test_get_device_ids_2(self) -> None:

        assert parallel.get_device_ids(1, 'cpu') == ['cpu']

        assert parallel.get_device_ids(2, 'cpu') == ['cpu', 'cpu']

        # TODO need a gpu test

    def test_get_processes(self) -> None:

        assert parallel.get_processes('cpu', max_processes=100) >= 1

    def test_add_documents_parallel(self) -> None:

        data = [{'text':f'something {str(i)}', '_id': str(i)} for i in range(100)]

        res = tensor_search.add_documents_orchestrator(config=self.config, index_name=self.index_name_1, docs=data,
                                                       batch_size=10, processes=1, auto_refresh=True)
        res = tensor_search.search(config=self.config, text='something', index_name=self.index_name_1)