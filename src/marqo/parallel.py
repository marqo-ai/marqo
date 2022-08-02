import os
import time
from typing import List, Dict

import torch
import numpy as np
from torch import multiprocessing as mp

from marqo.neural_search import neural_search
from marqo.marqo_logging import logger
from marqo.neural_search.configs import get_max_processes, get_threads_per_process
from marqo.neural_search import backend
from marqo.errors import MarqoApiError

try:
    mp.set_start_method('spawn', force=True)
except:
    pass

max_processes_cpu = get_max_processes()['max_processes_cpu']
max_processes_gpu = get_max_processes()['max_processes_gpu']
max_threads_per_process = get_threads_per_process()

def get_gpu_count(device: str):
    """ returns the number of gpus. if cpu is specified defaults to 0

    Args:
        device (_type_): 'cpu' or 'cuda'

    Returns:
        _type_: _description_
    """
    if device == 'cpu':
        return 0
    
    if device.startswith('cuda'):
        return torch.cuda.device_count()

    raise ValueError(f"expected on of 'cpu', 'cuda' or 'cuda:#' but received {device}")


def get_processes(device: str):
    """returns the processes available for either cpu or cuda

    Args:
        device (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if device == 'cpu':
        return max(1, min(mp.cpu_count(), max_processes_cpu))
    
    elif device.startswith('cuda'):
        assert torch.cuda.device_count() > 0, 'cannot find cuda enabled device'
        return max_processes_gpu

    raise ValueError(f"expected on of 'cpu', 'cuda' or 'cuda:#' but received {device}")

def get_device_ids(n_processes: int, device: str):
    """gets a list of device ids based on device
        e.g. get_device_ids(2, 'cpu')
        ['cpu', 'cpu']

        # single gpu machine
        e.g. get_device_ids(2, 'cuda')
        ['cuda:0', 'cuda:0']

        # two gpu machine
        e.g. get_device_ids(2, 'cuda')
        ['cuda:0', 'cuda:1']

        # two gpu machine
        e.g. get_device_ids(4, 'cuda')
        ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']

    Args:
        n_processes (int): _description_
        device (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if device.startswith('cpu'):
        return ['cpu']*n_processes
    elif device.startswith('cuda'):
        n_gpus = get_gpu_count('cuda')
        gpu_ids = list(range(n_gpus))*n_processes
        devices = [f'cuda:{_id}' for _id in gpu_ids]
        return devices[:n_processes]

    raise ValueError(f"expected on of 'cpu', 'cuda' or 'cuda:#' but received {device}")


class IndexChunk:

    """wrapper to pass through dopcuments to be indexed to multiprocessing
    """

    def __init__(self, config=None, index_name: str = None, docs: List[Dict] = [], 
                        auto_refresh: bool = False, batch_size: int = 50, device: str = None, process_id: int = 0):

        self.config = config
        self.index_name = index_name
        self.docs = docs
        self.auto_refresh = auto_refresh
        self.n_batch = batch_size
        self.n_docs = len(docs)
        self.n_chunks = max(1, self.n_docs // self.n_batch)
        self.device = device
        self.process_id = process_id
    
    def process(self):  

        # hf tokenizers setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        logger.info('starting add documents...')
        
        if self.device.startswith('cpu'):
            logger.info(f"restricting threads to {max_threads_per_process} for process={self.process_id}")
            torch.set_num_threads(max_threads_per_process)
        results = []
        start = time.time()
    
        total_progress_displays = 10
        progress_display_frequency = max(1, self.n_chunks // total_progress_displays)

        for n_processed,_doc in enumerate(np.array_split(self.docs, self.n_chunks)):
            t_chunk_start = time.time()
            percent_done = self._calculate_percent_done(n_processed + 1, self.n_chunks)
            
            if n_processed % progress_display_frequency == 0:
                logger.info(f'process={self.process_id} completed={percent_done}/100% on device={self.device}')
            
            results.append(neural_search.add_documents(config=self.config, index_name=self.index_name, docs=_doc,
                            auto_refresh=self.auto_refresh))
            t_chunk_end = time.time()

            time_left = round((self.n_chunks - (n_processed + 1))*(t_chunk_end - t_chunk_start), 0)
            if n_processed % progress_display_frequency == 0:
                logger.info(f'estimated time left for process {self.process_id} is {time_left} seconds ')

        end = time.time()
        logger.info(f'took {end - start} sec for {self.n_docs} documents')
        return results

    @staticmethod
    def _calculate_percent_done(current_step, total_steps, rounding=0):
        percent = current_step/max(1e-9,total_steps)
        return round(100*percent, rounding)
        
def _run_chunker(chunker: IndexChunk):
    """helper function to run the multiprocess by activating the chunker

    Args:
        chunker (IndexChunk): _description_

    Returns:
        _type_: _description_
    """
    res = chunker.process()
    return res

def add_documents_mp(config=None, index_name=None, docs=None, 
                     auto_refresh=None, batch_size=50):
    """add documents using parallel processing using ray

    Args:
        documents (_type_): _description_
        config (_type_, optional): _description_. Defaults to None.
        index_name (_type_, optional): _description_. Defaults to None.
        auto_refresh (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_documents = len(docs)

    logger.info(f"found {n_documents} documents")

    n_processes = get_processes(config.indexing_device)
    if n_documents < n_processes:
        n_processes = max(1, n_documents)
    logger.info(f"using {n_processes} processes")

    # get the device ids for each process based on the process count and available devices
    device_ids = get_device_ids(n_processes, config.indexing_device)
    
    start  = time.time()

    # we create the index if it does not exist
    logger.info("checking index exists and creating if not...")
    try:
        index_info = backend.get_index_info(config=config, index_name=index_name)
    except MarqoApiError as s:
        if s.status_code == 404 and "index_not_found_exception" in str(s):
            neural_search.create_vector_index(config=config, index_name=index_name)
            index_info = backend.get_index_info(config=config, index_name=index_name)
        else:
            raise s

    chunkers = [IndexChunk(config=config, index_name=index_name, docs=_docs,
                                auto_refresh=auto_refresh, batch_size=batch_size, process_id=p_id, device=device_ids[p_id]) 
                                for p_id,_docs in enumerate(np.array_split(docs, n_processes))]
    logger.info('Performing parallel now...')
    with mp.Pool(n_processes) as pool:
        results = pool.map(_run_chunker, chunkers)
    end = time.time()
    logger.info(f"finished indexing all documents. took {end - start} seconds to index {n_documents} documents")
    return results


