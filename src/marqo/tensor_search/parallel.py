import os
import time
from typing import List, Dict
import copy

import torch
import numpy as np
from torch import multiprocessing as mp

from marqo.tensor_search import tensor_search
from marqo.marqo_logging import logger

try:
    mp.set_start_method('spawn', force=True)
except:
    pass

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


def get_processes(device: str, max_processes: int):
    """returns the processes available for either cpu or cuda
    Args:
        device (str): _description_
    Raises:
        ValueError: _description_
    Returns:
        _type_: _description_
    """
    if device == 'cpu':
        return max(1, min(mp.cpu_count(), max_processes))
    
    elif device.startswith('cuda'):
        assert torch.cuda.device_count() > 0, 'cannot find cuda enabled device'
        return max_processes

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

    """wrapper to pass through documents to be indexed to multiprocessing
    """

    def __init__(self, config=None, index_name: str = None, docs: List[Dict] = [], 
                        auto_refresh: bool = False, batch_size: int = 50, 
                        device: str = None, process_id: int = 0, 
                        non_tensor_fields: List[str] = [],
                        threads_per_process: int = None, update_mode: str = 'replace',
                        use_existing_vectors: bool = False):

        self.config = copy.deepcopy(config)
        self.index_name = index_name
        self.docs = docs
        self.auto_refresh = auto_refresh
        self.n_batch = batch_size
        self.n_docs = len(docs)
        self.n_chunks = max(1, self.n_docs // self.n_batch)
        self.device = device
        self.process_id = process_id
        self.update_mode = update_mode
        self.config.indexing_device = device if device is not None else self.config.indexing_device
        self.threads_per_process = threads_per_process
        self.non_tensor_fields = non_tensor_fields
        self.use_existing_vectors = use_existing_vectors

    def process(self):  

        # hf tokenizers setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        logger.info(f'starting add documents using {self.n_chunks} chunks per process...')
        
        if self.device.startswith('cpu') and self.threads_per_process is not None:
            logger.info(f"restricting threads to {self.threads_per_process} for process={self.process_id}")
            torch.set_num_threads(self.threads_per_process)
        
        results = []
        start = time.time()
    
        total_progress_displays = 10
        progress_display_frequency = max(1, self.n_chunks // total_progress_displays)

        for n_processed,_doc in enumerate(np.array_split(self.docs, self.n_chunks)):
            t_chunk_start = time.time()
            percent_done = self._calculate_percent_done(n_processed + 1, self.n_chunks)
            
            if n_processed % progress_display_frequency == 0:
                logger.info(f'process={self.process_id} completed={percent_done}/100% on device={self.device}')

            results.append(tensor_search.add_documents(
                config=self.config, index_name=self.index_name, docs=_doc, auto_refresh=self.auto_refresh,
                update_mode=self.update_mode, non_tensor_fields=self.non_tensor_fields, use_existing_vectors=self.use_existing_vectors
            ))
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

def get_threads_per_process(processes: int):
    total_cpu = max(1, mp.cpu_count() - 2)
    return max(1, total_cpu//processes)

def add_documents_mp(config=None, index_name=None, docs=None, 
                     auto_refresh=None, batch_size=50, processes=1, device=None,
                     non_tensor_fields: List[str] = [], update_mode: str = None,
                     use_existing_vectors=None):
    """add documents using parallel processing using ray
    Args:
        documents (_type_): _description_
        config (_type_, optional): _description_. Defaults to None.
        index_name (_type_, optional): _description_. Defaults to None.
        auto_refresh (_type_, optional): _description_. Defaults to None.
        non_tensor_fields (_type, List[str]): _description_. Fields within documents not to create 
          tensors for. Defaults to create tensors for all fields.
        update_mode (str, optional):
        use_existing_vectors
    
    Assumes running on the same host right now. Ray or something else should 
    be used if the processing is distributed.

    Returns:
        _type_: _description_
    """

    selected_device = device if device is not None else config.indexing_device

    n_documents = len(docs)

    logger.info(f"found {n_documents} documents")

    n_processes = get_processes(selected_device, processes)
    if n_documents < n_processes:
        n_processes = max(1, n_documents)
    
    # we want to restrict threads for cpu based mp as some torch models cause deadlocking
    threads_per_process = get_threads_per_process(n_processes)
    logger.info(f"using {n_processes} processes")

    # get the device ids for each process based on the process count and available devices
    device_ids = get_device_ids(n_processes, selected_device)

    start = time.time()

    chunkers = [IndexChunk(
            config=config, index_name=index_name, docs=_docs, non_tensor_fields=non_tensor_fields,
            auto_refresh=auto_refresh, batch_size=batch_size, update_mode=update_mode, 
            use_existing_vectors=use_existing_vectors,
            process_id=p_id, device=device_ids[p_id], threads_per_process=threads_per_process)
        for p_id,_docs in enumerate(np.array_split(docs, n_processes))]
    logger.info(f'Performing parallel now across devices {device_ids}...')
    with mp.Pool(n_processes) as pool:
        results = pool.map(_run_chunker, chunkers)
    end = time.time()
    logger.info(f"finished indexing all documents. took {end - start} seconds to index {n_documents} documents")
    return results