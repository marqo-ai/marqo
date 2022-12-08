import multiprocessing as mp
from urllib.request import urlretrieve
from pathlib import Path
import os
from functools import partial
from typing import List
import numpy as np
import zipfile

def download_files(urls: str, local_dir: str) -> List[str]:

    results = []
    N = len(urls)
    for ii,url in enumerate(urls):
        result = download_file(url, local_dir)
        results.append(result)
        if ii % 10 == 0:
            print(f"{round(100*(ii+1)/N, 3)}%")
    return results

def download_file(url: str, local_dir: str) -> str:
    """_summary_

    Args:
        url (str): _description_
        local_dir (str): local directory to download to
    """

    if not local_dir.endswith('/'): local_dir += '/'

    Path(local_dir).mkdir(exist_ok=True, parents=True)

    full_local_path = local_dir + os.path.basename(url)

    if not os.path.isfile(full_local_path):
        full_local_path, _ = urlretrieve(url, full_local_path)

    return full_local_path

def download_parallel(urls: List[str], local_dir: str, n_processes=8) -> List[str]:
    
    N = len(urls)
    print(f"downloading {N} urls to {local_dir} using {n_processes} processes")

    func = partial(download_files, local_dir=local_dir)

    urls_split = np.array_split(urls, n_processes)
    urls_split = [split.tolist() for split in urls_split]
    with mp.Pool(n_processes) as pool:
        results = pool.map(func, urls_split)

    return results

def extract_zip(zip_file, local_dir):

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(local_dir)
