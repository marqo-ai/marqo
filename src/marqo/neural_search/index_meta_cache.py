"""This is a client index-IndexInfo cache. This keeps a mapping of index_name <> IndexInfo
This is a band-aid solution, as this introduces state into the Neural State application.

In the future this may be stored in redis or this logic be bundled with the
index in the search DB via a plugin.
"""
import asyncio
import datetime
import time
from multiprocessing import Process, Manager
from marqo.neural_search.models.index_info import IndexInfo
from typing import Dict
from marqo import errors
from marqo.neural_search import backend
from marqo.config import Config

index_info_cache = dict()


def empty_cache():
    global index_info_cache
    index_info_cache = dict()


def get_index_info(config: Config, index_name: str) -> IndexInfo:
    """Looks for the index name in the cache.

    If it isn't found there, it will try searching the cluster

    Args:
        config:
        index_name: name of the index

    Returns:
        IndexInfo - information about the index.

    Raises:
         MarqoError if the index isn't found on the cluster
    """
    if index_name in index_info_cache:
        return index_info_cache[index_name]
    else:
        found_index_info = backend.get_index_info(config=config, index_name=index_name)
        index_info_cache[index_name] = found_index_info
        return index_info_cache[index_name]


def get_cache() -> Dict[str, IndexInfo]:
    return index_info_cache


def refresh_index(config: Config, index_name: str) -> IndexInfo:
    """function to update an index, from the cluster.

    Args:
        config:
        index_name

    Returns:

    """
    found_index_info = backend.get_index_info(config=config, index_name=index_name)
    index_info_cache[index_name] = found_index_info
    return found_index_info


def populate_cache(config: Config):
    """Identify available index names and use them to populate the cache.

    Args:
        config:
    """
    for ix_name in backend.get_cluster_indices(config=config):
        try:
            found_index_info = backend.get_index_info(config=config, index_name=ix_name)
            index_info_cache[ix_name] = found_index_info
        except errors.MarqoNonNeuralIndexError as e:
            pass


