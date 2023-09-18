"""This is a client index-IndexInfo cache. This keeps a mapping of index_name <> IndexInfo\

In the future this may be stored in redis or this logic be bundled with the
index in the search DB via a plugin.
"""
import asyncio
import datetime
import time
import traceback
from multiprocessing import Process, Manager
from marqo.tensor_search.models.index_info import IndexInfo
from typing import Dict
from marqo import errors
from marqo.tensor_search import backend
from marqo.config import Config
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


index_info_cache = dict()

# the following is a non thread safe dict. Its purpose to be used by request
# threads to calculate whether to refresh an index's cached index_info.
# Because it is non thread safe, there is a chance multiple threads push out
# multiple refresh requests at the same. It isn't a critical problem if that
# happens.
index_last_refreshed_time = dict()


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


def refresh_index_info_on_interval(config: Config, index_name: str, interval_seconds: int) -> None:
    """Refreshes an index's index_info if interval_seconds have elapsed since the last time it was refreshed

    Non-thread safe, so there is a chance two threads both refresh index_info at the same time.
    """
    try:
        last_refreshed_time = index_last_refreshed_time[index_name]
    except KeyError:
        last_refreshed_time = datetime.datetime.min

    now = datetime.datetime.now()

    interval_as_time_delta = datetime.timedelta(seconds=interval_seconds)
    if now - last_refreshed_time >= interval_as_time_delta:
        # We assume that we will successfully refresh index info. We set the time to now ()
        # to lower the chance of other threads simultaneously refreshing the cache
        index_last_refreshed_time[index_name] = now
        try:
            backend.get_index_info(config=config, index_name=index_name)

        # If we get any errors, we set the index's last refreshed time to what we originally found
        # This lets another thread come in and update it. There is a chance that, in the mean time
        except (errors.IndexNotFoundError, errors.NonTensorIndexError):
            # trying to refresh the index, and not finding any tensor index is considered a
            # successful of the index.
            pass
        except Exception as e2:
            # any other exception is problematic. We reset the index to the last_refreshed_time to
            # let another thread refresh the index's index_info
            index_last_refreshed_time[index_name] = last_refreshed_time
            logger.warning("refresh_index_info_on_interval(): error during background index_info refresh. Reason:"
                           f"\n{e2}")
            logger.debug("refresh_index_info_on_interval(): error during background index_info refresh. "
                         f"Traceback: \n{traceback.print_stack()}")
            raise e2


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
        except errors.NonTensorIndexError as e:
            pass


