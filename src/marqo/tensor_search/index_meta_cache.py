"""This is a client index-IndexInfo cache. This keeps a mapping of index_name <> IndexInfo\

In the future this may be stored in redis or this logic be bundled with the
index in the search DB via a plugin.
"""
import threading
import time
from typing import Dict

from marqo import errors
from marqo.config import Config
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.tensor_search.models.index_info import IndexInfo
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)

index_info_cache = dict()

# the following is a non thread safe dict. Its purpose to be used by request
# threads to calculate whether to refresh an index's cached index_info.
# Because it is non thread safe, there is a chance multiple threads push out
# multiple refresh requests at the same. It isn't a critical problem if that
# happens.
cache_refresh_interval: int = 10  # seconds
cache_refresh_log_interval: int = 60
cache_refresh_last_logged_time: float = 0
refresh_thread = None
refresh_lock = threading.Lock()  # to ensure only one thread is operating on refresh_thread


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
    return
    # if index_name in index_info_cache:
    #     return index_info_cache[index_name]
    # else:
    #     found_index_info = backend.get_index_info(config=config, index_name=index_name)
    #     index_info_cache[index_name] = found_index_info
    #     return index_info_cache[index_name]


def get_cache() -> Dict[str, IndexInfo]:
    return index_info_cache


def refresh_index_info_on_interval(config: Config, index_name: str, interval_seconds: int) -> None:
    pass
    # """Refreshes an index's index_info if inteval_seconds have elapsed since the last time it was refreshed
    #
    # Non-thread safe, so there is a chance two threads both refresh index_info at the same time.
    # """
    # try:
    #     last_refreshed_time = index_last_refreshed_time[index_name]
    # except KeyError:
    #     last_refreshed_time = datetime.datetime.min
    #
    # now = datetime.datetime.now()
    #
    # interval_as_time_delta = datetime.timedelta(seconds=interval_seconds)
    # if now - last_refreshed_time >= interval_as_time_delta:
    #     # We assume that we will successfully refresh index info. We set the time to now ()
    #     # to lower the chance of other threads simultaneously refreshing the cache
    #     index_last_refreshed_time[index_name] = now
    #     try:
    #         backend.get_index_info(config=config, index_name=index_name)
    #
    #     # If we get any errors, we set the index's last refreshed time to what we originally found
    #     # This lets another thread come in and update it. There is a chance that, in the mean time
    #     except (errors.IndexNotFoundError, errors.NonTensorIndexError):
    #         # trying to refresh the index, and not finding any tensor index is considered a
    #         # successful of the index.
    #         pass
    #     except Exception as e2:
    #         # any other exception is problematic. We reset the index to the last_refreshed_time to
    #         # let another thread refresh the index's index_info
    #         index_last_refreshed_time[index_name] = last_refreshed_time
    #         logger.warning("refresh_index_info_on_interval(): error during background index_info refresh. Reason:"
    #                        f"\n{e2}")
    #         logger.debug("refresh_index_info_on_interval(): error during background index_info refresh. "
    #                      f"Traceback: \n{traceback.print_stack()}")
    #         raise e2


def get_index(config: Config, index_name: str, force_refresh=False) -> MarqoIndex:
    """
    Get an index.

    Args:
        force_refresh: Get index from Vespa even if already in cache. If False, Vespa is called only if index is not
        found in cache.

    Returns:

    """
    # Make sure refresh thread is running
    _check_refresh_thread(config)

    if force_refresh:
        _refresh_index(config, index_name)

    if index_name in index_info_cache:
        return index_info_cache[index_name]
    elif not force_refresh:
        _refresh_index(config, index_name)
        if index_name in index_info_cache:
            return index_info_cache[index_name]

    raise errors.IndexNotFoundError(f"Index {index_name} not found")


def _refresh_index(config: Config, index_name: str) -> None:
    """
    Refresh cache for a specific index
    """
    index_management = IndexManagement(config.vespa_client)
    try:
        index = index_management.get_index(index_name)
    except IndexNotFoundError as e:
        raise errors.IndexNotFoundError(f"Index {index_name} not found") from e

    index_info_cache[index_name] = index.copy_with_caching()


def _check_refresh_thread(config: Config):
    if refresh_lock.locked():
        # Another thread is running this function, skip as concurrent changes to the thread can error out
        logger.debug('Refresh thread is locked. Skipping')
        return

    with refresh_lock:
        global refresh_thread
        if refresh_thread is None or not refresh_thread.is_alive():
            if refresh_thread is not None:
                # If not None, then it has died
                logger.warn('Dead index cache refresh thread detected. Will start a new one')

            logger.info('Starting index cache refresh thread')

            def refresh():
                while True:
                    global cache_refresh_last_logged_time

                    populate_cache(config)

                    if time.time() - cache_refresh_last_logged_time > cache_refresh_log_interval:
                        cache_refresh_last_logged_time = time.time()
                        logger.info(f'Last index cache refresh at {cache_refresh_last_logged_time}')

                    time.sleep(cache_refresh_interval)

            refresh_thread = threading.Thread(target=refresh, daemon=True)
            refresh_thread.start()


def populate_cache(config: Config):
    """
    Refresh cache for all indexes
    """
    global index_info_cache
    index_management = IndexManagement(config.vespa_client)
    index_management.vespa_client.wait_for_application_convergence()
    indexes = index_management.get_all_indexes()

    # Enable caching and reset any existing model caches
    # Create a map for one-pass cache update
    index_map = dict()
    for index in indexes:
        index_clone = index.copy_with_caching()
        index_map[index.name] = index_clone

    index_info_cache = index_map
