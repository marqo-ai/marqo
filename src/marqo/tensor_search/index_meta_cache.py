"""This is a client index-IndexInfo cache. This keeps a mapping of index_name <> IndexInfo\

In the future this may be stored in redis or this logic be bundled with the
index in the search DB via a plugin.
"""
import threading
import time
from typing import Dict

from marqo.api import exceptions
from marqo.config import Config
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
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


def get_cache() -> Dict[str, MarqoIndex]:
    return index_info_cache


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

    if force_refresh or index_name not in index_info_cache:
        _refresh_index(config, index_name)

    if index_name in index_info_cache:
        return index_info_cache[index_name]

    # TODO: raise core_exceptions.IndexNotFoundError instead (fix associated tests)
    raise exceptions.IndexNotFoundError(f"Index {index_name} not found")


def _refresh_index(config: Config, index_name: str) -> None:
    """
    Refresh cache for a specific index
    """
    index_management = IndexManagement(config.vespa_client)
    try:
        index = index_management.get_index(index_name)
    except IndexNotFoundError as e:
        raise exceptions.IndexNotFoundError(f"Index {index_name} not found") from e

    index_info_cache[index_name] = index


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
                    try:
                        global cache_refresh_last_logged_time

                        populate_cache(config)

                        if time.time() - cache_refresh_last_logged_time > cache_refresh_log_interval:
                            cache_refresh_last_logged_time = time.time()
                            logger.info(f'Last index cache refresh at {cache_refresh_last_logged_time}')

                        time.sleep(cache_refresh_interval)
                    except Exception as e:
                        logger.error(f'Error in index cache refresh thread: {e}')

            refresh_thread = threading.Thread(target=refresh, daemon=True)
            refresh_thread.start()


def populate_cache(config: Config):
    """
    Refresh cache for all indexes
    """
    global index_info_cache
    index_management = IndexManagement(config.vespa_client)
    indexes = index_management.get_all_indexes()

    # Enable caching and reset any existing model caches
    # Create a map for one-pass cache update
    index_map = dict()
    for index in indexes:
        index_map[index.name] = index

    index_info_cache = index_map
