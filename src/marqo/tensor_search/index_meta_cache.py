"""This is a client index-IndexInfo cache. This keeps a mapping of index_name <> IndexInfo\

In the future this may be stored in redis or this logic be bundled with the
index in the search DB via a plugin.
"""
import threading
import time
from typing import Dict

from marqo import marqo_docs
from marqo.api import exceptions
from marqo.config import Config
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.vespa.exceptions import VespaError, VespaStatusError

logger = get_logger(__name__)

index_info_cache = dict()

# the following is a non thread safe dict. Its purpose to be used by request
# threads to calculate whether to refresh an index's cached index_info.
# Because it is non thread safe, there is a chance multiple threads push out
# multiple refresh requests at the same. It isn't a critical problem if that
# happens.
cache_refresh_interval: int = 1  # seconds
cache_refresh_log_interval: int = 60
cache_refresh_last_logged_time: float = 0
refresh_thread = None
refresh_lock = threading.Lock()  # to ensure only one thread is operating on refresh_thread


def empty_cache():
    global index_info_cache
    index_info_cache = dict()


def get_cache() -> Dict[str, MarqoIndex]:
    return index_info_cache


def get_index(index_management: IndexManagement, index_name: str, force_refresh=False) -> MarqoIndex:
    """
    Get an index.

    Args:
        index_management: IndexManagement object to load the index if not found in cache
        index_name (str): Name of the index to retrieve.
        force_refresh: Get index from Vespa even if already in cache. If False, Vespa is called only if index is not
        found in cache.

    Returns:
        The latest index if the index is not found in cache or force_refresh flag is True. Otherwise, the cached index

    Raises:
        IndexNotFoundError: If index is not found in cache.
    """
    # Make sure refresh thread is running
    _check_refresh_thread(index_management)

    if force_refresh or index_name not in index_info_cache:
        _refresh_index(index_management, index_name)

    if index_name in index_info_cache:
        return index_info_cache[index_name]

    # TODO: raise core_exceptions.IndexNotFoundError instead (fix associated tests)
    raise exceptions.IndexNotFoundError(f"Index {index_name} not found")


def _refresh_index(index_management: IndexManagement, index_name: str) -> None:
    """
    Refresh cache for a specific index
    """
    try:
        index = index_management.get_index(index_name)
    except IndexNotFoundError as e:
        raise exceptions.IndexNotFoundError(f"Index {index_name} not found") from e

    index_info_cache[index_name] = index


def _check_refresh_thread(index_management: IndexManagement):
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

                        populate_cache(index_management)

                        if time.time() - cache_refresh_last_logged_time > cache_refresh_log_interval:
                            cache_refresh_last_logged_time = time.time()
                            logger.info(f'Last index cache refresh at {cache_refresh_last_logged_time}')
                    except VespaError as e:
                        if isinstance(e, VespaStatusError) and e.status_code == 400:
                            # This can happen when settings schema doesn't exist
                            logger.warn(
                                'Failed to populate index cache due to 400 error from vector store. This can happen '
                                f'if Marqo settings schema does not exist. Error: {e}'
                            )
                        else:
                            logger.error(
                                "Failed to connect to vector store. If you are using an external vector store, "
                                "ensure that Marqo is configured properly for this. See "
                                f"{marqo_docs.configuring_marqo()} for more details. Error: {e}"
                            )
                    except Exception as e:
                        logger.error(f'Unexpected error in index cache refresh thread: {e}')

                    time.sleep(cache_refresh_interval)

            refresh_thread = threading.Thread(target=refresh, daemon=True)
            refresh_thread.start()


def start_refresh_thread(config: Config):
    _check_refresh_thread(config.index_management)


def populate_cache(index_management: IndexManagement):
    """
    Refresh cache for all indexes
    """
    global index_info_cache
    indexes = index_management.get_all_indexes()

    # Enable caching and reset any existing model caches
    # Create a map for one-pass cache update
    index_map = dict()
    for index in indexes:
        index_map[index.name] = index

    index_info_cache = index_map
