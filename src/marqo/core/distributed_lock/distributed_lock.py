import threading
import time
from contextlib import contextmanager
from typing import Optional

from kazoo.recipe.lock import Lock
from marqo.core.exceptions import MarqoError
from kazoo.client import KazooClient
from marqo.logging import get_logger
from kazoo.exceptions import LockTimeout, ConnectionClosedError


logger = get_logger(__name__)


class DistributedLock:
    """A distributed lock on Zookeeper with watch dog that expires after a certain period of time.

    Args:
        client: The Zookeeper client.
        path: The lock path.
        max_lock_period: The maximum period of time the lock can be held.
        watchdog_interval: The interval at which the watchdog checks the lock status.
        acquire_timeout: The timeout to acquire the lock.
    """

    def __init__(self, client: KazooClient, path: str, max_lock_period: int,
                 watchdog_interval: int = 5, acquire_timeout: int = 1):
        self.lock = Lock(client, path)
        self.max_lock_period = max_lock_period
        self.acquire_timeout = acquire_timeout
        self.lock_acquired_time = None
        self.watchdog_interval = watchdog_interval
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()

    def _release_lock(self):
        """Internal method to release the lock."""
        if self.lock.is_acquired:
            self.lock.release()
            logger.warning("Lock released by expiration timer.")

    def _watchdog(self):
        """Watchdog thread to monitor lock status and release the lock if it is held for too long."""
        while True:
            if self.lock_acquired_time and (time.time() - self.lock_acquired_time) > self.max_lock_period:
                self._release_lock()
            time.sleep(self.watchdog_interval)

    def acquire(self, acquire_timeout: Optional[int] = None) -> bool:
        """Acquire the lock.
        Args:
            acquire_timeout: The timeout to acquire the lock. If not provided, we use the class object's timeout.

        Returns:
            bool: True if the lock is acquired, False otherwise.
        """
        try:
            acquired = self.lock.acquire(timeout=self.acquire_timeout if acquire_timeout is None else acquire_timeout)
        except LockTimeout:
            return False
        if acquired:
            self.lock_acquired_time = time.time()
        return acquired

    def release(self):
        """Release the lock and reset the lock acquired time."""
        if self.lock.is_acquired:
            self.lock.release()
        self.lock_acquired_time = None


@contextmanager
def acquire_lock(lock: Optional[DistributedLock], error: MarqoError, acquire_timeout: Optional[float] = None) -> None:
    """Context manager to acquire and release a lock and handle exceptions.

    If the lock is not acquired, we raise the given error.
    If the lock is acquired, we release the lock after the context manager exits.
    If the connection to Zookeeper is closed, we log a warning and proceed without acquiring the lock.

    Args:
        lock: The lock to acquire. If None, no lock is acquired.
        error: The error to raise if the lock cannot be acquired.
            Must be a subclass of MarqoError.
        acquire_timeout: The timeout to acquire the lock.
            If None, we use the default timeout in the lock
    """
    if lock:
        try:
            if not lock.acquire(acquire_timeout):
                raise error
        except ConnectionClosedError:
            logger.warning("Zookeeper connection closed when trying to acquire lock. "
                           "Skipping lock acquisition and proceeding. "
                           "Marqo may not be protected by Zookeeper. "
                           "Please check your Zookeeper configuration and network settings.")
            pass
    try:
        yield
    finally:
        if lock:
            lock.release()
