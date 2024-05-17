import time
from contextlib import contextmanager
from typing import Optional

from kazoo.client import KazooClient
from kazoo.exceptions import LockTimeout, ConnectionClosedError

from marqo.core.distributed_lock.abstract_distributed_lock import AbstractExpiringDistributedLock
from marqo.core.exceptions import ConflictError
from marqo.logging import get_logger

logger = get_logger(__name__)


class DeploymentLock(AbstractExpiringDistributedLock):
    """A concrete implementation of an expiring distributed lock for deployment.

    This lock is used to lock the deployment process, namely the index creation and deletion process.
    """

    def __init__(self, zookeeper_client: KazooClient, path: str, max_lock_period: float = 120,
                 watchdog_interval: float = 5,
                 acquire_timeout: float = 1,
                 error_message: str = "Conflict error. Another deployment is in progress."):
        """
        Initialize the deployment lock.

        Args:
            zookeeper_client: The Zookeeper client.
            path: The lock path.
            max_lock_period: The maximum period of time the lock can be held.
            watchdog_interval: The interval at which the watchdog checks the lock status.
            acquire_timeout: The timeout to acquire the lock.
            error_message: The error message to display when the lock is not acquired.
        """
        super().__init__(zookeeper_client, path, max_lock_period, watchdog_interval)
        self.acquire_timeout = acquire_timeout
        self.error_message = error_message

    def _watchdog(self):
        """Watchdog thread to monitor lock status and release the lock if it is held for too long."""
        while True:
            if self.lock_acquired_time and (time.time() - self.lock_acquired_time) > self.max_lock_period:
                self.release()
                logger.warning(f"Lock released by expiration timer. {self.error_message}")
            time.sleep(self.watchdog_interval)

    def acquire(self, acquire_timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock.
        Args:
            acquire_timeout: The timeout to acquire the lock. If not provided, we use the class object's timeout.

        Returns:
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

    def is_acquired(self) -> bool:
        return self.lock.is_acquired


@contextmanager
def acquire_deployment_lock(lock: Optional[DeploymentLock] = None, acquire_timeout: Optional[float] = None,
                            error_message: Optional[float] = None) -> None:
    """Context manager to acquire and release a lock and handle exceptions. We use a custom context manager to handle
    the scenario where the Zookeeper connection is closed.

    If the lock is not acquired, we raise the given error.
    If the lock is acquired, we release the lock after the context manager exits.
    If the connection to Zookeeper is closed, we log a warning and proceed without acquiring the lock.

    Args:
        lock: The lock to acquire. If None, no lock is acquired.
        acquire_timeout: The timeout to acquire the lock.
            If None, we use the default timeout in the lock
        error_message: The error message to display when the lock is not acquired.
            If None, we use the default error from the lock.
    """
    if lock:
        try:
            if not lock.acquire(acquire_timeout):
                raise ConflictError(lock.error_message if error_message is None else error_message)
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