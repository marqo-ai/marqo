import time
from typing import Optional

from kazoo.client import KazooClient
from kazoo.exceptions import LockTimeout, ConnectionClosedError
from contextlib import contextmanager
from kazoo.protocol.states import KazooState

from marqo.core.distributed_lock.abstract_distributed_lock import AbstractExpiringDistributedLock
from marqo.core.exceptions import ConflictError, BackendCommunicationError
from marqo.logging import get_logger
from kazoo.handlers.threading import KazooTimeoutError

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
        if self.is_acquired:
            self.lock.release()
        self.lock_acquired_time = None

    @property
    def is_acquired(self) -> bool:
        return self.lock.is_acquired


@contextmanager
def acquire_deployment_lock(lock: Optional[DeploymentLock] = None, acquire_timeout: Optional[float] = None,
                            conflict_error_message: Optional[str] = None):
    """Acquire the deployment lock using a context manager.

    Args:
        lock: The deployment lock object.
        acquire_timeout: The timeout to acquire the lock. If not provided, we use the class object's timeout.
        conflict_error_message: The error message to display when the lock is not acquired. If not provided, we use the
            class object's error message.

    Raises:
        BackendCommunicationError: If the Zookeeper client cannot connect to the server.
        ConflictError: If the lock is not acquired.
    """
    if lock is None:
        raise RuntimeError("Deployment lock is not instantiated and cannot be used for index creation/deletion")
    elif lock.zookeeper_client.state != KazooState.CONNECTED:
        try:
            lock.zookeeper_client.start(5)
        except KazooTimeoutError as e:
            raise BackendCommunicationError("Marqo cannot connect to backend concurrent manager. "
                                            "Your request cannot be processed at this time. "
                                            "Please check your network settings and try again later.") from e
    else:
        try:
            if not lock.acquire(acquire_timeout):
                raise ConflictError(conflict_error_message if conflict_error_message else lock.error_message)
        except ConnectionClosedError:
            raise BackendCommunicationError("Marqo cannot connect to backend concurrent manager "
                                            "when acquiring the lock. "
                                            "Your request cannot be processed at this time. "
                                            "Please check your network settings and try again later.")
    try:
        yield
    finally:
        if lock:
            lock.release()
