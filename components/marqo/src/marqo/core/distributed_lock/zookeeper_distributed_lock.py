from kazoo.exceptions import LockTimeout, ConnectionClosedError
from kazoo.handlers.threading import KazooTimeoutError
from kazoo.protocol.states import KazooState

from marqo.core.distributed_lock.abstract_distributed_lock import AbstractDistributedLock
from marqo.core.exceptions import BackendCommunicationError, ZookeeperLockNotAcquiredError
from marqo.logging import get_logger
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = get_logger(__name__)

_DEPLOYMENT_LOCK_PATH = "/marqo__deployment_lock"


class ZookeeperDistributedLock(AbstractDistributedLock):
    """A concrete implementation of distributed lock using Zookeeper."""

    def __init__(self, zookeeper_client: ZookeeperClient,
                 path: str,
                 acquire_timeout: float = 0,
                 ):
        """
        Initialize the deployment lock.

        Args:
            zookeeper_client: The Zookeeper client.
            path: The lock path.
            acquire_timeout: The timeout to acquire the lock.
        """
        self._zookeeper_client = zookeeper_client
        self._path = path
        self._lock = self._zookeeper_client.Lock(self._path)
        self._acquire_timeout = acquire_timeout

    def acquire(self) -> bool:
        """
        Acquire the lock. Connect to the Zookeeper server if not connected.

        Returns:
            bool: True if the lock is acquired, raise an exception otherwise.
        Raises:
            BackendCommunicationError: If the Zookeeper client cannot connect to the server.
            ZookeeperLockNotAcquiredError: If the lock cannot be acquired within the timeout period.
        """
        if self._zookeeper_client.state != KazooState.CONNECTED:
            try:
                self._zookeeper_client.start()
            except KazooTimeoutError as e:
                raise BackendCommunicationError("Marqo cannot connect to Zookeeper") from e
        try:
            acquired = self._lock.acquire(timeout=self._acquire_timeout)
            if not acquired:
                raise ZookeeperLockNotAcquiredError("Failed to acquire the lock")
            else:
                return True
        except ConnectionClosedError as e:
            raise BackendCommunicationError("Marqo cannot connect to Zookeeper") from e
        except LockTimeout:
            raise ZookeeperLockNotAcquiredError("Failed to acquire the lock")

    def release(self):
        """Release the lock and reset the lock acquired time."""
        self._lock.release()

    @property
    def is_acquired(self) -> bool:
        return self._lock.is_acquired

    def __enter__(self):
        """Enter the context manager."""
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.release()


def get_deployment_lock(zookeeper_client: ZookeeperClient, acquire_timeout: float = 0) -> ZookeeperDistributedLock:
    """
    Get a deployment lock, used to lock the index creation/deletion operations.

    Args:
        zookeeper_client: The Zookeeper client.
        acquire_timeout: The timeout to acquire the lock, in seconds. Default is 0.

    Returns:
        ZookeeperDistributedLock: The deployment lock.
    """
    return ZookeeperDistributedLock(zookeeper_client, _DEPLOYMENT_LOCK_PATH, acquire_timeout)