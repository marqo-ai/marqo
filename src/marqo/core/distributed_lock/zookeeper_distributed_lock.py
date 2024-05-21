from kazoo.exceptions import LockTimeout, ConnectionClosedError
from kazoo.handlers.threading import KazooTimeoutError
from kazoo.protocol.states import KazooState

from marqo.core.distributed_lock.abstract_distributed_lock import AbstractDistributedLock
from marqo.vespa.marqo_zookeeper_client import MarqoZookeeperClient
from marqo.core.exceptions import BackendCommunicationError
from marqo.logging import get_logger

logger = get_logger(__name__)


class ZookeeperDistributedLock(AbstractDistributedLock):
    """A concrete implementation of distributed lock using Zookeeper."""

    def __init__(self, zookeeper_client: MarqoZookeeperClient,
                 path: str,
                 acquire_timeout: float = 1,
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
        Acquire the lock.

        Returns:
            bool: True if the lock is acquired, False otherwise.
        Raises:
            BackendCommunicationError: If the Zookeeper client cannot connect to the server.
        """
        if self._zookeeper_client.state != KazooState.CONNECTED:
            try:
                self._zookeeper_client.start()
            except KazooTimeoutError as e:
                raise BackendCommunicationError("Marqo cannot connect to backend concurrent manager. "
                                                "Your request cannot be processed at this time. "
                                                "Please check your network settings and try again later.") from e
        try:
            self._lock.acquire(timeout=self._acquire_timeout)
        except ConnectionClosedError:
            raise BackendCommunicationError("Marqo cannot connect to backend concurrent manager "
                                            "when acquiring the lock. "
                                            "Your request cannot be processed at this time. "
                                            "Please check your network settings and try again later.")
        except LockTimeout:
            return False
        return self._lock.is_acquired

    def release(self):
        """Release the lock and reset the lock acquired time."""
        if self.is_acquired:
            self._lock.release()

    @property
    def is_acquired(self) -> bool:
        return self._lock.is_acquired

    def __enter__(self):
        """Enter the context manager."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.release()