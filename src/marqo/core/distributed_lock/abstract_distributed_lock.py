import threading
from abc import ABC, abstractmethod

from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock

from marqo.logging import get_logger

logger = get_logger(__name__)


class AbstractExpiringDistributedLock(ABC):
    """Abstract class for a distributed lock with expiration.

    This lock is used to ensure that a resource is not accessed concurrently by multiple processes. It must also have a
    watchdog that releases the lock if it is held for too long.
    The lock should also have __enter__ and __exit__ methods to be used as a context manager.
    """

    @abstractmethod
    def __init__(self, zookeeper_client: KazooClient, path: str, max_lock_period: float, watchdog_interval: float):
        """Initialize the distributed lock.
        Args:
            zookeeper_client: The Zookeeper client.
            path: The lock path.
            max_lock_period: The maximum period of time the lock can be held.
            watchdog_interval: The interval at which the watchdog checks the lock status.
        """
        self.zookeeper_client = zookeeper_client
        self.lock = Lock(zookeeper_client, path)
        self.max_lock_period = max_lock_period
        self.watch_dog_interval = watchdog_interval
        self.lock_acquired_time = None
        self.watchdog_interval = watchdog_interval
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()

    @abstractmethod
    def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            bool: True if the lock is acquired, False otherwise.
        """
        pass

    @abstractmethod
    def release(self):
        """Release the lock."""
        pass

    @property
    @abstractmethod
    def is_acquired(self) -> bool:
        """Check if the lock is acquired."""
        pass

    @abstractmethod
    def _watchdog(self):
        """Watchdog thread to monitor lock status and release the lock if it is held for too long."""
        pass