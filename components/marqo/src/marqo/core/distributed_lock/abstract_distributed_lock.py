from abc import ABC, abstractmethod

from marqo.logging import get_logger

logger = get_logger(__name__)


class AbstractDistributedLock(ABC):
    """Abstract class for a distributed lock.

    This lock is used to ensure that a resource is not accessed concurrently by multiple processes.
    The lock should also have __enter__ and __exit__ methods to be used as a context manager.
    """

    @abstractmethod
    def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            bool: True if the lock is acquired, False otherwise.

        Raises:
            marqo.core.exceptions.BackendCommunicationError: If the Zookeeper client cannot connect to the server.
            marqo.core.exceptions.ZookeeperLockNotAcquiredError: If the lock cannot be acquired within the timeout period.
        """
        pass

    @abstractmethod
    def release(self):
        """Release the lock."""
        pass

    @property
    @abstractmethod
    def is_acquired(self) -> bool:
        """Check if the lock is acquired.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """Enter the context manager."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass
