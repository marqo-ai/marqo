from abc import ABC, abstractmethod
from typing import Any, Hashable


class MarqoAbstractCache(ABC):
    """Abstract class for Marqo cache implementations, MUST be thread-safe.

    The acceptable key must be Hashable.
    The acceptable value is Any.

    When a cache is full, self.__setitem__() calls self.popitem() repeatedly
    until there is enough room for the item to be added.

    The cache MUST be a thread-safe implementation.
    """

    @abstractmethod
    def get(self, key: Hashable, default=None) -> Any:
        """Return the value for key if key is in the cache, else default.
        Args:
            key: __description__
            default: __description__
        Returns:
            __description__
        """
        pass

    @abstractmethod
    def set(self, key: Hashable, value: Any) -> None:
        """Set the value for key in the cache.

        Args:
            key: __description__
            value: __description__
        """
        pass

    @abstractmethod
    def popitem(self) -> None:
        """Remove an item from the cache according to the defined eviction policy. The item is not returned.

        Raises:
            Exception: If there is an issue in removing an item (e.g., cache is already empty).
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the cache."""
        pass

    @abstractmethod
    def __contains__(self, key: Hashable) -> bool:
        """Return True if the key is in the cache, else False.

        Args:
            key: __description__
        Returns:
            If the key is in the cache, return True. Otherwise, return False.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Set the value for key in the cache if the cache is not full, else popitem() until there is enough room.

        Args:
            key: __description__
            value: __description__
        """
        pass

    @abstractmethod
    def __getitem__(self, key: Hashable) -> Any:
        """Return the value for key if key is in the cache, else raise KeyError.

        Args:
            key: __description__

        Raises:
            KeyError: If the key is not in the cache.

        Returns:
            __description__
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        pass

    @property
    @abstractmethod
    def maxsize(self) -> int:
        """Return the maximum size of the cache."""
        pass

    @property
    @abstractmethod
    def currsize(self) -> int:
        """Return the current size of the cache."""
        pass
