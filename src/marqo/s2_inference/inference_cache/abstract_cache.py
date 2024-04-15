from abc import ABC, abstractmethod
from typing import List


class MarqoAbstractCache(ABC):
    """Abstract class for Marqo cache implementations.

    The acceptable key is a string consisting of model_cache_key and content to identify the cache.
    The acceptable value is a list of floats representing the embeddings.

    When a cache is full, self.__setitem__() calls self.popitem() repeatedly
    until there is enough room for the item to be added.
    """
    @abstractmethod
    def get(self, key: str, default=None) -> List[float]:
        """Return the value for key if key is in the cache, else default.
        Args:
            key: __description__
            default: __description__
        Returns:
            __description__
        """
        pass

    @abstractmethod
    def set(self, key: str, value: List[float]) -> None:
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
    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the cache, else False.

        Args:
            key: __description__
        Returns:
            If the key is in the cache, return True. Otherwise, return False.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: List[float]) -> None:
        """Set the value for key in the cache if the cache is not full, else popitem() until there is enough room.

        Args:
            key: __description__
            value: __description__
        """
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> List[float]:
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