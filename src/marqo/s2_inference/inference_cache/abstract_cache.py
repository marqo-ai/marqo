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
        """Return the value for key if key is in the cache, else default."""
        pass

    @abstractmethod
    def set(self, key: str, value: List[float]):
        """Set the value for key in the cache."""
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the cache, else False."""
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: List[float]):
        """Set the value for key in the cache if the cache is not full, else popitem() until there is enough room."""
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> List[float]:
        """Return the value for key if key is in the cache, else raise KeyError."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of items in the cache."""
        pass