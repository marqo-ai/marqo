from typing import List, Optional, Union, Tuple

from marqo.api.exceptions import EnvVarError
from marqo.inference.inference_cache.abstract_cache import MarqoAbstractCache
from marqo.inference.inference_cache.enums import MarqoCacheType
from marqo.inference.inference_cache.marqo_lfu_cache import MarqoLFUCache
from marqo.inference.inference_cache.marqo_lru_cache import MarqoLRUCache


class MarqoInferenceCache:
    """MarqoInferenceCache is a thread-safe cache implementation for storing embeddings.

    The key is a string consisting of model_cache_key and content to identify the cache.
    The value is a list of floats representing the embeddings.
    """

    _CACHE_TYPES_MAPPING = {
        MarqoCacheType.LRU: MarqoLRUCache,
        MarqoCacheType.LFU: MarqoLFUCache,
    }

    def __init__(self, cache_size: int = 0, cache_type: Union[None, str, MarqoCacheType] = MarqoCacheType.LRU):
        self._cache = self._build_cache(cache_size, cache_type)

    def _build_cache(self, cache_size: int, cache_type: MarqoCacheType) -> Optional[MarqoAbstractCache]:
        """Return a cache instance based on the cache type and size.

        Args:
            cache_size: The maximum size of the cache.
            cache_type: The type of the cache.

        Returns:
            A cache instance based on the cache type and size. None if the cache_size is 0.

        Raises:
            EnvVarError: If the cache size or type is invalid.
        """
        if not isinstance(cache_size, int) or cache_size < 0:
            raise EnvVarError(f"Invalid cache size: {cache_size}. "
                              f"Must be a non-negative integer. "
                              f"Please set the 'MARQO_INFERENCE_CACHE_SIZE' "
                              f"environment variable to a non-negative integer.")
        elif cache_size == 0:
            return None
        elif cache_size > 0:
            if cache_type not in self._CACHE_TYPES_MAPPING:
                raise EnvVarError(f"Invalid cache type: {cache_type}. "
                                  f"Must be one of {self._CACHE_TYPES_MAPPING.keys()}."
                                  f"Please set the 'MARQO_INFERENCE_CACHE_TYPE' "
                                  f"environment variable to one of the valid cache types.")
            return self._CACHE_TYPES_MAPPING[cache_type](maxsize=cache_size)
        else:
            ValueError(f"Invalid cache size: {cache_size}.")

    def get(self, model_cache_key: str, content: str, default=None) -> Optional[List[float]]:
        key = self._generate_key(model_cache_key, content)
        return self._cache.get(key, default)

    def set(self, model_cache_key: str, content: str, value: List[float]) -> None:
        self.__setitem__(model_cache_key, content, value)

    def __getitem__(self, model_cache_key: str, content: str, key: str) -> List[float]:
        key = self._generate_key(model_cache_key, content)
        return self._cache[key]

    def __setitem__(self, model_cache_key: str, content: str, value: List[float]) -> None:
        key = self._generate_key(model_cache_key, content)
        self._cache[key] = value

    def __contains__(self, item: Tuple) -> bool:
        if len(item) != 2:
            raise ValueError("MarqoInferenceCache received an unsupported input for 'in' operation. "
                             "Expected input is a tuple with 'model-cache-key' and 'content'. "
                             "E.g., ('my-model-cache-key', 'content'). ")
        model_cache_key, content = item
        key = self._generate_key(model_cache_key, content)
        return key in self._cache

    def _generate_key(self, model_cache_key: str, content: str) -> str:
        if not isinstance(model_cache_key, str):
            raise TypeError(f"model_cache_key must be a string, not {type(model_cache_key)}")
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, not {type(content)}")
        return f"{model_cache_key}||{content}"

    def clear(self) -> None:
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()

    def is_enabled(self) -> bool:
        """Return True if the cache is enabled, else False."""
        return self._cache is not None

    @property
    def maxsize(self) -> int:
        """Return the maximum size of the cache."""
        return self._cache.maxsize

    @property
    def currsize(self) -> int:
        """Return the current size of the cache."""
        return self._cache.currsize
