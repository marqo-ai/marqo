from typing import List, Optional

from cachetools import LFUCache

from marqo.s2_inference.inference_cache.abstract_cache import MarqoAbstractCache


class MarqoLFUCache(MarqoAbstractCache):
    """Least Frequently Used (LFU) cache implementation using cachetools."""
    def __init__(self, maxsize: int):
        self.cache = LFUCache(maxsize=maxsize, )

    def get(self, key: str, default=None) -> Optional[List[float]]:
        return self.cache.get(key, default)

    def set(self, key: str, value: List[float]):
        self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __setitem__(self, key: str, value: List[float]):
        self.cache[key] = value

    def __getitem__(self, key: str) -> List[float]:
        return self.cache[key]

    def __len__(self) -> int:
        return len(self.cache)

