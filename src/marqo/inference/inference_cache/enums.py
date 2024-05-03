from enum import Enum


class MarqoCacheType(str, Enum):
    LRU = "LRU"
    LFU = "LFU"
