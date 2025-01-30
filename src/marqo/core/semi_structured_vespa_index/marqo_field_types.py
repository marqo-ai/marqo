from enum import Enum


class MarqoFieldTypes(Enum):
    BOOL = 'bool'
    INT_MAP = 'int_map'
    FLOAT_MAP = 'float_map'
    INT = 'int'
    FLOAT = 'float'
    STRING_ARRAY = 'string_array'
    STRING = 'string'
    TENSOR = 'tensor'
