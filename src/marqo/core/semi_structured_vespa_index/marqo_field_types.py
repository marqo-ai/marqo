from enum import Enum


class MarqoFieldTypes(Enum):
    """
    Enum class for Marqo field types. Used to specify the type of field in a Marqo index.
    """
    BOOL = 'bool'
    INT_MAP = 'int_map_entry'
    FLOAT_MAP = 'float_map_entry'
    INT = 'int'
    FLOAT = 'float'
    STRING_ARRAY = 'string_array'
    STRING = 'string'
    TENSOR = 'tensor'
