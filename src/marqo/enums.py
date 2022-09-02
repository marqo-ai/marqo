"""enums to be used by consumers of the Marqo client"""


class SearchMethods:
    LEXICAL = "LEXICAL"
    TENSOR = "TENSOR"


class Devices:
    cpu = "cpu"
    cuda = "cuda"

