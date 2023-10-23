from marqo.tensor_search.enums import MappingsObjectType

INDEX_NAMES_TO_IGNORE = {
    '.opendistro_security', 'bulk'
}
INDEX_NAME_PREFIXES_TO_IGNORE = {
    'security-auditlog-', '.kibana', '.opendistro'
}

MARQO_OBJECT_TYPES = {MappingsObjectType.multimodal_combination, MappingsObjectType.custom_vector}

ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}

ALLOWED_CUSTOMER_FIELD_TYPES = [str, int, float, bool, list, dict]

NON_TENSORISABLE_FIELD_TYPES = [int, float, bool, list]

ALLOWED_MULTIMODAL_FIELD_TYPES = [str]

# Must be written in JSON schema type form: https://json-schema.org/understanding-json-schema/reference/type
ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES = ["string"]

LUCENE_SPECIAL_CHARS = {
    '/', '*', '^', '\\', '!', '[', '||', '?',
    '&&', '"', ']', '-', '{', '~', '+', '}', ':', ')', '('
}

# these are chars that are not officially listed as Lucene special chars, but
# aren't treated as normal chars either
NON_OFFICIAL_LUCENE_SPECIAL_CHARS = {
    ' '
}

# Order is important here: transient, persistent, then defaults.
OPENSEARCH_CLUSTER_SETTINGS_TYPES = ["transient", "persistent", "defaults"]

# To convert from {key} to bytes, we multiply by 1024 ** {value}
# eg. 10kb = 10 * (1024 ** 1) bytes, 5mb = 5 * (1024 ** 2) bytes
BYTE_SUFFIX_EXPONENTS = {
    'b': 0, 
    'kb': 1, 
    'mb': 2, 
    'gb': 3, 
    'tb': 4, 
    'pb': 5, 
    'eb': 6, 
    'zb': 7, 
    'yb': 8
}

DEFAULT_MARQO_MAX_BACKEND_RETRY_ATTEMPTS = 0
DEFAULT_MARQO_MAX_BACKEND_RETRY_BACKOFF = 1