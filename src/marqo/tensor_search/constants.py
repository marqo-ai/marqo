from marqo.tensor_search.enums import MappingsObjectType

INDEX_NAMES_TO_IGNORE = {
    '.opendistro_security',
}
INDEX_NAME_PREFIXES_TO_IGNORE = {
    'security-auditlog-', '.kibana', '.opendistro'
}

MARQO_OBJECT_TYPES = {MappingsObjectType.multimodal_combination}

ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}

ALLOWED_CUSTOMER_FIELD_TYPES = [str, int, float, bool, list, dict]

NON_TENSORISABLE_FIELD_TYPES = [int, float, bool, list]

ALLOWED_MULTIMODAL_FIELD_TYPES = [str]

LUCENE_SPECIAL_CHARS = {
    '/', '*', '^', '\\', '!', '[', '||', '?',
    '&&', '"', ']', '-', '{', '~', '+', '}', ':', ')', '('
}

# these are chars that are not officially listed as Lucene special chars, but
# aren't treated as normal chars either
NON_OFFICIAL_LUCENE_SPECIAL_CHARS = {
    ' '
}

NUM_BYTES_IN_KB = 1024
SUPPORTED_SIZES_FOR_STATS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
