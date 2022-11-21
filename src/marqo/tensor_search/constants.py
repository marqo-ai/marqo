INDEX_NAMES_TO_IGNORE = {
    '.opendistro_security',
    '.opendistro-security'
}
INDEX_NAME_PREFIXES_TO_IGNORE = {
    'security-auditlog-'
}

ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}

ALLOWED_CUSTOMER_FIELD_TYPES = [str, int, float, bool]