# The threshold for a string to be considered short, long in unstructured Vespa.
# Users will be able to filter on short strings, but not on long strings.
SHORT_STRING_THRESHOLD = 20

UNSUPPORTED_FIELD_NAME_LIST = ["::"]

FILTER_STRING_BOOL_VALUE = ["true", "false"]

VESPA_FIELD_ID = "marqo__id"
VESPA_DOC_RELEVANCE = 'relevance'
VESPA_DOC_MATCH_FEATURES = 'matchfeatures'
VESPA_DOC_FIELDS_TO_IGNORE = {'sddocname'}