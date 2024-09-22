import semver

MARQO_RESERVED_PREFIX = 'marqo__'
MARQO_DOC_TENSORS = 'marqo__tensors'  # doc-level so must not clash with index field names
MARQO_DOC_HIGHLIGHTS = '_highlights'  # doc-level so must not clash with index field names
MARQO_DOC_CHUNKS = 'chunks'
MARQO_DOC_EMBEDDINGS = 'embeddings'
MARQO_DOC_ID = '_id'

MARQO_SEARCH_METHOD_TENSOR = 'tensor'
MARQO_SEARCH_METHOD_LEXICAL = 'lexical'

# For hybrid search
MARQO_DOC_HYBRID_TENSOR_SCORE = '_tensor_score'
MARQO_DOC_HYBRID_LEXICAL_SCORE = '_lexical_score'

MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION = semver.VersionInfo.parse('2.10.0')
MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION = semver.VersionInfo.parse('2.11.0')
MARQO_CUSTOM_VECTOR_NORMALIZATION_MINIMUM_VERSION = semver.VersionInfo.parse('2.13.0')

# For score modifiers
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_2_9 = 'marqo__mult_weights'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_2_9 = 'marqo__add_weights'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL = 'marqo__mult_weights_lexical'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL = 'marqo__add_weights_lexical'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR = 'marqo__mult_weights_tensor'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR = 'marqo__add_weights_tensor'
