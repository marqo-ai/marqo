MARQO_RESERVED_PREFIX = 'marqo__'
MARQO_DOC_TENSORS = 'marqo__tensors'  # doc-level so must not clash with index field names
MARQO_DOC_HIGHLIGHTS = '_highlights'  # doc-level so must not clash with index field names
MARQO_DOC_CHUNKS = 'chunks'
MARQO_DOC_EMBEDDINGS = 'embeddings'
MARQO_DOC_ID = '_id'

MARQO_SEARCH_METHOD_TENSOR = 'tensor'
MARQO_SEARCH_METHOD_LEXICAL = 'lexical'

# For hybrid search
MARQO_RAW_TENSOR_SCORE = '_raw_tensor_score'
MARQO_RAW_LEXICAL_SCORE = '_raw_lexical_score'

MARQO_HYBRID_SEARCH_MINIMUM_VERSION = '2.10.0'