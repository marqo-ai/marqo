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

MARQO_HYBRID_SEARCH_MINIMUM_VERSION = '2.10.0'