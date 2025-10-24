FIELD_ID = "marqo__id"

FIELD_SCORE_MODIFIERS_2_8 = 'marqo__score_modifiers' # retain for backwards compatibility
FIELD_SCORE_MODIFIERS_FLOAT = 'marqo__score_modifiers_float'
FIELD_SCORE_MODIFIERS_DOUBLE_LONG = 'marqo__score_modifiers_double_long'
FIELD_VECTOR_COUNT = 'marqo__vector_count'

RANK_PROFILE_BASE = 'base_rank_profile'
RANK_PROFILE_BM25 = 'bm25'
RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'
RANK_PROFILE_MODIFIERS = 'modifiers'
RANK_PROFILE_BM25_MODIFIERS_2_9 = 'bm25_modifiers'
RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS_2_9 = 'embedding_similarity_modifiers'

# Note field names are also used as query inputs, so make sure these reserved names have a marqo__ prefix
QUERY_INPUT_EMBEDDING = 'marqo__query_embedding'
QUERY_INPUT_BM25_AGGREGATOR = 'marqo__bm25_aggregator'

# For hybrid search
RANK_PROFILE_HYBRID_CUSTOM_SEARCHER = 'hybrid_custom_searcher'
RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY = 'hybrid_bm25_then_embedding_similarity'
RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 = 'hybrid_embedding_similarity_then_bm25'

QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL = "marqo__fields_to_rank_lexical"
QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR = "marqo__fields_to_rank_tensor"

VESPA_DOC_HYBRID_RAW_TENSOR_SCORE = 'marqo__raw_tensor_score'
VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE = 'marqo__raw_lexical_score'

SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
SUMMARY_ALL_VECTOR = 'all-vector-summary'
