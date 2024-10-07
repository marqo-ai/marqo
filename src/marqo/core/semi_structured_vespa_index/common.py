VESPA_FIELD_ID = "marqo__id"
STRINGS = "marqo__strings"
SHORT_STRINGS_FIELDS = "marqo__short_string_fields"
LONGS_STRINGS_FIELDS = "marqo__long_string_fields"
STRING_ARRAY = "marqo__string_array"

FIELD_VECTOR_COUNT = 'marqo__vector_count'

INT_FIELDS = "marqo__int_fields"
FLOAT_FIELDS = "marqo__float_fields"
BOOL_FIELDS = "marqo__bool_fields"

SCORE_MODIFIERS = "marqo__score_modifiers"

RANK_PROFILE_EMBEDDING_SIMILARITY = "embedding_similarity"
RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS = 'embedding_similarity_modifiers'
RANK_PROFILE_BM25 = "bm25"
RANK_PROFILE_BM25_MODIFIERS = "bm25_modifiers"

MARQO_DOC_MULTIMODAL_PARAMS = "multimodal_params"
VESPA_DOC_MULTIMODAL_PARAMS = "marqo__multimodal_params"

SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
SUMMARY_ALL_VECTOR = 'all-vector-summary'

VESPA_DOC_CHUNKS = "marqo__chunks"
VESPA_DOC_EMBEDDINGS = "marqo__embeddings"

VESPA_DOC_MATCH_FEATURES = 'matchfeatures'

# TODO: Reorganize these later
RANK_PROFILE_BASE = 'base_rank_profile'
RANK_PROFILE_BM25 = 'bm25'
RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'
RANK_PROFILE_MODIFIERS = 'modifiers'
RANK_PROFILE_BM25_MODIFIERS_2_9 = 'bm25_modifiers'
RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS_2_9 = 'embedding_similarity_modifiers'

# Note field names are also used as query inputs, so make sure these reserved names have a marqo__ prefix
QUERY_INPUT_EMBEDDING_2_10 = 'embedding_query'      # Keep for backwards compatibility
QUERY_INPUT_EMBEDDING = "marqo__query_embedding"    # TODO: see if this change from 'embedding_query' to 'embedding_query' changes anything
QUERY_INPUT_BM25_AGGREGATOR = 'marqo__bm25_aggregator'

# For hybrid search
RANK_PROFILE_HYBRID_CUSTOM_SEARCHER = 'hybrid_custom_searcher'
RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY = 'hybrid_bm25_then_embedding_similarity'
RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 = 'hybrid_embedding_similarity_then_bm25'

QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL = "marqo__fields_to_rank_lexical"
QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR = "marqo__fields_to_rank_tensor"

VESPA_DOC_HYBRID_RAW_TENSOR_SCORE = 'marqo__raw_tensor_score'
VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE = 'marqo__raw_lexical_score'