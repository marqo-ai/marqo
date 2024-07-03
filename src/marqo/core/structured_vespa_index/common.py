from marqo.core.models.marqo_index import DistanceMetric

FIELD_ID = "marqo__id"

FIELD_SCORE_MODIFIERS_2_8 = 'marqo__score_modifiers' # retain for backwards compatibility
FIELD_SCORE_MODIFIERS_FLOAT = 'marqo__score_modifiers_float'
FIELD_SCORE_MODIFIERS_DOUBLE_LONG = 'marqo__score_modifiers_double_long'
FIELD_VECTOR_COUNT = 'marqo__vector_count'

RANK_PROFILE_BM25 = 'bm25'
RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'
RANK_PROFILE_MODIFIERS = 'modifiers'
RANK_PROFILE_BM25_MODIFIERS = 'bm25_modifiers'
RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS = 'embedding_similarity_modifiers'

# For hybrid search
RANK_PROFILE_HYBRID_CUSTOM_SEARCHER = 'hybrid_custom_searcher'
RESULT_FIELD_RAW_TENSOR_SCORE = 'marqo__raw_tensor_score'
RESULT_FIELD_RAW_LEXICAL_SCORE = 'marqo__raw_lexical_score'

# Note field names are also used as query inputs, so make sure these reserved names have a marqo__ prefix
QUERY_INPUT_EMBEDDING = 'marqo__query_embedding'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS = 'marqo__mult_weights'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS = 'marqo__add_weights'
# Used for hybrid search (split in custom searcher)
QUERY_INPUT_FIELDS_TO_SEARCH_LEXICAL = "marqo__fields_to_search_lexical"
QUERY_INPUT_FIELDS_TO_SEARCH_TENSOR = "marqo__fields_to_search_tensor"

SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
SUMMARY_ALL_VECTOR = 'all-vector-summary'
