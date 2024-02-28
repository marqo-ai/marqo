from marqo.core.models.marqo_index import DistanceMetric

FIELD_ID = "marqo__id"
FIELD_SCORE_MODIFIERS = 'marqo__score_modifiers'
FIELD_VECTOR_COUNT = 'marqo__vector_count'

RANK_PROFILE_BM25 = 'bm25'
RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'
RANK_PROFILE_MODIFIERS = 'modifiers'
RANK_PROFILE_BM25_MODIFIERS = 'bm25_modifiers'
RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS = 'embedding_similarity_modifiers'

# Note field names are also used as query inputs, so make sure these reserved names have a marqo__ prefix
QUERY_INPUT_EMBEDDING = 'marqo__query_embedding'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS = 'marqo__mult_weights'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS = 'marqo__add_weights'

SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
SUMMARY_ALL_VECTOR = 'all-vector-summary'
