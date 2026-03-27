import semver

# These special characters are defined in vespa docu here: https://docs.vespa.ai/en/reference/query-language-reference.html?mode=cloud
CHARACTERS_TO_BE_ESCAPED_IN_VESPA = ('"', '\\')

MARQO_RESERVED_PREFIX = 'marqo__'
MARQO_DOC_TENSORS = 'marqo__tensors'  # doc-level so must not clash with index field names
MARQO_DOC_HIGHLIGHTS = '_highlights'  # doc-level so must not clash with index field names
MARQO_DOC_CHUNKS = 'chunks'
MARQO_DOC_EMBEDDINGS = 'embeddings'
MARQO_DOC_ID = '_id'
MARQO_FIELD_TYPES = "field_types"

MARQO_SEARCH_METHOD_TENSOR = 'tensor'
MARQO_SEARCH_METHOD_LEXICAL = 'lexical'

# For hybrid search
MARQO_DOC_HYBRID_TENSOR_SCORE = '_tensor_score'
MARQO_DOC_HYBRID_LEXICAL_SCORE = '_lexical_score'
MARQO_DOC_RECENCY_SCORE = '_recency_score'
MARQO_DOC_PRE_RERANK_SCORE = '_pre_rerank_score'

MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION = semver.VersionInfo.parse('2.10.0')
MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION = semver.VersionInfo.parse('2.11.0')
MARQO_CUSTOM_VECTOR_NORMALIZATION_MINIMUM_VERSION = semver.VersionInfo.parse('2.13.0')
MARQO_SEMI_UNSTRUCTURED_INDEX_VERSION = semver.VersionInfo.parse('2.13.0')
MARQO_GLOBAL_SCORE_MODIFIERS_MINIMUM_VERSION = semver.VersionInfo.parse('2.15.0')
MARQO_RERANK_DEPTH_MINIMUM_VERSION = semver.VersionInfo.parse('2.15.0')
MARQO_SORT_BY_MINIMUM_VERSION = semver.VersionInfo.parse('2.22.0')
MARQO_LANGUAGE_MINIMUM_VERSION = semver.VersionInfo.parse('2.16.0')
MARQO_STEMMING_MINIMUM_VERSION = semver.VersionInfo.parse('2.16.0')
MARQO_PARTIAL_UPDATE_MINIMUM_VERSION = semver.VersionInfo.parse('2.16.0')
MARQO_COLLAPSE_FIELDS_MINIMUM_VERSION = semver.VersionInfo.parse('2.23.0')
MARQO_TYPEAHEAD_SCHEMA_MINIMUM_VERSION = semver.VersionInfo.parse('2.23.0')
MARQO_UPDATE_SCHEMA_MINIMUM_VERSION = semver.VersionInfo.parse('2.23.0')
MARQO_COLLAPSE_MINIMAL_SUMMARY_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.6')
MARQO_RECENCY_SCORING_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.8')
MARQO_RECENCY_ADDITIVE_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.9')
MARQO_SECOND_PHASE_LEXICAL_SCORE_MODIFIERS_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.11')
MARQO_COLLAPSE_SORT_BY_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.13')
MARQO_CUSTOM_SCORE_RERANKERS_MINIMUM_VERSION = semver.VersionInfo.parse('2.26.0')

# For score modifiers
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_2_9 = 'marqo__mult_weights'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_2_9 = 'marqo__add_weights'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL = 'marqo__mult_weights_lexical'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL = 'marqo__add_weights_lexical'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR = 'marqo__mult_weights_tensor'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR = 'marqo__add_weights_tensor'
QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_GLOBAL = 'marqo__mult_weights_global'
QUERY_INPUT_CUSTOM_SCORE_RERANK_ADD_WEIGHTS_GLOBAL = 'marqo__custom_score_add_weights_global'
QUERY_INPUT_CUSTOM_SCORE_RERANK_MULT_WEIGHTS_GLOBAL = 'marqo__custom_score_mult_weights_global'
QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_GLOBAL = 'marqo__add_weights_global'
MARQO_GLOBAL_SCORE_MODIFIERS = 'global'
MARQO_CUSTOM_SCORE_RERANK_MODIFIERS = 'custom_score_rerank'
MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX = "marqo__score_"

# For recency scoring
QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE = 'marqo__recency_should_calculate_score'
QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE = 'marqo__recency_should_apply_score'
QUERY_INPUT_RECENCY_SCALE_SECONDS = 'marqo__recency_scale_seconds'
QUERY_INPUT_RECENCY_OFFSET_SECONDS = 'marqo__recency_offset_seconds'
QUERY_INPUT_RECENCY_DECAY_TO = 'marqo__recency_decay_to'
QUERY_INPUT_RECENCY_TIMESTAMP_KEY = 'marqo__recency_timestamp_key'
QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE = 'marqo__recency_decay_function_type'
QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT = 'marqo__recency_add_to_score_weight'

# For grow (future timestamp) scoring
QUERY_INPUT_RECENCY_GROW_ENABLED = 'marqo__recency_grow_enabled'
QUERY_INPUT_RECENCY_GROW_FROM = 'marqo__recency_grow_from'
QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE = 'marqo__recency_grow_function_type'
QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS = 'marqo__recency_grow_scale_seconds'
QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS = 'marqo__recency_grow_offset_seconds'
MARQO_RECENCY_GROW_MINIMUM_VERSION = semver.VersionInfo.parse('2.24.9')
MARQO_RECENCY_CENTER_AND_SUBQUERIES_MINIMUM_VERSION = semver.VersionInfo.parse('2.25.1')

# For recency center and subquery control
QUERY_INPUT_RECENCY_CENTER_SECONDS = 'marqo__recency_center_seconds'
QUERY_INPUT_RECENCY_APPLY_TO_TENSOR = 'marqo__recency_apply_to_tensor'
QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL = 'marqo__recency_apply_to_lexical'
