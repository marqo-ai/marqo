from marqo.tensor_search.enums import EnvVars


def default_env_vars() -> dict:
    """Returns a dict of default env vars.
    This is used by utils.read_env_vars_and_defaults() as the source for
    default env vars if they aren't defined in the environment.
    """
    return {
        EnvVars.VESPA_CONFIG_URL: "http://localhost:19071",
        EnvVars.VESPA_QUERY_URL: "http://localhost:8080",
        EnvVars.VESPA_DOCUMENT_URL: "http://localhost:8080",
        EnvVars.ZOOKEEPER_HOSTS: None,
        EnvVars.VESPA_CONTENT_CLUSTER_NAME: "content_default",
        EnvVars.VESPA_POOL_SIZE: 10,
        EnvVars.VESPA_FEED_POOL_SIZE: 10,
        EnvVars.VESPA_GET_POOL_SIZE: 10,
        EnvVars.VESPA_DELETE_POOL_SIZE: 10,
        EnvVars.VESPA_PARTIAL_UPDATE_POOL_SIZE: 10,
        EnvVars.VESPA_SEARCH_TIMEOUT_MS: 1000,
        EnvVars.MARQO_MAX_INDEX_FIELDS: None,
        EnvVars.MARQO_MAX_DOC_BYTES: 100000,
        EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: 10000,
        EnvVars.MARQO_MAX_SEARCH_LIMIT: 1000,
        EnvVars.MARQO_MAX_SEARCH_OFFSET: 10000,
        EnvVars.MARQO_MODELS_TO_PRELOAD: ["hf/e5-base-v2", "open_clip/ViT-B-32/laion2b_s34b_b79k"],
        EnvVars.MARQO_MAX_CONCURRENT_INDEX: 8,
        EnvVars.MARQO_MAX_CONCURRENT_SEARCH: 8,
        EnvVars.MARQO_THREAD_EXPIRY_TIME: 1800,  # 30 minutes
        EnvVars.MARQO_ENABLE_THROTTLING: "TRUE",
        EnvVars.MARQO_LOG_LEVEL: "info",
        EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST: 5,
        EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST: 20,
        # This env variable is set to "info" by default in run_marqo.sh, which overrides this value
        EnvVars.MARQO_MAX_CPU_MODEL_MEMORY: 4,
        EnvVars.MARQO_MAX_CUDA_MODEL_MEMORY: 4,  # For multi-GPU, this is the max memory for each GPU.
        EnvVars.MARQO_EF_CONSTRUCTION_MAX_VALUE: 4096,
        EnvVars.MARQO_MAX_VECTORISE_BATCH_SIZE: 16,
        EnvVars.MARQO_MAX_DELETE_DOCS_COUNT: 10000,
        EnvVars.MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES: None,
        EnvVars.MARQO_MAX_NUMBER_OF_REPLICAS: 1,
        EnvVars.MARQO_DEFAULT_EF_SEARCH: 2000,
        EnvVars.MARQO_ENABLE_BATCH_APIS: "FALSE",
        EnvVars.MARQO_ENABLE_UPGRADE_API: "FALSE",
        EnvVars.MARQO_ENABLE_DEBUG_API: "FALSE",
        EnvVars.MARQO_ENABLE_OPS_API: "FALSE",
        EnvVars.MARQO_MAX_CONCURRENT_PARTIAL_UPDATE: 100,
        EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE: 128,
        EnvVars.MARQO_INFERENCE_CACHE_SIZE: 0,
        EnvVars.MARQO_INFERENCE_CACHE_TYPE: "LRU",
        EnvVars.MARQO_BEST_AVAILABLE_DEVICE: "cpu",      # on_start_script will determine this.
        EnvVars.MARQO_MAX_TENSOR_FIELD_COUNT_UNSTRUCTURED: 100,
        EnvVars.MARQO_MAX_LEXICAL_FIELD_COUNT_UNSTRUCTURED: 100,
        EnvVars.ZOOKEEPER_CONNECTION_TIMEOUT: 15,
    }
