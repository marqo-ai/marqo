from marqo.tensor_search.enums import EnvVars


def default_env_vars() -> dict:
    """Returns a dict of default env vars.
    This is used by utils.read_env_vars_and_defaults() as the source for
    default env vars if they aren't defined in the environment.
    """
    return {
        EnvVars.MARQO_MAX_INDEX_FIELDS: None,
        EnvVars.MARQO_MAX_DOC_BYTES: 100000,
        EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: 10000,
        EnvVars.MARQO_MODELS_TO_PRELOAD: ["hf/all_datasets_v4_MiniLM-L6", "ViT-L/14"],
        EnvVars.MARQO_MAX_CONCURRENT_INDEX: 8,
        EnvVars.MARQO_MAX_CONCURRENT_SEARCH: 8,
        EnvVars.MARQO_THREAD_EXPIRY_TIME: 1800,  # 30 minutes
        EnvVars.MARQO_ENABLE_THROTTLING: "TRUE",
        EnvVars.MARQO_LOG_LEVEL: "info",
        # This env variable is set to "info" by default in run_marqo.sh, which overrides this value
        EnvVars.MARQO_MAX_CPU_MODEL_MEMORY: 4,
        EnvVars.MARQO_MAX_CUDA_MODEL_MEMORY: 4,  # For multi-GPU, this is the max memory for each GPU.
        EnvVars.MARQO_EF_CONSTRUCTION_MAX_VALUE: 4096,
        EnvVars.MARQO_MAX_VECTORISE_BATCH_SIZE: 16,
        EnvVars.MARQO_MAX_DELETE_DOCS_COUNT: 10000,
        EnvVars.MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES: None,
        EnvVars.MARQO_MAX_NUMBER_OF_REPLICAS: 1,
    }
