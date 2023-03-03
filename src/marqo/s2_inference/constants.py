from marqo.tensor_search.utils import read_env_vars_and_defaults

MODEL_TYPE_SIZE_MAPPING = {
    # A mapping to estimate the size of the
    # model (GB) based on type.
    "open_clip" : 1,
    "clip" : 1,
    "sbert" : 0.7,
}

DEFAULT_MODEL_SIZE = 0.5
MARQO_MAX_CPU_MODEL_MEMORY = read_env_vars_and_defaults('MARQO_MAX_CPU_MODEL_MEMORY')
MARQO_MAX_CUDA_MODEL_MEMORY = read_env_vars_and_defaults('MARQO_MAX_CUDA_MODEL_MEMORY')

