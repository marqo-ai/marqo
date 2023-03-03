from marqo.tensor_search.utils import read_env_vars_and_defaults

MODEL_TYPE_SIZE_MAPPING = {
    # A mapping to estimate the size of the
    # model (GB) based on type.
    "open_clip" : 1,
    "clip" : 1,
    "sbert" : 0.7,
    "random" : 0.1,
}

# Set this to be a special number so we can tell the model size is from default
DEFAULT_MODEL_SIZE = 0.66

MARQO_MAX_CPU_MODEL_MEMORY = read_env_vars_and_defaults('MARQO_MAX_CPU_MODEL_MEMORY')
MARQO_MAX_CUDA_MODEL_MEMORY = read_env_vars_and_defaults('MARQO_MAX_CUDA_MODEL_MEMORY')

