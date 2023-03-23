from marqo.tensor_search.utils import read_env_vars_and_defaults

# Some constants to help decide the model size.
# Priorities: MODEL_NAME > MODEL_TYPE > DEFAULT
MODEL_TYPE_SIZE_MAPPING = {
    # This mapping represents the default sizes for models of their types.
    "open_clip" : 1,
    "clip" : 1,
    "sbert" : 0.7,
    "random" : 0.1,
    "multilingual_clip" : 5,
    "clip_onnx":1,
    'sbert_onnx' : 0.7,
}
MODEL_NAME_SIZE_MAPPING = {
    # This mapping represents the default sizes for the precise model names.
    "vit-l-14" : 1.5,
    "vit-g" : 5,
    "vit-h" : 5,
    "vit-bigg-14":6,
}
# Set this to be a special number so we can tell the model size is from default
DEFAULT_MODEL_SIZE = 0.66

