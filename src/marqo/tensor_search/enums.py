from enum import Enum


class SearchMethod(str, Enum):
    # BM25/ TF-IDF:
    LEXICAL = "LEXICAL"
    # chunk_embeddings
    TENSOR = "TENSOR"


class TensorField:
    """Tensor search fields
    Protected document field names"""
    field_name = "__field_name"
    field_content = "__field_content"
    # the prefix will have the customer's field name appended to the end of it
    vector_prefix = "__vector_"
    marqo_knn_field = "__vector_marqo_knn_field"
    chunks = "__chunks"
    output_highlights = "_highlights"
    output_score = "_score"
    # output fields:
    tensor_facets = "_tensor_facets"
    embedding = "_embedding"
    found = "_found"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"

    def __str__(self):
        # To pass the str(Device.cpu) == "cpu" check in clip
        return self.value


class EnvVars:
    VESPA_CONFIG_URL = "VESPA_CONFIG_URL"
    VESPA_QUERY_URL = "VESPA_QUERY_URL"
    VESPA_DOCUMENT_URL = "VESPA_DOCUMENT_URL"
    VESPA_CONTENT_CLUSTER_NAME = "VESPA_CONTENT_CLUSTER_NAME"
    VESPA_POOL_SIZE = "VESPA_POOL_SIZE"
    MARQO_MAX_INDEX_FIELDS = "MARQO_MAX_INDEX_FIELDS"
    MARQO_MAX_DOC_BYTES = "MARQO_MAX_DOC_BYTES"
    MARQO_MAX_RETRIEVABLE_DOCS = "MARQO_MAX_RETRIEVABLE_DOCS"
    MARQO_MODELS_TO_PRELOAD = "MARQO_MODELS_TO_PRELOAD"
    MARQO_MAX_CONCURRENT_INDEX = "MARQO_MAX_CONCURRENT_INDEX"
    MARQO_MAX_CONCURRENT_SEARCH = "MARQO_MAX_CONCURRENT_SEARCH"
    MARQO_THREAD_EXPIRY_TIME = "MARQO_THREAD_EXPIRY_TIME"
    MARQO_ENABLE_THROTTLING = "MARQO_ENABLE_THROTTLING"
    MARQO_LOG_LEVEL = "MARQO_LOG_LEVEL"
    MARQO_ROOT_PATH = "MARQO_ROOT_PATH"
    MARQO_MAX_CPU_MODEL_MEMORY = "MARQO_MAX_CPU_MODEL_MEMORY"
    MARQO_MAX_CUDA_MODEL_MEMORY = "MARQO_MAX_CUDA_MODEL_MEMORY"
    MARQO_EF_CONSTRUCTION_MAX_VALUE = "MARQO_EF_CONSTRUCTION_MAX_VALUE"
    MARQO_MAX_VECTORISE_BATCH_SIZE = "MARQO_MAX_VECTORISE_BATCH_SIZE"
    MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES = "MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES"
    MARQO_MAX_DELETE_DOCS_COUNT = "MARQO_MAX_DELETE_DOCS_COUNT"
    MARQO_MAX_NUMBER_OF_REPLICAS = "MARQO_MAX_NUMBER_OF_REPLICAS"
    MARQO_DEFAULT_EF_SEARCH = "MARQO_DEFAULT_EF_SEARCH"
    MARQO_BEST_AVAILABLE_DEVICE = "MARQO_BEST_AVAILABLE_DEVICE"
    MARQO_ENABLE_BATCH_APIS = "MARQO_ENABLE_BATCH_APIS"


class RequestType:
    INDEX = "INDEX"
    SEARCH = "SEARCH"
    DELETE = "DELETE"
    CREATE = "CREATE"


class MappingsObjectType:
    multimodal_combination = "multimodal_combination"


class SearchDb:
    vespa = 'vespa'


class AvailableModelsKey:
    model = "model"
    most_recently_used_time = "most_recently_used_time"
    model_size = "model_size"


class ObjectStores:
    s3 = 's3'
    hf = 'hf'


class ModelProperties:
    auth_required = 'auth_required'
    model_location = 'model_location'


class InferenceParams:
    model_auth = "model_auth"
