from enum import Enum
from fastapi_utils.enums import StrEnum


class MediaType:
    text = 'text'
    image = 'image'
    default = 'text'


class MlModel:
    bert = "hf/all_datasets_v4_MiniLM-L6"
    clip = "ViT-L/14"


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
    doc_chunk_relation = "__doc_chunk_relation"
    chunk_ids = "__chunk_ids"
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

class IndexSettingsField:
    index_settings = "index_settings"
    index_defaults = "index_defaults"
    treat_urls_and_pointers_as_images = "treat_urls_and_pointers_as_images"
    model = "model"
    model_properties = "model_properties"
    normalize_embeddings = "normalize_embeddings"

    text_preprocessing = "text_preprocessing"
    split_length = "split_length"
    split_overlap = "split_overlap"
    split_method = "split_method"

    image_preprocessing = "image_preprocessing"
    patch_method = "patch_method"

    number_of_shards = "number_of_shards"
    number_of_replicas = "number_of_replicas"

    ann_parameters = "ann_parameters"
    ann_method = "method"
    ann_method_name = "name"
    ann_metric = "space_type"
    ann_engine = "engine"
    ann_method_parameters = "parameters"

    # method_parameters keys for "method"="hnsw"
    hnsw_ef_construction = "ef_construction"
    hnsw_m = "m"


class SplitMethod:
    # consider moving this enum into processing
    sentence = "sentence"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    def __str__(self):
        # To pass the str(Device.cpu) == "cpu" check in clip
        return self.value


class OpenSearchDataType:
    text = "text"
    keyword = "keyword"
    int = "int"
    float = "float"
    integer = "integer"
    to_be_defined = "to_be_defined"  # to be defined by OpenSearch


class EnvVars:
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
    MARQO_BEST_AVAILABLE_DEVICE = "MARQO_BEST_AVAILABLE_DEVICE"
    MARQO_MAX_ADD_DOCS_COUNT = "MARQO_MAX_ADD_DOCS_COUNT"


class RequestType:
    INDEX = "INDEX"
    SEARCH = "SEARCH"
    DELETE = "DELETE"
    CREATE = "CREATE"


class MappingsObjectType:
    multimodal_combination = "multimodal_combination"
    custom_vector = "custom_vector"


class SearchDb:
    opensearch = 'opensearch'


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
# Perhaps create a ThrottleType to differentiate thread_count and data_size throttling mechanisms


class HealthStatuses(str, Enum):
    green = "green"
    yellow = "yellow"
    red = "red"

    def _status_index(self):
        status_order = [self.green, self.yellow, self.red]
        return status_order.index(self)

    def __gt__(self, other):
        return self._status_index() > other._status_index()

    def __lt__(self, other):
        return self._status_index() < other._status_index()
