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
    chunks = "__chunks"
    output_highlights = "_highlights"
    output_score = "_score"

class IndexSettingsField:
    index_settings = "index_settings"
    index_defaults = "index_defaults"
    treat_urls_and_pointers_as_images = "treat_urls_and_pointers_as_images"
    model = "model"
    normalize_embeddings = "normalize_embeddings"

    text_preprocessing = "text_preprocessing"
    split_length = "split_length"
    split_overlap = "split_overlap"
    split_method = "split_method"

    image_preprocessing = "image_preprocessing"
    patch_method = "patch_method"

    number_of_shards = "number_of_shards"


class SplitMethod:
    # consider moving this enum into processing
    sentence = "sentence"


class Device (str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class OpenSearchDataType:
    text = "text"
    keyword = "keyword"
    int = "int"
    float = "float"
    integer = "integer"
    to_be_defined = "to_be_defined"  # to be defined by OpenSearch
