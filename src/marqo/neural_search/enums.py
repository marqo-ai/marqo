class MediaType:
    text = 'text'
    image = 'image'
    default = 'text'

class Normalization:
    normalize = True 



class MlModel:
    bert = "hf/all_datasets_v4_MiniLM-L6"
    clip = "ViT-L/14"

class SearchMethod:
    # BM25/ TF-IDF:
    LEXICAL = "LEXICAL"
    # chunk_embeddings
    NEURAL = "NEURAL"

class NeuralField:
    """Neural Search fields
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

class NeuralSettingsField:
    neural_settings = "neural_settings"
    index_defaults = "index_defaults"
    treat_urls_and_pointers_as_images = "treat_urls_and_pointers_as_images"
    model = "model"
    normalize_embeddings = "normalize_embeddings"

    text_preprocessing = "text_preprocessing"
    split_length = "split_length"
    split_overlap = "split_overlap"
    split_method = "split_method"


class SplitMethod:
    # consider moving this enum into processing
    sentence = "sentence"

