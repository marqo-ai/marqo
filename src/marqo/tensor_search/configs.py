from marqo.tensor_search import enums as ns_enums
from marqo.tensor_search.enums import IndexSettingsField as NsFields, EnvVars


def get_default_index_settings():
    # if new fields are added, also update index.py to take in the new params
    return {
        NsFields.index_defaults: {
            NsFields.treat_urls_and_pointers_as_images: False, # only used for models that have text and vision encoders
            NsFields.model: ns_enums.MlModel.bert,
#            NsFields.model_properties: dict(),
            NsFields.normalize_embeddings: True,
            NsFields.text_preprocessing: {
                NsFields.split_length: 2,
                NsFields.split_overlap: 0,
                NsFields.split_method: ns_enums.SplitMethod.sentence
            },
            # TODO move these into a processing dict with sub-dicts
            NsFields.image_preprocessing: {
                NsFields.patch_method: None
            },
            NsFields.ann_parameters: get_default_ann_parameters()
        },
        NsFields.number_of_shards: 5,
        NsFields.number_of_replicas : 1,
    }

def get_default_ann_parameters():
    return {
        NsFields.ann_method_name: "hnsw",
        NsFields.ann_metric: "cosinesimil",

        # `ann_engine` not exposed to customer (via index settings).
        NsFields.ann_engine: "lucene",
        NsFields.ann_method_parameters: {
            NsFields.hnsw_ef_construction: 128,
            NsFields.hnsw_m: 16
        }
    }


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
        EnvVars.MARQO_THREAD_EXPIRY_TIME: 1800,     # 30 minutes
        EnvVars.MARQO_ENABLE_THROTTLING: "TRUE",
        EnvVars.MARQO_LOG_LEVEL: "info",           # This env variable is set to "info" by default in run_marqo.sh, which overrides this value
        EnvVars.MARQO_MAX_CPU_MODEL_MEMORY: 4,
        EnvVars.MARQO_MAX_CUDA_MODEL_MEMORY: 4,  # For multi-GPU, this is the max memory for each GPU.
        EnvVars.MARQO_EF_CONSTRUCTION_MAX_VALUE: 4096,
        EnvVars.MARQO_MAX_VECTORISE_BATCH_SIZE: 16,
        EnvVars.MARQO_MAX_DELETE_DOCS_COUNT: 10000,
        EnvVars.MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES: None,
        EnvVars.MARQO_MAX_NUMBER_OF_REPLICAS: 1,
    }

