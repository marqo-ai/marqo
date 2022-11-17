from marqo.tensor_search import enums as ns_enums
from marqo.tensor_search.enums import IndexSettingsField as NsFields, EnvVars
from torch import multiprocessing as mp


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
            }
        },
        NsFields.number_of_shards: 5
    }


def default_env_vars() -> dict:

    return {
        EnvVars.MARQO_MAX_INDEX_FIELDS: None,
        EnvVars.MARQO_MAX_DOC_BYTES: 100000,
        EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: None
    }

