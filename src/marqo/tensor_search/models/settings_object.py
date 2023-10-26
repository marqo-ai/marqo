"""
The settings object should be validated by JSON schema, rather than PyDantic, so that it can be used as a template for
documentation and potentially front-end validation (for usability). 
"""
from marqo.tensor_search import enums as ns_enums
from marqo.tensor_search.enums import IndexSettingsField as NsFields, EnvVars, ObjectStores
from marqo.tensor_search.utils import read_env_vars_and_defaults, read_env_vars_and_defaults_ints

settings_schema = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "required": [
        NsFields.index_defaults,
        NsFields.number_of_shards,
        NsFields.number_of_replicas
    ],
    "additionalProperties": False,
    "properties": {
        NsFields.index_defaults: {
            "type": "object",
            "required": [
                NsFields.treat_urls_and_pointers_as_images,
                NsFields.model,
                NsFields.normalize_embeddings,
                NsFields.text_preprocessing,
                NsFields.image_preprocessing
            ],
            "additionalProperties": False,
            "properties": {
                NsFields.treat_urls_and_pointers_as_images: {
                    "type": "boolean",
                    "examples": [
                        False
                    ]
                },
                NsFields.model: {
                    "type": "string",
                    "examples": [
                        "hf/all_datasets_v4_MiniLM-L6"
                    ]
                },
                NsFields.model_properties: {
                    "type": "object",
                },
                NsFields.normalize_embeddings: {
                    "type": "boolean",
                    "examples": [
                        True
                    ]
                },
                NsFields.text_preprocessing: {
                    "type": "object",
                    "required": [
                        NsFields.split_length,
                        NsFields.split_overlap,
                        NsFields.split_method
                    ],
                    "properties": {
                        NsFields.split_length: {
                            "type": "integer",
                            "examples": [
                                2
                            ]
                        },
                        NsFields.split_overlap: {
                            "type": "integer",
                            "examples": [
                                0
                            ]
                        },
                        NsFields.split_method: {
                            "type": "string",
                            "examples": [
                                "sentence"
                            ]
                        }
                    },
                    "examples": [{
                        NsFields.split_length: 2,
                        NsFields.split_overlap: 0,
                        NsFields.split_method: "sentence"
                    }]
                },
                NsFields.image_preprocessing: {
                    "type": "object",
                    "required": [
                        NsFields.patch_method
                    ],
                    "properties": {
                        NsFields.patch_method: {
                            "type": ["null", "string"],
                            "examples": [
                                None
                            ]
                        }
                    },
                    "examples": [{
                        NsFields.patch_method: None
                    }]
                },
                NsFields.ann_parameters: {
                    "type": "object",
                    "required": [
                        # Non required for backwards compatibility
                    ],
                    "properties": {
                        NsFields.ann_method: {
                            "type": "string",
                            "enum": ["hnsw"],
                            "examples": [
                                "hnsw"
                            ]
                        },
                        NsFields.ann_engine: {
                            "type": "string",
                            "enum": ["lucene"],
                            "examples": [
                                "lucene"
                            ]
                        },
                        NsFields.ann_metric: {
                            "type": "string",
                            "enum": ["l1", "l2", "linf", "cosinesimil"],
                            "examples": [
                                "cosinesimil"
                            ]
                        },
                        NsFields.ann_method_parameters: {
                            "type": "object",
                            "required": [],
                            "properties": {
                                NsFields.hnsw_ef_construction: {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": read_env_vars_and_defaults_ints(EnvVars.MARQO_EF_CONSTRUCTION_MAX_VALUE),
                                    "examples": [
                                        128
                                    ]
                                },
                                NsFields.hnsw_m: {
                                    "type": "integer",
                                    "minimum": 2,
                                    "maximum": 100,
                                    "examples": [
                                        16
                                    ]
                                },
                            },
                            "examples": [{
                                NsFields.hnsw_ef_construction: 128,
                                NsFields.hnsw_m: 16
                            }]
                        }
                    },
                    "examples": [{
                        NsFields.ann_method: "hnsw",
                        NsFields.ann_engine: "lucene",
                        NsFields.ann_metric: "cosinesimil",
                        NsFields.ann_method_parameters: {
                            NsFields.hnsw_ef_construction: 128,
                            NsFields.hnsw_m: 16
                        }
                    }]
                }
            },
            "examples": [{
                NsFields.treat_urls_and_pointers_as_images: False,
                NsFields.model: "hf/all_datasets_v4_MiniLM-L6",
                NsFields.normalize_embeddings: True,
                NsFields.text_preprocessing: {
                    NsFields.split_length: 2,
                    NsFields.split_overlap: 0,
                    NsFields.split_method: "sentence"
                },
                NsFields.image_preprocessing: {
                    NsFields.patch_method: None
                },
                NsFields.ann_parameters: {
                    NsFields.ann_method: "hnsw",
                    NsFields.ann_engine: "lucene",
                    NsFields.ann_metric: "cosinesimil",
                    NsFields.ann_method_parameters: {
                        NsFields.hnsw_ef_construction: 128,
                        NsFields.hnsw_m: 16
                    }
                }
            }]
        },
        NsFields.number_of_shards: {
            "type": "integer",
            "minimum": 1,
            "examples": [
                5
            ]
        },
        NsFields.number_of_replicas: {
            "type": "integer",
            "minimum": 0,
            "maximum": read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_NUMBER_OF_REPLICAS),
            "examples": [
                1
            ]
        },
    },
    "examples": [{
        NsFields.index_defaults: {
            NsFields.treat_urls_and_pointers_as_images: False,
            NsFields.model: "hf/all_datasets_v4_MiniLM-L6",
            NsFields.normalize_embeddings: True,
            NsFields.text_preprocessing: {
                NsFields.split_length: 2,
                NsFields.split_overlap: 0,
                NsFields.split_method: "sentence"
            },
            NsFields.image_preprocessing: {
                NsFields.patch_method: None
            },
            NsFields.ann_parameters: {
                NsFields.ann_method: "hnsw",
                NsFields.ann_engine: "lucene",
                NsFields.ann_metric: "cosinesimil",
                NsFields.ann_method_parameters: {
                    NsFields.hnsw_ef_construction: 128,
                    NsFields.hnsw_m: 16
                }
            }
        },
        NsFields.number_of_shards: 3,
        NsFields.number_of_replicas: 0
    }]
}
