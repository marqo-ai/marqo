from marqo.tensor_search import enums as ns_enums
from marqo.tensor_search.enums import IndexSettingsField as NsFields, EnvVars

settings_schema = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "required": [
        NsFields.index_defaults,
        NsFields.number_of_shards
    ],
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
                }
            }]
        },
        NsFields.number_of_shards: {
            "type": "integer",
            "minimum": 1,
            "examples": [
                5
            ]
        }
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
            }
        },
        NsFields.number_of_shards: 5
    }]
}