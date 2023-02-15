
settings_schema = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "required": [
        "index_defaults",
        "number_of_shards"
    ],
    "properties": {
        "index_defaults": {
            "type": "object",
            "required": [
                "treat_urls_and_pointers_as_images",
                "model",
                "normalize_embeddings",
                "text_preprocessing",
                "image_preprocessing"
            ],
            "properties": {
                "treat_urls_and_pointers_as_images": {
                    "type": "boolean",
                    "examples": [
                        False
                    ]
                },
                "model": {
                    "type": "string",
                    "examples": [
                        "hf/all_datasets_v4_MiniLM-L6"
                    ]
                },
                "normalize_embeddings": {
                    "type": "boolean",
                    "examples": [
                        True
                    ]
                },
                "text_preprocessing": {
                    "type": "object",
                    "required": [
                        "split_length",
                        "split_overlap",
                        "split_method"
                    ],
                    "properties": {
                        "split_length": {
                            "type": "integer",
                            "examples": [
                                2
                            ]
                        },
                        "split_overlap": {
                            "type": "integer",
                            "examples": [
                                0
                            ]
                        },
                        "split_method": {
                            "type": "string",
                            "examples": [
                                "sentence"
                            ]
                        }
                    },
                    "examples": [{
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence"
                    }]
                },
                "image_preprocessing": {
                    "type": "object",
                    "required": [
                        "patch_method"
                    ],
                    "properties": {
                        "patch_method": {
                            "type": "null",
                            "examples": [
                                None
                            ]
                        }
                    },
                    "examples": [{
                        "patch_method": None
                    }]
                }
            },
            "examples": [{
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            }]
        },
        "number_of_shards": {
            "type": "integer",
            "minimum": 1,
            "examples": [
                5
            ]
        }
    },
    "examples": [{
        "index_defaults": {
            "treat_urls_and_pointers_as_images": False,
            "model": "hf/all_datasets_v4_MiniLM-L6",
            "normalize_embeddings": True,
            "text_preprocessing": {
                "split_length": 2,
                "split_overlap": 0,
                "split_method": "sentence"
            },
            "image_preprocessing": {
                "patch_method": None
            }
        },
        "number_of_shards": 5
    }]
}