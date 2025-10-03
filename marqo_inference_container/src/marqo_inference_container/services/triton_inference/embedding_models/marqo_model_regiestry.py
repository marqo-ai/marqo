from marqo_inference_container.services.errors import UnsupportedModelError

_MODEL_REGISTRY: dict[str, dict] = {
    "marqo/marqo-fashionSigLIP": {
        "name": "hf-hub:Marqo/marqo-fashionSigLIP",
        "dimensions": 768,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-fashionSigLIP-image-encoder",
            "sources": ["s3://opensource-li-backup/triton_models/marqo-fashionSigLIP-image-encoder/1/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [3, 224, 224],
                    "dataType": "TYPE_FP32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [768],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-fashionSigLIP-text-encoder",
            "sources": ["s3://opensource-li-backup/triton_models/marqo-fashionSigLIP-text-encoder/1/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [64],
                    "dataType": "TYPE_INT32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [768],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "hf/e5-base-v2": {
        "name": "intfloat/e5-base-v2",
        "dimensions": 768,
        "type": "hf",
        "poolingMethod": "mean",
        "tritonTextEncoder": {
            "maxBatchSize": 32,
            "name": "e5-base-v2-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/infloat-e5-base-v2/model.onnx"],
            "input": [
                {
                    "name": "input_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "attention_mask",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "token_type_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                }
            ],
            "output": [
                {
                    "name": "last_hidden_state",
                    "dims": [-1, 768],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "hf/e5-small-v2": {
        "name": "intfloat/e5-small-v2",
        "dimensions": 384,
        "type": "hf",
        "poolingMethod": "mean",
        "tritonTextEncoder": {
            "maxBatchSize": 32,
            "name": "e5-small-v2-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/infloat-e5-small-v2/model.onnx"],
            "input": [
                {
                    "name": "input_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "attention_mask",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "token_type_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                }
            ],
            "output": [
                {
                    "name": "last_hidden_state",
                    "dims": [-1, 384],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "hf/all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "type": "hf",
        "poolingMethod": "mean",
        "tritonTextEncoder": {
            "maxBatchSize": 16,
            "name": "all-MiniLM-L6-v2-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/sentence-transformers-all-minilm-l6-v2/model.onnx"],
            "input": [
                {
                    "name": "input_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "attention_mask",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                },
                {
                    "name": "token_type_ids",
                    "dims": [-1],
                    "dataType": "TYPE_INT64"
                }
            ],
            "output": [
                {
                    "name": "last_hidden_state",
                    "dims": [-1, 384],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },
}


def get_model_properties(model_name: str) -> dict:
    if model_name not in _MODEL_REGISTRY:
        raise UnsupportedModelError(f"Model {model_name} is not registered.")
    return _MODEL_REGISTRY[model_name]
