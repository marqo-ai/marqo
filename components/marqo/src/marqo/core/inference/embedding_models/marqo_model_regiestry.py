from ..api.exceptions import UnsupportedModelError


_MODEL_REGISTRY: dict[str, dict] = {
    "Marqo/marqo-fashionCLIP": {
        "name": "hf-hub:Marqo/marqo-fashionCLIP",
        "dimensions": 512,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-fashionCLIP-image-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-fashionCLIP/image-encoder/model.onnx"],
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
                    "dims": [512],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-fashionCLIP-text-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-fashionCLIP/text-encoder/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [77],
                    "dataType": "TYPE_INT32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [512],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "Marqo/marqo-fashionSigLIP": {
        "name": "hf-hub:Marqo/marqo-fashionSigLIP",
        "dimensions": 768,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-fashionSigLIP-image-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-fashionSigLIP/image-encoder/model.onnx",
                        "s3://marqo-opensource-models/marqo-fashionSigLIP/image-encoder/model.onnx.data"],
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
            "sources": ["s3://marqo-opensource-models/marqo-fashionSigLIP/text-encoder/model.onnx"],
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

    "Marqo/marqo-ecommerce-embeddings-L": {
        "name": "hf-hub:Marqo/marqo-ecommerce-embeddings-L",
        "dimensions": 1024,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-ecommerce-embeddings-L-image-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-ecommerce-embeddings-L/image-encoder/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [3, 256, 256],
                    "dataType": "TYPE_FP32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 16,
            "name": "marqo-ecommerce-embeddings-L-text-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-ecommerce-embeddings-L/text-encoder/model.onnx"],
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
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "Marqo/marqo-ecommerce-embeddings-B": {
        "name": "hf-hub:Marqo/marqo-ecommerce-embeddings-B",
        "dimensions": 768,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "marqo-ecommerce-embeddings-B-image-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-ecommerce-embeddings-B/image-encoder/model.onnx"],
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
            "maxBatchSize": 16,
            "name": "marqo-ecommerce-embeddings-B-text-encoder",
            "sources": ["s3://marqo-opensource-models/marqo-ecommerce-embeddings-B/text-encoder/model.onnx"],
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

    "timm/ViT-L-16-SigLIP2-256": {
        "name": "hf-hub:timm/ViT-L-16-SigLIP2-256",
        "dimensions": 1024,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "timm-ViT-L-16-SigLIP2-256-image-encoder",
            "sources": ["s3://marqo-opensource-models/timm-ViT-L-16-SigLIP2-256/image-encoder/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [3, 256, 256],
                    "dataType": "TYPE_FP32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 16,
            "name": "timm-ViT-L-16-SigLIP2-256-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/timm-ViT-L-16-SigLIP2-256/text-encoder/model.onnx",
                "s3://marqo-opensource-models/timm-ViT-L-16-SigLIP2-256/text-encoder/model.onnx.data"
            ],
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
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "open_clip/ViT-L-16-SigLIP-256/webli": {
        "name": "hf-hub:timm/ViT-L-16-SigLIP-256",
        "dimensions": 1024,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "timm-ViT-L-16-SigLIP-256-image-encoder",
            "sources": ["s3://marqo-opensource-models/timm-ViT-L-16-SigLIP-256/image-encoder/model.onnx"],
            "input": [
                {
                    "name": "input",
                    "dims": [3, 256, 256],
                    "dataType": "TYPE_FP32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 16,
            "name": "timm-ViT-L-16-SigLIP-256-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/timm-ViT-L-16-SigLIP-256/text-encoder/model.onnx",
            ],
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
                    "dims": [1024],
                    "dataType": "TYPE_FP32"
                }
            ]
        }
    },

    "open_clip/ViT-L-14/laion2b_s32b_b82k": {
        "name": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "dimensions": 768,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "laion-CLIP-ViT-L-14-laion2B-s32B-b82K-image-encoder",
            "sources": ["s3://marqo-opensource-models/laion-CLIP-ViT-L-14-laion2B-s32B-b82K/image-encoder/model.onnx"],
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
            "maxBatchSize": 16,
            "name": "laion-CLIP-ViT-L-14-laion2B-s32B-b82K-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/laion-CLIP-ViT-L-14-laion2B-s32B-b82K/text-encoder/model.onnx",
            ],
            "input": [
                {
                    "name": "input",
                    "dims": [77],
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

    "open_clip/ViT-B-32/laion2b_s34b_b79k": {
            "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dimensions": 512,
            "type": "open_clip",
            "tritonImageEncoder": {
                "maxBatchSize": 8,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-image-encoder",
                "sources": ["s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/image-encoder/model.onnx"],
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
                        "dims": [512],
                        "dataType": "TYPE_FP32"
                    }
                ]
            },
            "tritonTextEncoder": {
                "maxBatchSize": 16,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-text-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/text-encoder/model.onnx",
                ],
                "input": [
                    {
                        "name": "input",
                        "dims": [77],
                        "dataType": "TYPE_INT32"
                    }
                ],
                "output": [
                    {
                        "name": "output",
                        "dims": [512],
                        "dataType": "TYPE_FP32"
                    }
                ]
            }
        },

    "open_clip/ViT-B-16-SigLIP/webli": {
            "name": "hf-hub:timm/ViT-B-16-SigLIP",
            "dimensions": 768,
            "type": "open_clip",
            "tritonImageEncoder": {
                "maxBatchSize": 8,
                "name": "timm-ViT-B-16-SigLIP-image-encoder",
                "sources": ["s3://marqo-opensource-models/timm-ViT-B-16-SigLIP/image-encoder/model.onnx"],
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
                "maxBatchSize": 16,
                "name": "timm-ViT-B-16-SigLIP-text-encoder",
                "sources": [
                    "s3://marqo-opensource-models/timm-ViT-B-16-SigLIP/text-encoder/model.onnx",
                ],
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

    "open_clip/ViT-L-14/laion400m_e32b_b82k": {
        "name": "hf-hub:timm/vit_large_patch14_clip_224.laion400m_e32",
        "dimensions": 768,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "timm-ViT-L-14-laion400m_e32b_b82k-image-encoder",
            "sources": ["s3://marqo-opensource-models/timm-vit_large_patch14_clip_224.laion400m_e32/image-encoder/model.onnx"],
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
            "maxBatchSize": 16,
            "name": "timm-ViT-L-14-laion400m_e32b_b82k-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/timm-vit_large_patch14_clip_224.laion400m_e32/text-encoder/model.onnx",
            ],
            "input": [
                {
                    "name": "input",
                    "dims": [77],
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

    "laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": {
        "name": "open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k",
        "dimensions": 512,
        "type": "open_clip",
        "tritonImageEncoder": {
            "maxBatchSize": 8,
            "name": "laion-CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k-image-encoder",
            "sources": ["s3://marqo-opensource-models/laion-CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/image-encoder/model.onnx"],
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
                    "dims": [512],
                    "dataType": "TYPE_FP32"
                }
            ]
        },
        "tritonTextEncoder": {
            "maxBatchSize": 16,
            "name": "laion-CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k-text-encoder",
            "sources": [
                "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/text-encoder/model.onnx",
            ],
            "input": [
                {
                    "name": "input",
                    "dims": [77],
                    "dataType": "TYPE_INT32"
                }
            ],
            "output": [
                {
                    "name": "output",
                    "dims": [512],
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


def validate_model_properties(model_properties: dict):
    pass