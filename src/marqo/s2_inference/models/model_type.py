from enum import Enum


class ModelType(str, Enum):
    """Enums for the different types of models that can be used for inference."""
    OpenCLIP = "open_clip"
    CLIP = 'clip'
    SBERT = 'sbert'
    Test = 'test'
    SBERT_ONNX = 'sbert_onnx'
    CLIP_ONNX = 'clip_onnx'
    MultilingualClip = "multilingual_clip"
    FP16_CLIP = "fp16_clip"
    Random = 'random'
    HF_MODEL = 'hf'
    NoModel = "no_model"