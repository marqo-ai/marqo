from marqo.s2_inference.hf_utils import HF_MODEL
from marqo.s2_inference.sbert_onnx_utils import SBERT_ONNX
from marqo.s2_inference.sbert_utils import SBERT, TEST
from marqo.s2_inference.random_utils import Random
from marqo.s2_inference.clip_utils import CLIP, OPEN_CLIP, MULTILINGUAL_CLIP, FP16_CLIP, get_multilingual_clip_properties
from marqo.s2_inference.types import Any, Dict, List, Optional, Union, FloatTensor
from marqo.s2_inference.onnx_clip_utils import CLIP_ONNX

# we need to keep track of the embed dim and model load functions/classes
# we can use this as a registry

def _get_model_load_mappings() -> Dict:
    return {'clip':CLIP,
            'open_clip': OPEN_CLIP,
            'sbert':SBERT,
            'test':TEST,
            'sbert_onnx':SBERT_ONNX,
            'clip_onnx': CLIP_ONNX,
            "multilingual_clip" : MULTILINGUAL_CLIP,
            "fp16_clip": FP16_CLIP,
            'random':Random,
            'hf':HF_MODEL}

def load_model_properties() -> Dict:
    all_properties = dict()
    for key,val in _get_model_load_mappings().items():
        all_properties[key] = val

    return all_properties
