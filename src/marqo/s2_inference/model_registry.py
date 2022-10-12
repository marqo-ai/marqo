from marqo.s2_inference.hf_utils import HF_MODEL
from marqo.s2_inference.sbert_onnx_utils import SBERT_ONNX
from marqo.s2_inference.sbert_utils import SBERT, TEST
from marqo.s2_inference.random_utils import Random
from marqo.s2_inference.clip_utils import CLIP, OPEN_CLIP
from marqo.s2_inference.x_clip_utils import XCLIP
from marqo.s2_inference.types import Any, Dict, List, Optional, Union, FloatTensor

# we need to keep track of the embed dim and model load functions/classes
# we can use this as a registry 
def _get_clip_properties() -> Dict:
    CLIP_MODEL_PROPERTIES = {
            'RN50':
                {"name": "RN50",
                "dimensions": 1024,
                "notes": "CLIP resnet50",
                "type": "clip",
                },
            'RN101':
                {"name": "RN101",
                "dimensions": 512,
                "notes": "CLIP resnet101",
                "type": "clip",
                },
            'RN50x4':
                {"name": "RN50x4",
                "dimensions": 640,
                "notes": "CLIP resnet50x4",
                "type": "clip",
                },              
            'RN50x16':
                {"name": "RN50x16",
                "dimensions": 768,
                "notes": "CLIP resnet50x16",
                "type": "clip",
                },
            'RN50x64':
                {"name": "RN50x64",
                "dimensions": 1024,
                "notes": "CLIP resnet50x64",
                "type": "clip",
                },
            'ViT-B/32':
                {"name": "ViT-B/32",
                "dimensions": 512,
                "notes": "CLIP ViT-B/32",
                "type": "clip",
                },
            'ViT-B/16':
                {"name": "ViT-B/16",
                "dimensions": 512,
                "notes": "CLIP ViT-B/16",
                "type":"clip",
                },
            'ViT-L/14':
                {"name": "ViT-L/14",
                "dimensions": 768,
                "notes": "CLIP ViT-L/14",
                "type":"clip",
                },
            'ViT-L/14@336px':
                {"name": "ViT-L/14@336px",
                "dimensions": 768,
                "notes": "CLIP ViT-L/14@336px",
                "type":"clip",
                },

        }
    return CLIP_MODEL_PROPERTIES
def _get_open_clip_properties() -> Dict:

    #use this link to find all the model_configs
    #https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs

    OPEN_CLIP_MODEL_PROPERTIES = {
        'open_clip/RN50/openai':
           {'name': 'open_clip/RN50/openai',
            'dimensions': 1024,
            'note': 'clip model from open_clip implementation',
            'type': 'open_clip',
            'pretrained': 'openai'},

        'open_clip/RN50/yfcc15m':
            {'name': 'open_clip/RN50/yfcc15m',
             'dimensions': 1024, 'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'yfcc15m'},

        'open_clip/RN50/cc12m':
            {'name': 'open_clip/RN50/cc12m',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'cc12m'},

        'open_clip/RN50-quickgelu/openai':
            {'name': 'open_clip/RN50-quickgelu/openai',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'openai'},

        'open_clip/RN50-quickgelu/yfcc15m':
            {'name': 'open_clip/RN50-quickgelu/yfcc15m',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'yfcc15m'},

        'open_clip/RN50-quickgelu/cc12m':
            {'name': 'open_clip/RN50-quickgelu/cc12m',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'cc12m'},

        'open_clip/RN101/openai':
            {'name': 'open_clip/RN101/openai',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'openai'},

        'open_clip/RN101/yfcc15m':
            {'name': 'open_clip/RN101/yfcc15m',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'yfcc15m'},

        'open_clip/RN101-quickgelu/openai':
            {'name': 'open_clip/RN101-quickgelu/openai',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/RN101-quickgelu/yfcc15m':
            {'name': 'open_clip/RN101-quickgelu/yfcc15m',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'yfcc15m'},

        'open_clip/RN50x4/openai':
            {'name': 'open_clip/RN50x4/openai',
             'dimensions': 640,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/RN50x16/openai':
            {'name': 'open_clip/RN50x16/openai',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/RN50x64/openai':
            {'name': 'open_clip/RN50x64/openai',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'openai'},

        'open_clip/ViT-B-32/openai':
            {'name': 'open_clip/ViT-B-32/openai',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/ViT-B-32/laion400m_e31':
            {'name': 'open_clip/ViT-B-32/laion400m_e31',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e31'},

        'open_clip/ViT-B-32/laion400m_e32':
            {'name': 'open_clip/ViT-B-32/laion400m_e32',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e32'},

        'open_clip/ViT-B-32/laion2b_e16':
            {'name': 'open_clip/ViT-B-32/laion2b_e16',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion2b_e16'},

        'open_clip/ViT-B-32/laion2b_s34b_b79k':
            {'name': 'open_clip/ViT-B-32/laion2b_s34b_b79k',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion2b_s34b_b79k'},

        'open_clip/ViT-B-32-quickgelu/openai':
            {'name': 'open_clip/ViT-B-32-quickgelu/openai',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/ViT-B-32-quickgelu/laion400m_e31':
            {'name': 'open_clip/ViT-B-32-quickgelu/laion400m_e31',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion400m_e31'},

        'open_clip/ViT-B-32-quickgelu/laion400m_e32':
            {'name': 'open_clip/ViT-B-32-quickgelu/laion400m_e32',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e32'},

        'open_clip/ViT-B-16/openai':
            {'name': 'open_clip/ViT-B-16/openai',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'openai'},

        'open_clip/ViT-B-16/laion400m_e31':
            {'name': 'open_clip/ViT-B-16/laion400m_e31',
             'dimensions': 512,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e31'},

        'open_clip/ViT-B-16/laion400m_e32':
            {'name': 'open_clip/ViT-B-16/laion400m_e32',
             'dimensions': 512, 'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e32'},

        'open_clip/ViT-B-16-plus-240/laion400m_e31':
            {'name': 'open_clip/ViT-B-16-plus-240/laion400m_e31',
             'dimensions': 640,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e31'},

        'open_clip/ViT-B-16-plus-240/laion400m_e32':
            {'name': 'open_clip/ViT-B-16-plus-240/laion400m_e32',
             'dimensions': 640,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e32'},

        'open_clip/ViT-L-14/openai':
            {'name': 'open_clip/ViT-L-14/openai',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/ViT-L-14/laion400m_e31':
            {'name': 'open_clip/ViT-L-14/laion400m_e31',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip', 'pretrained': 'laion400m_e31'},

        'open_clip/ViT-L-14/laion400m_e32':
            {'name': 'open_clip/ViT-L-14/laion400m_e32',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion400m_e32'},

        'open_clip/ViT-L-14/laion2b_s32b_b82k':
            {'name': 'open_clip/ViT-L-14/laion2b_s32b_b82k',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion2b_s32b_b82k'},

        'open_clip/ViT-L-14-336/openai':
            {'name': 'open_clip/ViT-L-14-336/openai',
             'dimensions': 768,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'openai'},

        'open_clip/ViT-H-14/laion2b_s32b_b79k':
            {'name': 'open_clip/ViT-H-14/laion2b_s32b_b79k',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion2b_s32b_b79k'},

        'open_clip/ViT-g-14/laion2b_s12b_b42k':
            {'name': 'open_clip/ViT-g-14/laion2b_s12b_b42k',
             'dimensions': 1024,
             'note': 'clip model from open_clip implementation',
             'type': 'open_clip',
             'pretrained': 'laion2b_s12b_b42k'}}

    return OPEN_CLIP_MODEL_PROPERTIES


def _get_sbert_properties() -> Dict:
    SBERT_MODEL_PROPERTIES = {
            "sentence-transformers/all-MiniLM-L6-v1": 
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-MiniLM-L6-v2": 
                {"name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "tokens":256,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-mpnet-base-v1": 
                {"name": "sentence-transformers/all-mpnet-base-v1",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-mpnet-base-v2": 
                {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            'sentence-transformers/stsb-xlm-r-multilingual':
                {"name": 'sentence-transformers/stsb-xlm-r-multilingual',
                 "dimensions": 768,
                 "tokens": 128,
                 "type": "sbert",
                 "notes": ""},
            "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},

            "flax-sentence-embeddings/all_datasets_v3_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
    }
    return SBERT_MODEL_PROPERTIES

def _get_hf_properties() -> Dict:
    HF_MODEL_PROPERTIES = {
            "hf/all-MiniLM-L6-v1": 
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all-MiniLM-L6-v2": 
                {"name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "tokens":256,
                "type":"hf",
                "notes": ""},
            "hf/all-mpnet-base-v1": 
                {"name": "sentence-transformers/all-mpnet-base-v1",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all-mpnet-base-v2": 
                {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},

            "hf/all_datasets_v3_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v3_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},

            "hf/all_datasets_v3_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""}
    }
    return HF_MODEL_PROPERTIES

def _get_sbert_onnx_properties() -> Dict:
    SBERT_ONNX_MODEL_PROPERTIES = {
            "onnx/all-MiniLM-L6-v1": 
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all-MiniLM-L6-v2": 
                {"name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "tokens":256,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all-mpnet-base-v1": 
                {"name": "sentence-transformers/all-mpnet-base-v1",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all-mpnet-base-v2": 
                {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},

            "onnx/all_datasets_v3_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all_datasets_v3_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all_datasets_v4_MiniLM-L12": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all_datasets_v4_MiniLM-L6": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},

            "onnx/all_datasets_v3_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
            "onnx/all_datasets_v4_mpnet-base": 
                {"name": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert_onnx",
                "notes": ""},
    }
    return SBERT_ONNX_MODEL_PROPERTIES

def _get_sbert_test_properties() -> Dict:
    TEST_MODEL_PROPERTIES = {
            "sentence-transformers/test": 
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 16,
                "tokens":128,
                "type":"test",
                "notes": ""},
            "test": 
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 16,
                "tokens":128,
                "type":"test",
                "notes": ""},
    }
    return TEST_MODEL_PROPERTIES

def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
            "random": 
                {"name": "random",
                "dimensions": 384,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/large": 
                {"name": "random/large",
                "dimensions": 768,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/small": 
                {"name": "random/small",
                "dimensions": 32,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/medium": 
                {"name": "random/medium",
                "dimensions": 128,
                "tokens":128,
                "type":"random",
                "notes": ""},

    }
    return RANDOM_MODEL_PROPERTIES

def _get_XCLIP_properties() -> Dict:
    XCLIP_MODEL_PROPERTIES = {
            "microsoft/xclip-base-patch16-ucf-4-shot":
                {"name": "microsoft/xclip-base-patch16-ucf-4-shot",
                "dimensions": 384,
                "tokens":128,
                "type":"random",
                "notes": ""},
    }
    return XCLIP_MODEL_PROPERTIES

def _get_model_load_mappings() -> Dict:
    return {'clip':CLIP,
            'open_clip': OPEN_CLIP,
            'sbert':SBERT, 
            'test':TEST, 
            'sbert_onnx':SBERT_ONNX,
            'random':Random,
            'xclip': XCLIP,
            'hf':HF_MODEL}

def load_model_properties() -> Dict:
    # also truncate the name if not already
    sbert_model_properties = _get_sbert_properties()
    sbert_model_properties.update({k.split('/')[-1]:v for k,v in sbert_model_properties.items()})

    sbert_onnx_model_properties = _get_sbert_onnx_properties()

    clip_model_properties = _get_clip_properties()
    test_model_properties = _get_sbert_test_properties()
    random_model_properties = _get_random_properties()
    hf_model_properties = _get_hf_properties()
    open_clip_model_properties = _get_open_clip_properties()
    XCLIP_properties = _get_XCLIP_properties()

    # combine the above dicts
    model_properties = dict(clip_model_properties.items())
    model_properties.update(sbert_model_properties)
    model_properties.update(test_model_properties)
    model_properties.update(sbert_onnx_model_properties)
    model_properties.update(random_model_properties)
    model_properties.update(hf_model_properties)
    model_properties.update(open_clip_model_properties)
    model_properties.update(XCLIP_properties)

    all_properties = dict()
    all_properties['models'] = model_properties

    all_properties['loaders'] = dict()
    for key,val in _get_model_load_mappings().items():
        all_properties['loaders'][key] = val

    return all_properties

