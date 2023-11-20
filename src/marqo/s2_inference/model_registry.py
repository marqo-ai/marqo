from marqo.s2_inference.hf_utils import HF_MODEL
from marqo.s2_inference.sbert_onnx_utils import SBERT_ONNX
from marqo.s2_inference.sbert_utils import SBERT, TEST
from marqo.s2_inference.random_utils import Random
from marqo.s2_inference.clip_utils import CLIP, OPEN_CLIP, MULTILINGUAL_CLIP, FP16_CLIP, get_multilingual_clip_properties
from marqo.s2_inference.types import Any, Dict, List, Optional, Union, FloatTensor
from marqo.s2_inference.onnx_clip_utils import CLIP_ONNX
from marqo.s2_inference.no_model_utils import NO_MODEL

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
    # use this link to find all the model_configs
    # https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs
    OPEN_CLIP_MODEL_PROPERTIES = {
        'open_clip/RN50/openai': {'name': 'open_clip/RN50/openai',
                                  'dimensions': 1024,
                                  'note': 'open_clip models',
                                  'type': 'open_clip',
                                  'pretrained': 'openai'},
        'open_clip/RN50/yfcc15m': {'name': 'open_clip/RN50/yfcc15m',
                                   'dimensions': 1024,
                                   'note': 'open_clip models',
                                   'type': 'open_clip',
                                   'pretrained': 'yfcc15m'},
        'open_clip/RN50/cc12m': {'name': 'open_clip/RN50/cc12m', 'dimensions': 1024, 'note': 'open_clip models',
                                 'type': 'open_clip', 'pretrained': 'cc12m'},
        'open_clip/RN50-quickgelu/openai': {'name': 'open_clip/RN50-quickgelu/openai',
                                            'dimensions': 1024,
                                            'note': 'open_clip models',
                                            'type': 'open_clip',
                                            'pretrained': 'openai'},
        'open_clip/RN50-quickgelu/yfcc15m': {'name': 'open_clip/RN50-quickgelu/yfcc15m',
                                             'dimensions': 1024,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'yfcc15m'},
        'open_clip/RN50-quickgelu/cc12m': {'name': 'open_clip/RN50-quickgelu/cc12m',
                                           'dimensions': 1024,
                                           'note': 'open_clip models',
                                           'type': 'open_clip',
                                           'pretrained': 'cc12m'},
        'open_clip/RN101/openai': {'name': 'open_clip/RN101/openai',
                                   'dimensions': 512,
                                   'note': 'open_clip models',
                                   'type': 'open_clip',
                                   'pretrained': 'openai'},
        'open_clip/RN101/yfcc15m': {'name': 'open_clip/RN101/yfcc15m',
                                    'dimensions': 512,
                                    'note': 'open_clip models',
                                    'type': 'open_clip',
                                    'pretrained': 'yfcc15m'},
        'open_clip/RN101-quickgelu/openai': {'name': 'open_clip/RN101-quickgelu/openai',
                                             'dimensions': 512,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'openai'},
        'open_clip/RN101-quickgelu/yfcc15m': {'name': 'open_clip/RN101-quickgelu/yfcc15m',
                                              'dimensions': 512,
                                              'note': 'open_clip models',
                                              'type': 'open_clip',
                                              'pretrained': 'yfcc15m'},
        'open_clip/RN50x4/openai': {'name': 'open_clip/RN50x4/openai',
                                    'dimensions': 640,
                                    'note': 'open_clip models',
                                    'type': 'open_clip',
                                    'pretrained': 'openai'},
        'open_clip/RN50x16/openai': {'name': 'open_clip/RN50x16/openai',
                                     'dimensions': 768,
                                     'note': 'open_clip models',
                                     'type': 'open_clip',
                                     'pretrained': 'openai'},
        'open_clip/RN50x64/openai': {'name': 'open_clip/RN50x64/openai',
                                     'dimensions': 1024,
                                     'note': 'open_clip models',
                                     'type': 'open_clip',
                                     'pretrained': 'openai'},
        'open_clip/ViT-B-32/openai': {'name': 'open_clip/ViT-B-32/openai',
                                      'dimensions': 512,
                                      'note': 'open_clip models',
                                      'type': 'open_clip',
                                      'pretrained': 'openai'},
        'open_clip/ViT-B-32/laion400m_e31': {'name': 'open_clip/ViT-B-32/laion400m_e31',
                                             'dimensions': 512,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e31'},
        'open_clip/ViT-B-32/laion400m_e32': {'name': 'open_clip/ViT-B-32/laion400m_e32',
                                             'dimensions': 512,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e32'},
        'open_clip/ViT-B-32/laion2b_e16': {'name': 'open_clip/ViT-B-32/laion2b_e16',
                                           'dimensions': 512,
                                           'note': 'open_clip models',
                                           'type': 'open_clip',
                                           'pretrained': 'laion2b_e16'},
        'open_clip/ViT-B-32/laion2b_s34b_b79k': {'name': 'open_clip/ViT-B-32/laion2b_s34b_b79k',
                                                 'dimensions': 512,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s34b_b79k'},
        'open_clip/ViT-B-32-quickgelu/openai': {'name': 'open_clip/ViT-B-32-quickgelu/openai',
                                                'dimensions': 512,
                                                'note': 'open_clip models',
                                                'type': 'open_clip',
                                                'pretrained': 'openai'},
        'open_clip/ViT-B-32-quickgelu/laion400m_e31': {'name': 'open_clip/ViT-B-32-quickgelu/laion400m_e31',
                                                       'dimensions': 512,
                                                       'note': 'open_clip models',
                                                       'type': 'open_clip',
                                                       'pretrained': 'laion400m_e31'},
        'open_clip/ViT-B-32-quickgelu/laion400m_e32': {'name': 'open_clip/ViT-B-32-quickgelu/laion400m_e32',
                                                       'dimensions': 512,
                                                       'note': 'open_clip models',
                                                       'type': 'open_clip',
                                                       'pretrained': 'laion400m_e32'},
        'open_clip/ViT-B-16/openai': {'name': 'open_clip/ViT-B-16/openai',
                                      'dimensions': 512,
                                      'note': 'open_clip models',
                                      'type': 'open_clip',
                                      'pretrained': 'openai'},
        'open_clip/ViT-B-16/laion400m_e31': {'name': 'open_clip/ViT-B-16/laion400m_e31',
                                             'dimensions': 512,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e31'},
        'open_clip/ViT-B-16/laion400m_e32': {'name': 'open_clip/ViT-B-16/laion400m_e32',
                                             'dimensions': 512,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e32'},
        'open_clip/ViT-B-16/laion2b_s34b_b88k': {'name': 'open_clip/ViT-B-16/laion2b_s34b_b88k',
                                                 'dimensions': 512,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s34b_b88k'},
        'open_clip/ViT-B-16-plus-240/laion400m_e31': {'name': 'open_clip/ViT-B-16-plus-240/laion400m_e31',
                                                      'dimensions': 640,
                                                      'note': 'open_clip models',
                                                      'type': 'open_clip',
                                                      'pretrained': 'laion400m_e31'},
        'open_clip/ViT-B-16-plus-240/laion400m_e32': {'name': 'open_clip/ViT-B-16-plus-240/laion400m_e32',
                                                      'dimensions': 640,
                                                      'note': 'open_clip models',
                                                      'type': 'open_clip',
                                                      'pretrained': 'laion400m_e32'},
        'open_clip/ViT-L-14/openai': {'name': 'open_clip/ViT-L-14/openai',
                                      'dimensions': 768,
                                      'note': 'open_clip models',
                                      'type': 'open_clip',
                                      'pretrained': 'openai'},
        'open_clip/ViT-L-14/laion400m_e31': {'name': 'open_clip/ViT-L-14/laion400m_e31',
                                             'dimensions': 768,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e31'},
        'open_clip/ViT-L-14/laion400m_e32': {'name': 'open_clip/ViT-L-14/laion400m_e32',
                                             'dimensions': 768,
                                             'note': 'open_clip models',
                                             'type': 'open_clip',
                                             'pretrained': 'laion400m_e32'},
        'open_clip/ViT-L-14/laion2b_s32b_b82k': {'name': 'open_clip/ViT-L-14/laion2b_s32b_b82k',
                                                 'dimensions': 768,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s32b_b82k'},
        'open_clip/ViT-L-14-336/openai': {'name': 'open_clip/ViT-L-14-336/openai',
                                          'dimensions': 768,
                                          'note': 'open_clip models',
                                          'type': 'open_clip',
                                          'pretrained': 'openai'},
        'open_clip/ViT-H-14/laion2b_s32b_b79k': {'name': 'open_clip/ViT-H-14/laion2b_s32b_b79k',
                                                 'dimensions': 1024,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s32b_b79k'},
        'open_clip/ViT-g-14/laion2b_s12b_b42k': {'name': 'open_clip/ViT-g-14/laion2b_s12b_b42k',
                                                 'dimensions': 1024,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s12b_b42k'},
        'open_clip/ViT-g-14/laion2b_s34b_b88k': {'name': 'open_clip/ViT-g-14/laion2b_s34b_b88k',
                                                 'dimensions': 1024,
                                                 'note': 'open_clip models',
                                                 'type': 'open_clip',
                                                 'pretrained': 'laion2b_s34b_b88k'},
        'open_clip/ViT-bigG-14/laion2b_s39b_b160k': {'name': 'open_clip/ViT-bigG-14/laion2b_s39b_b160k',
                                                     'dimensions': 1280,
                                                     'note': 'open_clip models',
                                                     'type': 'open_clip',
                                                     'pretrained': 'laion2b_s39b_b160k'},
        'open_clip/roberta-ViT-B-32/laion2b_s12b_b32k': {'name': 'open_clip/roberta-ViT-B-32/laion2b_s12b_b32k',
                                                         'dimensions': 512,
                                                         'note': 'open_clip models',
                                                         'type': 'open_clip',
                                                         'pretrained': 'laion2b_s12b_b32k'},
        'open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k': {
            'name': 'open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k',
            'dimensions': 512,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion5b_s13b_b90k'},
        'open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k': {
            'name': 'open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k',
            'dimensions': 1024,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'frozen_laion5b_s13b_b90k'},
        'open_clip/convnext_base/laion400m_s13b_b51k': {'name': 'open_clip/convnext_base/laion400m_s13b_b51k',
                                                        'dimensions': 512,
                                                        'note': 'open_clip models',
                                                        'type': 'open_clip',
                                                        'pretrained': 'laion400m_s13b_b51k'},
        'open_clip/convnext_base_w/laion2b_s13b_b82k': {'name': 'open_clip/convnext_base_w/laion2b_s13b_b82k',
                                                        'dimensions': 640,
                                                        'note': 'open_clip models',
                                                        'type': 'open_clip',
                                                        'pretrained': 'laion2b_s13b_b82k'},
        'open_clip/convnext_base_w/laion2b_s13b_b82k_augreg': {
            'name': 'open_clip/convnext_base_w/laion2b_s13b_b82k_augreg',
            'dimensions': 640,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion2b_s13b_b82k_augreg'},
        'open_clip/convnext_base_w/laion_aesthetic_s13b_b82k': {
            'name': 'open_clip/convnext_base_w/laion_aesthetic_s13b_b82k',
            'dimensions': 640,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion_aesthetic_s13b_b82k'},
        'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k': {
            'name': 'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k',
            'dimensions': 640,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion_aesthetic_s13b_b82k'},
        'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg': {
            'name': 'open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg',
            'dimensions': 640,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion_aesthetic_s13b_b82k_augreg'},
        'open_clip/convnext_large_d/laion2b_s26b_b102k_augreg': {
            'name': 'open_clip/convnext_large_d/laion2b_s26b_b102k_augreg',
            'dimensions': 768,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion2b_s26b_b102k_augreg'},
        'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft': {
            'name': 'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft',
            'dimensions': 768,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion2b_s29b_b131k_ft'},
        'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup': {
            'name': 'open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup',
            'dimensions': 768,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'laion2b_s29b_b131k_ft_soup'},
        # Comment out as they are not currently available in open_clip release 2.18.1
        # It is discussed here https: // github.com / mlfoundations / open_clip / issues / 477
        # 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg': {
        #     'name': 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg',
        #     'dimensions': 1024,
        #     'note': 'open_clip models',
        #     'type': 'open_clip',
        #     'pretrained': 'laion2b_s34b_b82k_augreg'},
        # 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_rewind': {
        #     'name': 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_rewind',
        #     'dimensions': 1024,
        #     'note': 'open_clip models',
        #     'type': 'open_clip',
        #     'pretrained': 'laion2b_s34b_b82k_augreg_rewind'},
        # 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup': {
        #     'name': 'open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup',
        #     'dimensions': 1024,
        #     'note': 'open_clip models',
        #     'type': 'open_clip',
        #     'pretrained': 'laion2b_s34b_b82k_augreg_soup'},
        'open_clip/coca_ViT-B-32/laion2b_s13b_b90k': {'name': 'open_clip/coca_ViT-B-32/laion2b_s13b_b90k',
                                                      'dimensions': 512,
                                                      'note': 'open_clip models',
                                                      'type': 'open_clip',
                                                      'pretrained': 'laion2b_s13b_b90k'},
        'open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k': {
            'name': 'open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k',
            'dimensions': 512,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'mscoco_finetuned_laion2b_s13b_b90k'},
        'open_clip/coca_ViT-L-14/laion2b_s13b_b90k': {'name': 'open_clip/coca_ViT-L-14/laion2b_s13b_b90k',
                                                      'dimensions': 768,
                                                      'note': 'open_clip models',
                                                      'type': 'open_clip',
                                                      'pretrained': 'laion2b_s13b_b90k'},
        'open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k': {
            'name': 'open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k',
            'dimensions': 768,
            'note': 'open_clip models',
            'type': 'open_clip',
            'pretrained': 'mscoco_finetuned_laion2b_s13b_b90k'}
    }
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
                "notes": ""},

            "hf/e5-small":
                {"name": 'intfloat/e5-small',
                 "dimensions": 384,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 0.1342,
                 "notes": ""},
            "hf/e5-base":
                {"name": 'intfloat/e5-base',
                 "dimensions": 768,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-large":
                {"name": 'intfloat/e5-large',
                 "dimensions": 1024,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 1.3,
                 "notes": ""},
            "hf/e5-large-unsupervised":
                {"name": 'intfloat/e5-large-unsupervised',
                 "dimensions": 1024,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 1.3,
                 "notes": ""},
            "hf/e5-base-unsupervised":
                {"name": 'intfloat/e5-base-unsupervised',
                 "dimensions": 768,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-small-unsupervised":
                {"name": 'intfloat/e5-small-unsupervised',
                 "dimensions": 384,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 0.134,
                 "notes": ""},
            "hf/multilingual-e5-small":
                {"name": 'intfloat/multilingual-e5-small',
                 "dimensions": 384,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.471,
                 "notes": ""},
            "hf/multilingual-e5-base":
                {"name": 'intfloat/multilingual-e5-base',
                 "dimensions": 768,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 1.11,
                 "notes": ""},
            "hf/multilingual-e5-large":
                {"name": 'intfloat/multilingual-e5-large',
                 "dimensions": 1024,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 2.24,
                 "notes": ""},
            "hf/e5-small-v2":
                {"name": 'intfloat/e5-small-v2',
                 "dimensions": 384,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.134,
                 "notes": ""},
            "hf/e5-base-v2":
                {"name": 'intfloat/e5-base-v2',
                 "dimensions": 768,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-large-v2":
                {"name": 'intfloat/e5-large-v2',
                 "dimensions": 1024,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 1.34,
                 "text_query_prefix": "query: ",     # Only putting this in 1 model for testing purposes.
                 "text_chunk_prefix": "passage: ",
                 "notes": ""},
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
            "prefix-test-model":
                {
                    "name": "sentence-transformers/all-MiniLM-L6-v1",
                    "dimensions": 16,
                    "tokens": 128,
                    "type": "test",
                    "notes": "",
                    "text_query_prefix": "test query: ",
                    "text_chunk_prefix": "test passage: "
                }
    }
    return TEST_MODEL_PROPERTIES

def _get_onnx_clip_properties() -> Dict:
    ONNX_CLIP_MODEL_PROPERTIES = {
        "onnx32/openai/ViT-L/14":
            {
                "name":"onnx32/openai/ViT-L/14",
                "dimensions" : 768,
                "type":"clip_onnx",
                "note":"the onnx float32 version of openai ViT-L/14",
                "repo_id": "Marqo/onnx-openai-ViT-L-14",
                "visual_file": "onnx32-openai-ViT-L-14-visual.onnx",
                "textual_file": "onnx32-openai-ViT-L-14-textual.onnx",
                "token": None,
                "resolution" : 224,
            },
        "onnx16/openai/ViT-L/14":
            {
                "name": "onnx16/openai/ViT-L/14",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float16 version of openai ViT-L/14",
                "repo_id": "Marqo/onnx-openai-ViT-L-14",
                "visual_file": "onnx16-openai-ViT-L-14-visual.onnx",
                "textual_file": "onnx16-openai-ViT-L-14-textual.onnx",
                "token": None,
                "resolution" : 224,
            },
        "onnx32/open_clip/ViT-L-14/openai":
            {
                "name": "onnx32/open_clip/ViT-L-14/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-L-14/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx32-open_clip-ViT-L-14-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai"
            },
        "onnx16/open_clip/ViT-L-14/openai":
            {
                "name": "onnx16/open_clip/ViT-L-14/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-L-14/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai"
            },
        "onnx32/open_clip/ViT-L-14/laion400m_e32":
            {
                "name" : "onnx32/open_clip/ViT-L-14/laion400m_e32",
                "dimensions" : 768,
                "type" : "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-L-14/lainon400m_e32",
                "repo_id" : "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file" : "onnx32-open_clip-ViT-L-14-laion400m_e32-visual.onnx",
                "textual_file" : "onnx32-open_clip-ViT-L-14-laion400m_e32-textual.onnx",
                "token" : None,
                "resolution" : 224,
                "pretrained" : "laion400m_e32"
            },
        "onnx16/open_clip/ViT-L-14/laion400m_e32":
            {
                "name": "onnx16/open_clip/ViT-L-14/laion400m_e32",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-L-14/lainon400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-laion400m_e32-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32"
            },
        "onnx32/open_clip/ViT-L-14/laion2b_s32b_b82k":
            {
                "name": "onnx32/open_clip/ViT-L-14/laion2b_s32b_b82k",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-L-14/laion2b_s32b_b82k",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx32-open_clip-ViT-L-14-laion2b_s32b_b82k-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-laion2b_s32b_b82k-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laionb_s32b_b82k",
                "image_mean" : (0.5, 0.5, 0.5),
                "image_std" : (0.5, 0.5, 0.5),

            },
        "onnx16/open_clip/ViT-L-14/laion2b_s32b_b82k":
            {
                "name": "onnx16/open_clip/ViT-L-14/laion2b_s32b_b82k",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-L-14/laion2b_s32b_b82k",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-laion2b_s32b_b82k-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-laion2b_s32b_b82k-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laionb_s32b_b82k",
                "image_mean": (0.5, 0.5, 0.5),
                 "image_std": (0.5, 0.5, 0.5),
            },
        "onnx32/open_clip/ViT-L-14-336/openai":
            {
                "name": "onnx32/open_clip/ViT-L-14-336/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-L-14-336/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14-336",
                "visual_file": "onnx32-open_clip-ViT-L-14-336-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-336-openai-textual.onnx",
                "token": None,
                "resolution": 336,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,

            },
        "onnx16/open_clip/ViT-L-14-336/openai":
            {
                "name": "onnx16/open_clip/ViT-L-14-336/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-L-14-336/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14-336",
                "visual_file": "onnx16-open_clip-ViT-L-14-336-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-336-openai-textual.onnx",
                "token": None,
                "resolution": 336,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/openai":
            {
                "name": "onnx32/open_clip/ViT-B-32/openai",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-B-32/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/openai":
            {
                "name": "onnx16/open_clip/ViT-B-32/openai",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-B-32/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion400m_e31":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion400m_e31",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-B-32/laion400m_e31",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion400m_e31-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion400m_e31-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e31",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/laion400m_e31":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion400m_e31",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-B-32/laion400m_e31",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion400m_e31-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion400m_e31-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e31",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion400m_e32":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion400m_e32",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-B-32/laion400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion400m_e32-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32",
                "image_mean": None,
                "image_std": None,
            },
        "onnx16/open_clip/ViT-B-32/laion400m_e32":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion400m_e32",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-B-32/laion400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion400m_e32-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion2b_e16":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion2b_e16",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float32 version of open_clip ViT-B-32/laion2b_e16",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion2b_e16-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion2b_e16-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion2b_e16",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/laion2b_e16":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion2b_e16",
                "dimensions": 512,
                "type": "clip_onnx",
                "note": "the onnx float16 version of open_clip ViT-B-32/laion2b_e16",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion2b_e16-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion2b_e16-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion2b_e16",
                "image_mean": None,
                "image_std": None,
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/openai':
            {
              'name': 'onnx32/open_clip/ViT-B-32-quickgelu/openai',
              'dimensions': 512,
              'type': 'clip_onnx',
              'note': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/openai',
              'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
              'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-openai-visual.onnx',
              'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-openai-textual.onnx',
              'token': None,
              'resolution': 224, 'pretrained': 'openai',
              'image_mean': None,
              'image_std': None
             },

        'onnx16/open_clip/ViT-B-32-quickgelu/openai':
            {
               'name': 'onnx16/open_clip/ViT-B-32-quickgelu/openai',
               'dimensions': 512,
               'type': 'clip_onnx',
               'note': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/openai',
               'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
               'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-openai-visual.onnx',
               'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-openai-textual.onnx',
               'token': None,
               'resolution': 224,
               'pretrained': 'openai',
               'image_mean': None,
               'image_std': None
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/openai':
            {
                'name': 'onnx16/open_clip/ViT-B-16/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-16/openai',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
             },

        'onnx32/open_clip/ViT-B-16/openai':
            {
                'name': 'onnx32/open_clip/ViT-B-16/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-16/openai',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-16/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-16/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-16/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-16/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-16/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-16/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-16/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-16/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e31',
                'dimensions': 640,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-16-plus-240/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e31',
                'dimensions': 640, 'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-16-plus-240/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e32',
                'dimensions': 640,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-B-16-plus-240/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e32',
                'dimensions': 640, 'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-B-16-plus-240/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-H-14/laion2b_s32b_b79k':
            {
                'name': 'onnx16/open_clip/ViT-H-14/laion2b_s32b_b79k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-H-14/laion2b_s32b_b79k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-H-14',
                'visual_file': 'onnx16-open_clip-ViT-H-14-laion2b_s32b_b79k-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-H-14-laion2b_s32b_b79k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s32b_b79k',
                'image_mean': None,
                'image_std': None,
            },

        'onnx32/open_clip/ViT-H-14/laion2b_s32b_b79k':
            {
                'name': 'onnx32/open_clip/ViT-H-14/laion2b_s32b_b79k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-H-14/laion2b_s32b_b79k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-H-14',
                'visual_file': 'onnx32-open_clip-ViT-H-14-laion2b_s32b_b79k-visual.zip',
                'textual_file': 'onnx32-open_clip-ViT-H-14-laion2b_s32b_b79k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s32b_b79k',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-g-14/laion2b_s12b_b42k':
            {
                'name': 'onnx16/open_clip/ViT-g-14/laion2b_s12b_b42k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip ViT-g-14/laion2b_s12b_b42k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-g-14',
                'visual_file': 'onnx16-open_clip-ViT-g-14-laion2b_s12b_b42k-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-g-14-laion2b_s12b_b42k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s12b_b42k',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k':
            {
                'name': 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip ViT-g-14/laion2b_s12b_b42k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-g-14',
                'visual_file': 'onnx32-open_clip-ViT-g-14-laion2b_s12b_b42k-visual.zip',
                'textual_file': 'onnx32-open_clip-ViT-g-14-laion2b_s12b_b42k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s12b_b42k',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/openai':
            {
                'name': 'onnx16/open_clip/RN50/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/openai':
            {
                'name': 'onnx32/open_clip/RN50/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN50/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN50/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/cc12m':
            {
                'name': 'onnx16/open_clip/RN50/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-cc12m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/cc12m':
            {
                'name': 'onnx32/open_clip/RN50/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-cc12m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/openai':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/openai':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/cc12m':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50-quickgelu/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-cc12m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/cc12m':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50-quickgelu/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-cc12m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101/openai':
            {
                'name': 'onnx16/open_clip/RN101/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN101/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx16-open_clip-RN101-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN101/openai':
            {
                'name': 'onnx32/open_clip/RN101/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN101/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx32-open_clip-RN101-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None,
            },

        'onnx16/open_clip/RN101/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN101/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN101/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx16-open_clip-RN101-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None,
            },

        'onnx32/open_clip/RN101/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN101/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN101/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx32-open_clip-RN101-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101-quickgelu/openai':
            {
                'name': 'onnx16/open_clip/RN101-quickgelu/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN101-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx16-open_clip-RN101-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN101-quickgelu/openai':
            {
                'name': 'onnx32/open_clip/RN101-quickgelu/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN101-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx32-open_clip-RN101-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101-quickgelu/yfcc15m':
            {'name': 'onnx16/open_clip/RN101-quickgelu/yfcc15m',
             'dimensions': 512,
             'type': 'clip_onnx',
             'note': 'the onnx float16 version of open_clip RN101-quickgelu/yfcc15m',
             'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
             'visual_file': 'onnx16-open_clip-RN101-quickgelu-yfcc15m-visual.onnx',
             'textual_file': 'onnx16-open_clip-RN101-quickgelu-yfcc15m-textual.onnx',
             'token': None,
             'resolution': 224,
             'pretrained': 'yfcc15m',
             'image_mean': None,
             'image_std': None
             },

        'onnx32/open_clip/RN101-quickgelu/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN101-quickgelu/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN101-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx32-open_clip-RN101-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x4/openai':
            {
                'name': 'onnx16/open_clip/RN50x4/openai',
                'dimensions': 640,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50x4/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x4',
                'visual_file': 'onnx16-open_clip-RN50x4-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x4-openai-textual.onnx',
                'token': None,
                'resolution': 288,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x4/openai':
            {
                'name': 'onnx32/open_clip/RN50x4/openai',
                'dimensions': 640,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50x4/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x4',
                'visual_file': 'onnx32-open_clip-RN50x4-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x4-openai-textual.onnx',
                'token': None,
                'resolution': 288,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x16/openai':
            {
                'name': 'onnx16/open_clip/RN50x16/openai',
                'dimensions': 768,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50x16/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x16',
                'visual_file': 'onnx16-open_clip-RN50x16-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x16-openai-textual.onnx',
                'token': None,
                'resolution': 384,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x16/openai':
            {
                'name': 'onnx32/open_clip/RN50x16/openai',
                'dimensions': 768,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50x16/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x16',
                'visual_file': 'onnx32-open_clip-RN50x16-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x16-openai-textual.onnx',
                'token': None,
                'resolution': 384,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x64/openai':
            {
                'name': 'onnx16/open_clip/RN50x64/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float16 version of open_clip RN50x64/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x64',
                'visual_file': 'onnx16-open_clip-RN50x64-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x64-openai-textual.onnx',
                'token': None,
                'resolution': 448,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x64/openai':
            {
                'name': 'onnx32/open_clip/RN50x64/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'note': 'the onnx float32 version of open_clip RN50x64/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x64',
                'visual_file': 'onnx32-open_clip-RN50x64-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x64-openai-textual.onnx',
                'token': None,
                'resolution': 448,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },
    }
    return ONNX_CLIP_MODEL_PROPERTIES


def _get_fp16_clip_properties() -> Dict:
    FP16_CLIP_MODEL_PROPERTIES = {
        "fp16/ViT-L/14": {
            "name": "fp16/ViT-L/14",
            "dimensions": 768,
            "type": "fp16_clip",
            "notes": "The faster version (fp16, load from `cuda`) of openai clip model"
        },
        'fp16/ViT-B/32':
            {"name": "fp16/ViT-B/32",
             "dimensions": 512,
             "notes": "The faster version (fp16, load from `cuda`) of openai clip model",
             "type": "fp16_clip",
             },
        'fp16/ViT-B/16':
            {"name": "fp16/ViT-B/16",
             "dimensions": 512,
             "notes": "The faster version (fp16, load from `cuda`) of openai clip model",
             "type": "fp16_clip",
             },
    }

    return FP16_CLIP_MODEL_PROPERTIES


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
            'hf':HF_MODEL,
            'no_model': NO_MODEL}

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
    onnx_clip_model_properties = _get_onnx_clip_properties()
    multilingual_clip_model_properties = get_multilingual_clip_properties()
    fp16_clip_model_properties = _get_fp16_clip_properties()

    # combine the above dicts
    model_properties = dict(clip_model_properties.items())
    model_properties.update(sbert_model_properties)
    model_properties.update(test_model_properties)
    model_properties.update(sbert_onnx_model_properties)
    model_properties.update(random_model_properties)
    model_properties.update(hf_model_properties)
    model_properties.update(open_clip_model_properties)
    model_properties.update(onnx_clip_model_properties)
    model_properties.update(multilingual_clip_model_properties)
    model_properties.update(fp16_clip_model_properties)

    all_properties = dict()
    all_properties['models'] = model_properties

    all_properties['loaders'] = dict()
    for key,val in _get_model_load_mappings().items():
        all_properties['loaders'][key] = val

    return all_properties
