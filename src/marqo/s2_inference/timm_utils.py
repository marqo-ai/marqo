import timm
import torch
from tqdm import tqdm
import os
import json
from pathlib import Path

import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.clip_utils import format_and_load_CLIP_images, format_and_load_CLIP_image
from marqo.s2_inference.types import *
from marqo.s2_inference.clip_utils import CLIP

logger = get_logger(__name__)

class TimmModel(CLIP):

    # https://github.com/rwightman/pytorch-image-models

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.config = None
        self.transform = None
        
        self.model = None
        self.preprocess = None

    def load(self):
        loaded = _load_timm(self.model_name, self.device)
        self.model = loaded['model']
        self.preprocess = loaded['preprocess']

    def encode(self, images: Union[str, ImageType, List[Union[str, ImageType]]], 
                        normalize = True) -> FloatTensor:
        
        if self.model is None:
            self.load()

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])
    
        with torch.no_grad():
            outputs = self.model(self.image_input_processed)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

def _load_timm(model_name, device):

    # default is ~/.cache/torch/hub/checkpoints/
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    preprocess = create_transform(**config)
    return {'model':model, 'preprocess':preprocess}

def _create_registry(excluded_models=[]):

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    available_models = timm.list_models(pretrained=True)

    available_models = [m for m in available_models if m not in excluded_models]

    models = dict()
    logger.info("constructing timm registry...")
    for model_name in tqdm(available_models[:10]):
        m = TimmModel(model_name, 'cpu')
        m.load()

        out = m.encode(filename)

        models[model_name] = {"name": model_name,
                            "dimensions": out.shape[-1],
                            "notes": f"timm {model_name}",
                            "type": "timm",
                            "loader":"timm_utils.TimmModel"
                            }

        del out

        logger.info(f"{models[model_name]}")

    return models

def read_json(filename):

    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dumps(data)

def _update_registry(save_name='./registry/timm_model_registry.json'):

    file_path, filename = os.path.split(save_name)

    if os.path.isfile(save_name):
        timm_registry = 
    else:
        Path(file_path).mkdir(parents=True, exist_ok=True)


# def _get_embedding_dim(model_name):
