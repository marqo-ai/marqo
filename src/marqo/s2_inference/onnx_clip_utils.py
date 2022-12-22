# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
import onnx
import os
import validators
import requests
import numpy as np
import clip
import torch
from PIL import Image
import open_clip
from huggingface_hub import hf_hub_download
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
import onnxruntime as ort

# Loading shared functions from clip_utils.py. This part should be decoupled from models in the future
from marqo.s2_inference.clip_utils import get_allowed_image_types, format_and_load_CLIP_image, format_and_load_CLIP_images, load_image_from_path,_is_image

logger = get_logger(__name__)

_HF_MODEL_DOWNLOAD = {

    #Please check the link https://huggingface.co/Marqo for available models.

    
    "onnx32/openai/ViT-L/14":
        {
            "repo_id": "Marqo/onnx-openai-ViT-L-14",
            "visual_file": "onnx32-openai-ViT-L-14-visual.onnx",
            "textual_file": "onnx32-openai-ViT-L-14-textual.onnx",
            "token": None
        },

    "onnx16/openai/ViT-L/14":
        {
            "repo_id": "Marqo/onnx-openai-ViT-L-14",
            "visual_file": "onnx16-openai-ViT-L-14-visual.onnx",
            "textual_file": "onnx16-openai-ViT-L-14-textual.onnx",
            "token": None

        }
}


class CLIP_ONNX(object):
    """
    Load a clip model and convert it to onnx version for faster inference
    """

    def __init__(self, model_name = "onnx32/openai/ViT-L/14", device = "cpu", embedding_dim: int = None, truncate: bool = True,
                 load=True, **kwargs):
        self.model_name = model_name
        self.onnx_type, self.source, self.clip_model = self.model_name.split("/", 2)
        self.device = device
        self.truncate = truncate
        self.provider = ['CUDAExecutionProvider', "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        self.visual_session = None
        self.textual_session = None
        self.model_info = _HF_MODEL_DOWNLOAD[self.model_name]

        if self.onnx_type == "onnx16":
            self.visual_type = np.float16
        elif self.onnx_type == "onnx32":
            self.visual_type = np.float32

    def load(self):
        self.load_clip()
        self.load_onnx()

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    def load_clip(self):
        if self.source == "openai":
            clip_model, self.clip_preprocess = clip.load(self.clip_model, device="cpu", jit=False)
            self.tokenizer = clip.tokenize
            del clip_model
        elif self.source =="open_clip":
            clip_name, pre_trained = self.clip_model.split("/", 2)
            clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_name, pre_trained, device="cpu")
            self.tokenizer = open_clip.get_tokenizer(clip_name)
            del clip_model

    def encode_text(self, sentence, normalize=True):
        text = clip.tokenize(sentence, truncate=self.truncate).cpu()
        text_onnx = text.detach().cpu().numpy().astype(np.int32)

        onnx_input_text = {self.textual_session.get_inputs()[0].name: text_onnx}
        # The onnx output has the shape [1,1,768], we need to squeeze the dimension
        outputs = torch.squeeze(torch.tensor(np.array(self.textual_session.run(None, onnx_input_text))))

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode_image(self, images, normalize=True):
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        image_input_processed = torch.stack([self.clip_preprocess(_img) for _img in image_input])
        images_onnx = image_input_processed.detach().cpu().numpy().astype(self.visual_type)

        onnx_input_image = {self.visual_session.get_inputs()[0].name: images_onnx}
        # The onnx output has the shape [1,1,768], we need to squeeze the dimension
        outputs = torch.squeeze(torch.tensor(np.array(self.visual_session.run(None, onnx_input_image))))

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]],
               default: str = 'text', normalize=True, **kwargs) -> FloatTensor:

        if self.clip_preprocess is None or self.tokenizer is None:
            self.load_clip()
        if self.visual_session is None or self.textual_session is None:
            self.load_onnx()

        infer = kwargs.pop('infer', True)

        if infer and _is_image(inputs):
            is_image = True
        else:
            is_image = False
            if default == 'text':
                is_image = False
            elif default == 'image':
                is_image = True
            else:
                raise ValueError(f"expected default='image' or default='text' but received {default}")

        if is_image:
            logger.debug('image')
            return self.encode_image(inputs, normalize=True)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=True)

    def load_onnx(self):
        self.visual_file = self.download_model(self.model_info["repo_id"], self.model_info["visual_file"])
        self.textual_file = self.download_model(self.model_info["repo_id"], self.model_info["textual_file"])
        self.visual_session = ort.InferenceSession(self.visual_file, providers=self.provider)
        self.textual_session = ort.InferenceSession(self.textual_file, providers=self.provider)

    @staticmethod
    def download_model(repo_id:str, filename:str, cache_folder:str = None) -> str:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename,
                                 cache_dir=cache_folder)
        return file_path










