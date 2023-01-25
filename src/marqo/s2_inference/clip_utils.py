# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
import os

import PIL.Image
import validators
import requests
import numpy as np
import clip
import torch
from PIL import Image, UnidentifiedImageError
import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers
from clip.model import build_model
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
import marqo.s2_inference.model_registry as model_registry
from marqo.s2_inference.errors import InvalidModelDeviceError, InvalidModelPropertiesError
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from marqo.s2_inference.processing.custom_clip_utils import HFTokenizer, download_pretrained_from_url

logger = get_logger(__name__)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def get_allowed_image_types():
    return set(('.jpg', '.png', '.bmp', '.jpeg'))

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image: ImageType) -> ImageType:
    # Take a PIL.Image.Image and return its RGB version
    return image.convert("RGB")


def _get_transform(n_px: int, image_mean:List[float] = None, image_std: List[float] = None) -> torch.Tensor:
    '''

    Args:
        n_px: the size of the processed image
        image_mean: the mean of the image used for normalization
        image_std: the std of the image used for normalization

    Returns:
        the processed image tensor wit shape (3, n_px, n_px)

    '''
    img_mean = image_mean or OPENAI_DATASET_MEAN
    img_std = image_std or OPENAI_DATASET_STD
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(img_mean, img_std),
    ])



def format_and_load_CLIP_images(images: List[Union[str, ndarray, ImageType]]) -> List[ImageType]:
    """takes in a list of strings, arrays or urls and either loads and/or converts to PIL
        for the clip model

    Args:
        images (List[Union[str, np.ndarray, ImageType]]): list of file locations or arrays (can be mixed)

    Raises:
        TypeError: _description_

    Returns:
        List[ImageType]: list of PIL images
    """
    if not isinstance(images, list):
        raise TypeError(f"expected list but received {type(images)}")

    results = []
    for image in images:
        results.append(format_and_load_CLIP_image(image))
    
    return results

def load_image_from_path(image_path: str) -> ImageType:
    """Loads an image into PIL from a string path that is either local or a url

    Args:
        image_path (str): Local or remote path to image.

    Raises:
        ValueError: If the local path is invalid, and is not a url
        UnidentifiedImageError: If the image is irretrievable or unprocessable.

    Returns:
        ImageType: In-memory PIL image.
    """
    
    if os.path.isfile(image_path):
        img = Image.open(image_path)
    elif validators.url(image_path):
        resp = requests.get(image_path, stream=True)
        if not resp.ok:
            raise UnidentifiedImageError(f"image url {image_path} returned a {resp.status_code}. Reason {resp.reason}")
        img = Image.open(resp.raw)
    else:
        raise UnidentifiedImageError(f"input str of {image_path} is not a local file or a valid url")

    return img

def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType]) -> ImageType:
    """standardizes the input to be a PIL image

    Args:
        image (Union[str, np.ndarray, ImageType]): can be a local file, url or array

    Raises:
        ValueError: _description_
        TypeError: _description_

    Returns:
        ImageType: PIL image
    """
    # check for the input type
    if isinstance(image, str):
        img = load_image_from_path(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'), 'RGB')

    elif isinstance(image, ImageType):
        img = image
    else:
        raise UnidentifiedImageError(f"input of type {type(image)} did not match allowed types of str, np.ndarray, ImageType")

    return img


def _is_image(inputs: Union[str, List[Union[str, ImageType, ndarray]]]) -> bool:
    # some logic to determine if something is an image or not
    # assume the batch is the same type
    # maybe we use something like this https://github.com/ahupp/python-magic
    
    _allowed = get_allowed_image_types()

    # we assume the batch is this way if a list
    # otherwise apply over each element
    if isinstance(inputs, list):

        if len(inputs) == 0:
            raise UnidentifiedImageError("received empty list, expected at least one element.")

        thing = inputs[0]
    else:
        thing = inputs
    
    # if it is a string, determine if it is a local file or url
    if isinstance(thing, str):
        name, extension = os.path.splitext(thing.lower())
        
        # if it has the correct extension, asssume yes
        if extension in _allowed:
            return True
        
        # if it is a local file without extension, then raise an error
        if os.path.isfile(thing):
            # we could also read the first part of the file and infer
            raise UnidentifiedImageError(f"local file [{thing}] extension {extension} does not match allowed file types of {_allowed}")
        else:
            # if it is not a local file and does not have an extension
            # check if url
            if validators.url(thing):
                return True
            else:
                False

    # if it is an array, then it is an image
    elif isinstance(thing, (ImageType, ndarray)):
        return True
    else:
        raise UnidentifiedImageError(f"expected type Image or str for inputs but received type {type(thing)}")

class CLIP:
    
    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(self, model_type: str = "ViT-B/32", device: str = 'cpu',  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:

        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_dimension = embedding_dim
        self.truncate = truncate
        self.model_properties = kwargs["model_properties"]

    def load(self) -> None:

        path = self.model_properties.get("localpath", None) or self.model_properties.get("url", None)

        if path is None:
            # The original method to load the openai clip model
            # https://github.com/openai/CLIP/issues/30
            self.model, self.preprocess = clip.load(self.model_type, device='cpu', jit=False)
            self.model = self.model.to(self.device)
            self.tokenizer = clip.tokenize
        else:
            logger.info("Detecting custom clip model path. We use generic model loading.")
            if os.path.isfile(path):
                self.model_path = path
            elif validators.url(path):
                self.model_path = download_pretrained_from_url(path)

            self.jit = self.model_properties.get("jit", False)
            self.device = self.model_properties.get("device", "cpu")
            self.mean = self.model_properties.get("mean", None)
            self.std = self.model_properties.get("std", None)


            self.model, self.preprocess = self.custom_clip_load()
            self.model.eval()

            self.tokenizer = self.load_tokenizer()


    def custom_clip_load(self):
        # This function can load both openai clip and open_clip models
        # Check https://github.com/mlfoundations/open_clip/blob/db7504f070b4e76e6c8578ee7b73596267083a19/src/clip/openai_clip.py#L121-L189
        try:
            # loading JIT archive
            model = torch.jit.load(self.model_path, map_location=self.device if self.jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if self.jit:
                self.jit = False
            state_dict = torch.load(self.model_path, map_location="cpu")

        if not self.jit:
            try:
                model = build_model(state_dict or model.state_dict()).to(self.device)
            except KeyError:
                sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
                model = build_model(sd).to(self.device)

            if str(self.device) == "cpu":
                model.float()
            return model, _get_transform(model.visual.input_resolution, self.mean, self.std)

        # patch the device names
        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(self.device)), example_inputs=[])
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

        def patch_device(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)

        # patch dtype to float32 on CPU
        if str(self.device) == "cpu":
            float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
            float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
            float_node = float_input.node()

            def patch_float(module):
                graphs = [module.graph] if hasattr(module, "graph") else []
                if hasattr(module, "forward1"):
                    graphs.append(module.forward1.graph)

                for graph in graphs:
                    for node in graph.findAllNodes("aten::to"):
                        inputs = list(node.inputs())
                        for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                            if inputs[i].node()["value"] == 5:
                                inputs[i].node().copyAttributes(float_node)

            model.apply(patch_float)
            patch_float(model.encode_image)
            patch_float(model.encode_text)

            model.float()

        return model, _get_transform(model.visual.input_resolution, self.mean, self.std)


    def load_tokenizer(self):
        tokenizer_name = self.model_properties.get("tokenizer", "clip")

        if tokenizer_name == "clip":
            return clip.tokenize
        else:
            return HFTokenizer(tokenizer_name)


    def _convert_output(self, output):

        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def encode_text(self, sentence: Union[str, List[str]], normalize = True) -> FloatTensor:
        
        if self.model is None:
            self.load()
        try:
            text = self.tokenizer(sentence, truncate=self.truncate).to(self.device)
        except Exception:
            text = self.tokenizer(sentence).to(self.device)


        with torch.no_grad():
            outputs =  self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]], 
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
            outputs = self.model.encode_image(self.image_input_processed)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]], 
                                default: str = 'text', normalize = True, **kwargs) -> FloatTensor:

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
                raise UnidentifiedImageError(f"expected default='image' or default='text' but received {default}")

        if is_image:
            logger.debug('image')
            return self.encode_image(inputs, normalize=normalize)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=normalize)


class FP16_CLIP(CLIP):
    def __init__(self, model_type: str = "fp16/ViT-B/32", device: str = 'cuda',  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:
        super().__init__(model_type, device, embedding_dim, truncate, **kwargs)

        if not self.device.startswith("cuda"):
            raise InvalidModelDeviceError(f"FP16 clip model `{self.model_type}` is only available with device `cuda`.")

        self.model_name = self.model_type.replace("fp16/", "")


    def load(self) -> None:

        # https://github.com/openai/CLIP/issues/30
        self.model, self.preprocess = clip.load(self.model_name, device='cuda', jit=False)
        self.model = self.model.to(self.device)
        self.tokenizer = clip.tokenize
        self.model.eval()

class OPEN_CLIP(CLIP):
    def __init__(self, model_type: str = "open_clip/ViT-B-32-quickgelu/laion400m_e32", device: str = 'cpu',  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:
        super().__init__(model_type, device,  embedding_dim, truncate , **kwargs)
        self.model_name = model_type.split("/", 3)[1]
        self.pretrained = model_type.split("/", 3)[2]


    def load(self) -> None:
        # https://github.com/mlfoundations/open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained = self.pretrained, device=self.device, jit=False)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:

        if self.model is None:
            self.load()

        text = self.tokenizer(sentence).to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)


class MULTILINGUAL_CLIP(CLIP):
    def __init__(self, model_type: str = "multilingual-clip/ViT-L/14", device: str = 'cpu',  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:

        self.model_name = model_type
        self.model_info = model_registry._get_multilingual_clip_properties()[self.model_name]
        self.visual_name = self.model_info["visual_model"]
        self.textual_name = self.model_info["textual_model"]
        self.device = device
        self.tokenizer = None
        self.preprocess = None


    def load(self) -> None:
        if self.visual_name.startswith("openai/"):
            clip_name = self.visual_name.replace("openai/", "")
            self.visual_model, self.preprocess = clip.load(name = clip_name, device = "cpu", jit = False)
            self.visual_model = self.visual_model.to(self.device)
            self.visual_model = self.visual_model.visual

        elif self.visual_name.startswith("open_clip/"):
            clip_name = self.visual_name.replace("open_clip/", "")
            self.visual_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name=clip_name.split("/")[0], pretrained= clip_name.split("/")[1], device = self.device)
            self.visual_model = self.visual_model.visual

        self.textual_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.textual_name, self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.textual_name)

        self.textual_model.eval()
        self.visual_model.eval()

    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:

        if self.textual_model is None:
            self.load()

        with torch.no_grad():
            outputs = self.textual_model.forward(sentence, self.tokenizer)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                     normalize=True) -> FloatTensor:

        if self.visual_model is None:
            self.load()

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])

        with torch.no_grad():
            outputs = self.visual_model.forward(self.image_input_processed)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)









