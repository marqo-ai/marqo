# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
import os
import validators
import requests
import numpy as np
import clip
import torch
from PIL import Image

from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
import transformers

import onnx
import onnxruntime as ort

logger = get_logger(__name__)


def get_allowed_image_types():
    return set(('.jpg', '.png', '.bmp', '.jpeg'))


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


def _load_image_from_path(image: str) -> ImageType:
    """loads an image into PIL from a string path that is
    either local or a url

    Args:
        image (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        ImageType: _description_
    """

    if os.path.isfile(image):
        img = Image.open(image)
    elif validators.url(image):
        img = Image.open(requests.get(image, stream=True).raw)
    else:
        raise ValueError(f"input str of {image} is not a local file or a valid url")

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
        img = _load_image_from_path(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'), 'RGB')

    elif isinstance(image, ImageType):
        img = image
    else:
        raise TypeError(f"input of type {type(image)} did not match allowed types of str, np.ndarray, ImageType")

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
            raise TypeError("received empty list, expected at least one element.")

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
            raise TypeError(
                f"local file [{thing}] extension {extension} does not match allowed file types of {_allowed}")
        else:
            # if it is not a local file and does not have an extension
            # check if url
            if validators.url(thing):
                return True
            else:
                False
                # raise ValueError(f"{thing} cannot be identified as a local file, url or image")

    # if it is an array, then it is an image
    elif isinstance(thing, (ImageType, ndarray)):
        return True
    else:
        raise TypeError(f"expected type Image or str for inputs but received type {type(thing)}")


class JINA_ONNX_CLIP(object):
    """
    Load a clip model and convert it to onnx version for faster inference
    """

    def __init__(self, model_name, device = "cpu", embedding_dim: int = None, truncate: bool = True,
                 visual_path="clip_visual.onnx", textual_path="clip_textual.onnx",
                 load=True, **kwargs):
        self.model_name = model_name
        self.clip_name = model_name.split("jina_onnx/")[1]
        self.clip_model = None
        self.clip_preprocess = None
        self.device = device
        self.image_onnx = None
        self.text_onnx = None
        self.visual_session = None
        self.text_session = None
        self.onnx_model = None
        self.truncate = truncate


    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    def clip_load(self):
        if self.clip_model is None or self.clip_preprocess is None:
            self.clip_model, self.clip_preprocess = clip.load(self.clip_name, device=self.device, jit=False)

    def encode_text(self, sentence, normalize=True):
        if self.visual_session is None:
            self.load()

        tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        result = tokenizer(
            sentence,
            max_length=77,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        text_results = {
            'input_ids': result['input_ids'].numpy().astype("int32"),
            'attention_mask': result['attention_mask'].numpy().astype("int32"),
        }

        outputs = torch.tensor(self.text_session.run(None, text_results)[0])

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode_image(self, images, normalize=True):

        if self.visual_session is None:
            self.load()

        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        self.image_input_processed = torch.stack([self.clip_preprocess(_img).to(self.device) for _img in image_input])

        self.images_onnx = self.image_input_processed.detach().cpu().numpy()

        outputs = torch.tensor(self.visual_session.run(None, {"pixel_values" : self.images_onnx})[0])


        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]],
               default: str = 'text', normalize=True, **kwargs) -> FloatTensor:

        if self.visual_session is None:
            self.load()

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

    def load(self):
        self.clip_load()
        self.visual_session = ort.InferenceSession("visual.onnx", providers=["CUDAExecutionProvider"])
        self.text_session = ort.InferenceSession("textual.onnx", providers=["CUDAExecutionProvider"])

