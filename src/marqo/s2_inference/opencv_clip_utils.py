# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
import os
import validators
import requests
import numpy as np
import clip
import torch
import cv2
import open_clip

from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger

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


def _load_image_from_path(image: str) -> ndarray:
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
        img = cv2.imread(image)
    elif validators.url(image):
        resp = requests.get(image, stream = True).raw
        img = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype="uint8"), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"input str of {image} is not a local file or a valid url")
    return _convert_to_rgb(img)

def _convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType]) -> ndarray:
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
        img = image

    elif isinstance(image, ImageType):
        img = np.array(image)
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


class FULLOPENCV_CLIP:
    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(self, model_type: str = "fullopencv/ViT-B/32", device: str = 'cpu', embedding_dim: int = None,
                 truncate: bool = True, **kwargs) -> None:

        self.model_type = model_type.split("fullopencv/")[1]
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_dimension = embedding_dim
        self.truncate = truncate

    def load(self) -> None:
        # https://github.com/openai/CLIP/issues/30
        self.model, _ = clip.load(self.model_type, device='cpu', jit=False)
        self.preprocess = self.opencv_process()
        self.model = self.model.to(self.device)
        self.tokenizer = clip.tokenize
        self.model.eval()

    def opencv_process(self):
        from augmennt import transforms as at
        from torchvision.transforms import Normalize

        at_transform = at.Compose([
            at.Resize(224, interpolation="BICUBIC"),
            at.CenterCrop(224),
            at.ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return at_transform

    @staticmethod
    def _convert_to_ndarray(image):
        return np.array(image)

    def _convert_output(self, output):

        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:

        if self.model is None:
            self.load()

        text = self.tokenizer(sentence, truncate=self.truncate).to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                     normalize=True) -> FloatTensor:

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
               default: str = 'text', normalize=True, **kwargs) -> FloatTensor:

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
            return self.encode_image(inputs, normalize=normalize)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=normalize)


