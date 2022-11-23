# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
from clip_onnx import clip_onnx
import os
import validators
import requests
import numpy as np
import clip
import torch
from PIL import Image
import cv2

from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger

logger = get_logger(__name__)


def get_allowed_image_types():
    return set(('.jpg', '.png', '.bmp', '.jpeg'))


def format_and_load_CLIP_images(images: List[Union[str, ndarray, ImageType]]) -> List[ndarray]:
    """takes in a list of strings, arrays or urls and either loads and/or converts to PIL
        for the clip model

    Args:
        images (List[Union[str, np.ndarray, ImageType]]): list of file locations or arrays (can be mixed)

    Raises:
        TypeError: _description_

    Returns:
        List[ndarray]: list of ndarray images (cv2 support)
    """
    if not isinstance(images, list):
        raise TypeError(f"expected list but received {type(images)}")

    results = []
    for image in images:
        results.append(format_and_load_CLIP_image(image))

    return results

def _convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_image_from_path(image: str) -> ndarray:
    """loads an ndarray image from a string path that is
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

    if isinstance(img, ndarray):
        return _convert_to_rgb(img)
    else:
        raise ValueError(f"input str of {image} is not valid")



def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType]) -> ndarray:
    """standardizes the input to be a PIL image

    Args:
        image (Union[str, np.ndarray, ImageType]): can be a local file, url or array

    Raises:
        ValueError: _description_
        TypeError: _description_

    Returns:
        ndarray: cv2 suporrted image type
    """
    # check for the input type
    if isinstance(image, str):
        img = _load_image_from_path(image)
    elif isinstance(image, np.ndarray):
        img = image.astype("uint8")
    elif isinstance(image, ImageType):
        img = np.array(image).astype("unit8")
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


class Fast_CLIP(object):
    """
    This model uses the open_cv based preprocessing to speed up the preprocessing of the image.

    (NOTE THAT due to different preprocessing strategies, the results might be different from the original clip)

    This image uses the onnx model to speed up the inference speed.
    """

    def __init__(self, model_name, device = "cpu", embedding_dim: int = None, truncate: bool = True,
                 load=True, **kwargs):
        self.model_name = model_name
        self.clip_name = model_name.split("fast/")[1]
        self.clip_model = None
        self.clip_preprocess = None
        self.device = device
        self.image_onnx = None
        self.text_onnx = None
        self.visual_path = "onnx-" + self.clip_name.replace("/", "-") + "-visual"
        self.textual_path = "onnx-" + self.clip_name.replace("/", "-") + "-textual"
        self.onnx_model = None
        self.truncate = truncate
        self.providers = ["CPUExecutionProvider",]
        if self.device == "cuda":
            self.providers = self.providers + ['TensorrtExecutionProvider','CUDAExecutionProvider']

    def load(self):
        try:
            self.load_onnx()
        except:
            print("Can not find existing onnx model. Start converting")
            self.onnx_converter()

    @staticmethod
    def _convert_to_ndarray(image):
        return np.array(image)

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
            self.clip_model, _ = clip.load(self.clip_name, device="cpu", jit=False)
            self.resolution = self.clip_model.visual.input_resolution
            self.clip_preprocess = self.opencv_process()

    def opencv_process(self):
        from augmennt import transforms as at
        from torchvision.transforms import Normalize

        at_transform = at.Compose([
            at.Resize(self.resolution, interpolation="BICUBIC"),
            at.CenterCrop(self.resolution),
            at.ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return at_transform

    def onnx_converter(self):
        self.clip_load()
        if self.image_onnx is None or self.text_onnx is None:
            dummy_input = np.random.rand(1000, 1000, 3) * 255
            dummy_input = dummy_input.astype("uint8")

            image = self.clip_preprocess(dummy_input).unsqueeze(0).cpu()

            text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu()

            self.onnx_model = clip_onnx(self.clip_model, visual_path=self.visual_path, textual_path=self.textual_path)
            self.onnx_model.convert2onnx(image, text, verbose=True)
            self.onnx_model.start_sessions(providers=self.providers)

    def encode_text(self, sentence, normalize=True):
        if self.onnx_model is None:
            self.load()

        sentence = clip.tokenize(sentence, truncate=self.truncate).cpu()
        sentence_onnx = sentence.detach().cpu().numpy().astype(np.int32)
        outputs = torch.tensor(self.onnx_model.encode_text(sentence_onnx))

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode_image(self, images, normalize=True):

        if self.onnx_model is None:
            self.load()

        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        self.image_input_processed = torch.stack([self.clip_preprocess(_img).to(self.device) for _img in image_input])

        self.images_onnx = self.image_input_processed.detach().cpu().numpy().astype(np.float32)
        outputs = torch.tensor(self.onnx_model.encode_image(self.images_onnx))

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]],
               default: str = 'text', normalize=True, **kwargs) -> FloatTensor:

        if self.onnx_model is None:
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

    def load_onnx(self):
        self.clip_load()
        self.onnx_model = clip_onnx(None)
        self.onnx_model.load_onnx(visual_path=self.visual_path,
                                  textual_path=self.textual_path,
                                  logit_scale=100.0000)  # model.logit_scale.exp()
        self.onnx_model.start_sessions(self.providers)



