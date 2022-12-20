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
from timeit import default_timer as timer

logger = get_logger(__name__)

_HF_MODE_DOWNLOAD = {
    "onnx32/openai/ViT-L/14":
        {
            "repo_id": "Marqo/onnx32-openai-ViT-L-14",
            "visual_file": "onnx32-openai-ViT-L-14-visual.onnx",
            "textual_file": "onnx32-openai-ViT-L-14-textual.onnx",
            "token": None
        },
}



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
        start = timer()
        img = _load_image_from_path(image)
        end = timer()
        print(f"Loading image from path time {round((end - start)*1000)} ms")
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
        self.model_info = _HF_MODE_DOWNLOAD[self.model_name]

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
        sentence = clip.tokenize(sentence, truncate=self.truncate).cpu()
        sentence_onnx = sentence.detach().cpu().numpy().astype(np.int32)

        outputs = torch.squeeze(torch.tensor(np.array(self.textual_session.run(None, {"input":sentence_onnx}))))

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)


    def encode_image(self, images, normalize=True):
        start = timer()
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]
        end = timer()
        print(f"Image loading Time = {round((end - start) * 1000)}ms")

        image_input_processed = torch.stack([self.clip_preprocess(_img).to(self.device) for _img in image_input])
        images_onnx = image_input_processed.detach().cpu().numpy().astype(np.float32)

        # The onnx output has the shape [1,1,768], we need to squeeze the dimension
        outputs = torch.squeeze(torch.tensor(np.array(self.visual_session.run(None, {"input": images_onnx}))))

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
        self.visual_session.disable_fallback()
        self.textual_session.disable_fallback()

    @staticmethod
    def download_model(repo_id:str, filename:str, cache_folder:str = None) -> str:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename,
                                 cache_dir=cache_folder)
        return file_path










# class ONNX_CLIP_16(ONNX_CLIP):
#     def __init__(self, model_name, device="cpu", embedding_dim: int = None, truncate: bool = True,
#                  load=True, **kwargs):
#
#         self.model_name = model_name
#         self.clip_name = model_name.split("onnx16/")[1]
#         self.clip_model = None
#         self.clip_preprocess = None
#         self.device = device
#         self.image_onnx = None
#         self.text_onnx = None
#         self.visual_path = "onnx-" + self.clip_name.replace("/", "-") + "-visual"
#         self.textual_path = "onnx-" + self.clip_name.replace("/", "-") + "-textual"
#         self.onnx_model = None
#         self.truncate = truncate
#         self.providers = ["CPUExecutionProvider", ]
#         self.tokenize = None
#         if self.device == "cuda":
#             self.providers = ['CUDAExecutionProvider',] + self.providers
#         self.visual_path_fp16 = "onnx16-" + self.clip_name.replace("/", "-") + "-visual"
#         self.textual_path_fp16 = "onnx16-" + self.clip_name.replace("/", "-") + "-textual"
#
#
#     def clip_load(self):
#         if self.clip_model is None or self.clip_preprocess is None or self.tokenize is None:
#             self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
#             self.tokenize = open_clip.get_tokenizer('ViT-L-14')
#
#     def onnx_converter(self):
#         self.clip_load()
#         if self.image_onnx is None or self.text_onnx is None:
#             dummy_input = np.random.rand(1000, 1000, 3) * 255
#             dummy_input = dummy_input.astype("uint8")
#
#             image = self.clip_preprocess(Image.fromarray(dummy_input).convert("RGB")).unsqueeze(0).cpu()
#
#             text = self.tokenize(["a diagram", "a dog", "a cat"]).cpu()
#
#             self.onnx_model = clip_onnx(self.clip_model, visual_path=self.visual_path,
#                                         textual_path=self.textual_path)
#             self.onnx_model.convert2onnx(image, text, verbose=True)
#
#
#             print("Start float16-onnx Conversion")
#             self.visual_model_fp16 = float16_converter.convert_float_to_float16_model_path(self.visual_path)
#             self.textual_model_fp16 = float16_converter.convert_float_to_float16_model_path(self.textual_path)
#
#             onnx.save_model(self.visual_model_fp16, self.visual_path_fp16)
#             onnx.save_model(self.textual_model_fp16, self.textual_path_fp16)
#
#             self.load_onnx()
#
#
#     def load_onnx(self):
#         self.clip_load()
#         print("Loading visual_session and textual_session for onnx-float16")
#         self.visual_session = onnxruntime.InferenceSession(self.visual_path_fp16,
#                                                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#         self.textual_session = onnxruntime.InferenceSession(self.textual_path_fp16,
#                                                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#
#     def encode_text(self, sentence, normalize=True):
#         sentence = self.tokenize(sentence).cpu()
#         sentence_onnx = sentence.detach().cpu().numpy().astype(np.int64)
#         outputs = torch.tensor(self.textual_session.run(None, {"input":sentence_onnx})).to(self.device)[0]
#
#         if normalize:
#             _shape_before = outputs.shape
#             outputs /= self.normalize(outputs)
#             assert outputs.shape == _shape_before
#         return self._convert_output(outputs)
#
#     def encode_image(self, images, normalize=True):
#         if isinstance(images, list):
#             image_input = format_and_load_CLIP_images(images)
#         else:
#             image_input = [format_and_load_CLIP_image(images)]
#
#
#         start1 = timer()
#         self.image_input_processed = torch.stack([self.clip_preprocess(_img).to(self.device) for _img in image_input])
#         self.images_onnx = self.image_input_processed.detach().cpu().numpy().astype(np.float16)
#         end1  = timer()
#
#         start2 = timer()
#         outputs = torch.tensor(self.visual_session.run(None, {"input":self.images_onnx})).to(self.device)[0]
#
#         if normalize:
#             _shape_before = outputs.shape
#             outputs /= self.normalize(outputs)
#             assert outputs.shape == _shape_before
#         if self.device.startswith("cuda"):
#             torch.cuda.synchronize(device=None)
#         end2 = timer()
#
#         print(f"preprocessing time {round((end1-start1)*1000)}ms, encoding time {round((end2 - start2)*1000)}ms")
#         return self._convert_output(outputs)
#
#     def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]],
#                default: str = 'text', normalize=True, **kwargs) -> FloatTensor:
#
#         infer = kwargs.pop('infer', True)
#
#         if infer and _is_image(inputs):
#             is_image = True
#         else:
#             is_image = False
#             if default == 'text':
#                 is_image = False
#             elif default == 'image':
#                 is_image = True
#             else:
#                 raise ValueError(f"expected default='image' or default='text' but received {default}")
#
#         if is_image:
#             logger.debug('image')
#             return self.encode_image(inputs, normalize=True)
#         else:
#             logger.debug('text')
#             return self.encode_text(inputs, normalize=True)