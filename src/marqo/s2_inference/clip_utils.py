import os
from marqo.tensor_search.enums import ModelProperties, InferenceParams
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
import validators
import requests
import numpy as np
import clip
import torch
from PIL import Image, UnidentifiedImageError
import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.errors import IncompatibleModelDeviceError, InvalidModelPropertiesError
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from marqo.s2_inference.processing.custom_clip_utils import HFTokenizer, download_model
from torchvision.transforms import InterpolationMode
from marqo.s2_inference.configs import ModelCache

logger = get_logger(__name__)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
BICUBIC = InterpolationMode.BICUBIC


def get_allowed_image_types():
    return set(('.jpg', '.png', '.bmp', '.jpeg'))


def _convert_image_to_rgb(image: ImageType) -> ImageType:
    # Take a PIL.Image.Image and return its RGB version
    return image.convert("RGB")


def _get_transform(n_px: int, image_mean:List[float] = None, image_std: List[float] = None) -> torch.Tensor:
    '''This function returns a transform to preprocess the image. The processed image will be passed into
    clip model for inference.
    Args:
        n_px: the size of the processed image
        image_mean: the mean of the image used for normalization
        image_std: the std of the image used for normalization

    Returns:
        the processed image tensor with shape (3, n_px, n_px)
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


def format_and_load_CLIP_images(images: List[Union[str, ndarray, ImageType]], image_download_headers: dict) -> List[ImageType]:
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
        results.append(format_and_load_CLIP_image(image, image_download_headers))
    
    return results


def load_image_from_path(image_path: str, image_download_headers: dict, timeout=3) -> ImageType:
    """Loads an image into PIL from a string path that is either local or a url

    Args:
        image_path (str): Local or remote path to image.
        image_download_headers (dict): header for the image download
        timeout (number): timeout (in seconds)
    Raises:
        ValueError: If the local path is invalid, and is not a url
        UnidentifiedImageError: If the image is irretrievable or unprocessable.

    Returns:
        ImageType: In-memory PIL image.
    """
    if os.path.isfile(image_path):
        img = Image.open(image_path)
    elif validators.url(image_path):
        try:
            resp = requests.get(image_path, stream=True, timeout=timeout, headers=image_download_headers)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.RequestException
                ) as e:
            raise UnidentifiedImageError(
                f"image url `{image_path}` is unreachable, perhaps due to timeout. "
                f"Timeout threshold is set to {timeout} seconds."
                f"\nConnection error type: `{e.__class__.__name__}`")
        if not resp.ok:
            raise UnidentifiedImageError(f"image url `{image_path}` returned {resp.status_code}. Reason: {resp.reason}")
        img = Image.open(resp.raw)
    else:
        raise UnidentifiedImageError(f"input str of `{image_path}` is not a local file or a valid url")

    return img


def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType], image_download_headers: dict) -> ImageType:
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
        img = load_image_from_path(image, image_download_headers)
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

    def __init__(self, model_type: str = "ViT-B/32", device: str = None,  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:

        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_dimension = embedding_dim
        self.truncate = truncate
        self.model_properties = kwargs.get("model_properties", dict())

        # model_auth gets passed through add_docs and search requests:
        self.model_auth = kwargs.get(InferenceParams.model_auth, None)


    def _download_from_repo(self):
        """Downloads model from an external repo like s3 and returns the filepath

        Returns:
            The model's filepath

        Raises:
            RunTimeError if an empty filepath is detected. This is important
                because OpenCLIP will instantiate a model with random weights, if
                a filepath isn't specified, and the model isn't a publicly
                available HF or OpenAI one.
        """
        model_location = ModelLocation(**self.model_properties[ModelProperties.model_location])
        download_model_params = {"repo_location": model_location}

        if model_location.auth_required:
            download_model_params['auth'] = self.model_auth

        model_file_path = download_model(**download_model_params)
        if model_file_path is None or model_file_path == '':
            raise RuntimeError(
                'download_model() needs to return a valid filepath to the model! Instead, received '
                f' filepath `{model_file_path}`')
        return model_file_path

    def load(self) -> None:

        model_location_presence = ModelProperties.model_location in self.model_properties

        path = self.model_properties.get("localpath", None) or self.model_properties.get("url",None)

        if path is None and not model_location_presence:
            # We must load the model into CPU then transfer it to the desired device, always
            # The original method to load the openai clip model
            # https://github.com/openai/CLIP/issues/30
            self.model, self.preprocess = clip.load(self.model_type, device='cpu', jit=False, download_root=ModelCache.clip_cache_path)
            self.model = self.model.to(self.device)
            self.tokenizer = clip.tokenize
        else:
            logger.info("Detecting custom clip model path. We use generic clip model loading.")
            if path and model_location_presence:
                raise InvalidModelPropertiesError(
                    "Only one of `url`, `localpath` or `model_location can be specified in "
                    "model_properties`. Please ensure that only one of these is specified in "
                    "model_properties and retry.")
            if model_location_presence:
                self.model_path = self._download_from_repo()
            elif os.path.isfile(path):
                self.model_path = path
            elif validators.url(path):
                self.model_path = download_model(url=path)
            else:
                raise InvalidModelPropertiesError(f"Marqo can not load the custom clip model."
                                                  f"The provided model path `{path}` is neither a local file nor a valid url."
                                                  f"Please check your provided model url and retry"
                                                  f"Check `https://docs.marqo.ai/0.0.12/Models-Reference/dense_retrieval/` for more info.")

            self.jit = self.model_properties.get("jit", False)
            self.model, self.preprocess = self.custom_clip_load()
            self.tokenizer = clip.tokenize

            self.model.eval()

    def custom_clip_load(self):
        self.model_name = self.model_properties.get("name", None)

        logger.info(f"The name of the custom clip model is {self.model_name}. We use openai clip load")
        # We must load the model into CPU then transfer it to the desired device, always
        model, preprocess = clip.load(name=self.model_path, device="cpu", jit= self.jit, download_root=ModelCache.clip_cache_path)
        model = model.to(self.device)
        return model, preprocess


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

        text = self.tokenizer(sentence, truncate=self.truncate).to(self.device)

        with torch.no_grad():
            outputs =  self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                    normalize = True, image_download_headers: Optional[Dict] = None) -> FloatTensor:
        
        if self.model is None:
            self.load()
        if image_download_headers is None:
            image_download_headers = dict()

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images, image_download_headers)
        else:
            image_input = [format_and_load_CLIP_image(images, image_download_headers)]

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
            image_download_headers = kwargs.get("image_download_headers", dict())
            return self.encode_image(inputs, normalize=normalize, image_download_headers=image_download_headers)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=normalize)


class FP16_CLIP(CLIP):
    def __init__(self, model_type: str = "fp16/ViT-B/32", device: str = None,  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:
        super().__init__(model_type, device, embedding_dim, truncate, **kwargs)
        '''This class loads the provided clip model directly from cuda in float16 version. The inference time is halved
        with very minor accuracy drop. 
        '''

        if not self.device.startswith("cuda"):
            logger.warning(f"The fp16 clip model `{self.model_type} is loaded with device `{self.device}`."
                              f"FP16 clip model `{self.model_type}` is only available with device `cuda`.\n"
                              f"With current device `{self.device}`, the model will be loaded in `float32` mode. \n"
                              f"Please check you cuda availability or try the fp32 version `{self.model_type.replace('fp16/','')}`"
                              f"Check `https://docs.marqo.ai/0.0.13/Models-Reference/dense_retrieval/#generic-clip-models` for more info.")

        self.model_name = self.model_type.replace("fp16/", "")


    def load(self) -> None:
        # https://github.com/openai/CLIP/issues/30
        self.model, self.preprocess = clip.load(self.model_name, device=self.device, jit=False, download_root=ModelCache.clip_cache_path)
        self.model = self.model.to(self.device)
        self.tokenizer = clip.tokenize
        self.model.eval()


class OPEN_CLIP(CLIP):
    def __init__(self, model_type: str = "open_clip/ViT-B-32-quickgelu/laion400m_e32", device: str = None,  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:
        super().__init__(model_type, device,  embedding_dim, truncate , **kwargs)
        self.model_name = model_type.split("/", 3)[1] if model_type.startswith("open_clip/") else model_type
        self.pretrained = model_type.split("/", 3)[2] if model_type.startswith("open_clip/") else model_type

    def load(self) -> None:
        # https://github.com/mlfoundations/open_clip
        path = self.model_properties.get("localpath", None) or self.model_properties.get("url", None)

        model_location_presence = ModelProperties.model_location in self.model_properties
        if path is None and not model_location_presence:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name,
                                                                                   pretrained=self.pretrained,
                                                                                   device=self.device, jit=False, cache_dir=ModelCache.clip_cache_path)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
        else:
            if path and model_location_presence:
                raise InvalidModelPropertiesError(
                    "Only one of `url`, `localpath` or `model_location can be specified in "
                    "model_properties`. Please ensure that only one of these is specified in "
                    "model_properties and retry.")
            logger.info("Detecting custom clip model path. We use generic clip model loading.")
            if model_location_presence:
                self.model_path = self._download_from_repo()
            elif os.path.isfile(path):
                self.model_path = path
            elif validators.url(path):
                self.model_path = download_model(url=path)
            else:
                raise InvalidModelPropertiesError(
                    f"Marqo cannot load the custom clip model. "
                    f"The provided model path `{path}` is neither a local file nor a valid url. "
                    f"Please check your provided model url and retry. "
                    f"Check `https://docs.marqo.ai/0.0.13/Models-Reference/dense_retrieval/#generic-clip-models` for more info.")

            self.precision = self.model_properties.get("precision", "fp32")
            self.jit = self.model_properties.get("jit", False)
            self.mean = self.model_properties.get("mean", None)
            self.std = self.model_properties.get("std", None)

            self.model, self.preprocess = self.custom_clip_load()
            self.tokenizer = self.load_tokenizer()

            self.model.eval()

    def custom_clip_load(self):
        self.model_name = self.model_properties.get("name", None)

        logger.info(f"The name of the custom clip model is {self.model_name}. We use open_clip load")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name, jit = self.jit, pretrained=self.model_path, precision = self.precision,
            image_mean=self.mean, image_std=self.std, device = self.device, cache_dir=ModelCache.clip_cache_path)

        return model, preprocess

    def load_tokenizer(self):
        tokenizer_name = self.model_properties.get("tokenizer", "clip")

        if tokenizer_name == "clip":
            return open_clip.tokenize
        else:
            logger.info(f"Custom HFTokenizer is provided. Loading...")
            return HFTokenizer(tokenizer_name)


    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                     image_download_headers: Optional[Dict] = None,
                     normalize=True) -> FloatTensor:

        if self.model is None:
            self.load()
        if image_download_headers is None:
            image_download_headers = dict()
        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images, image_download_headers)
        else:
            image_input = [format_and_load_CLIP_image(images, image_download_headers)]

        self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])

        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu"):
            outputs = self.model.encode_image(self.image_input_processed).to(torch.float32)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)


    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:

        if self.model is None:
            self.load()

        text = self.tokenizer(sentence).to(self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu"):
            outputs = self.model.encode_text(text).to(torch.float32)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)


class MULTILINGUAL_CLIP(CLIP):
    def __init__(self, model_type: str = "multilingual-clip/ViT-L/14", device: str = None,  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:

        self.model_name = model_type
        self.model_info = get_multilingual_clip_properties()[self.model_name]
        self.visual_name = self.model_info["visual_model"]
        self.textual_name = self.model_info["textual_model"]
        self.device = device
        self.tokenizer = None
        self.preprocess = None


    def load(self) -> None:
        if self.visual_name.startswith("openai/"):
            clip_name = self.visual_name.replace("openai/", "")
            # We must load the model into CPU then transfer it to the desired device, always
            # The reason is this issue: https://github.com/openai/CLIP/issues/30
            self.visual_model, self.preprocess = clip.load(name = clip_name, device = "cpu", jit = False, download_root=ModelCache.clip_cache_path)
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
                     normalize=True, image_download_headers: Optional[dict] = None) -> FloatTensor:

        if self.visual_model is None:
            self.load()
        if image_download_headers is None:
            image_download_headers = dict()

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images, image_download_headers)
        else:
            image_input = [format_and_load_CLIP_image(images, {})]

        self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])

        with torch.no_grad():
            outputs = self.visual_model.forward(self.image_input_processed)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)


def get_multilingual_clip_properties() -> Dict:
    """This is moved here from the model registry to avoid a circular import"""
    # Models are from github repo
    # https://github.com/FreddeFrallan/Multilingual-CLIP
    MULTILINGUAL_CLIP_PROPERTIES = {
        "multilingual-clip/XLM-Roberta-Large-Vit-L-14":
            {
                "name": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                "visual_model": "openai/ViT-L/14",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                "dimensions": 768,
                "type": "multilingual_clip",
            },

        "multilingual-clip/XLM-R Large Vit-B/16+":
            {
                "name": "multilingual-clip/XLM-R Large Vit-B/16+",
                "visual_model": "open_clip/ViT-B-16-plus-240/laion400m_e32",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
                "dimensions": 640,
                "type": "multilingual_clip",
            },

        "multilingual-clip/XLM-Roberta-Large-Vit-B-32":
            {
                "name": "multilingual-clip/XLM-Roberta-Large-Vit-B-32",
                "visual_model": "openai/ViT-B/32",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-B-32',
                "dimensions": 512,
                "type": "multilingual_clip",
            },

        "multilingual-clip/LABSE-Vit-L-14":
            {
                "name": "multilingual-clip/LABSE-Vit-L-14",
                "visual_model": "openai/ViT-L/14",
                "textual_model": 'M-CLIP/LABSE-Vit-L-14',
                "dimensions": 768,
                "type": "multilingual_clip",
            }
    }
    return MULTILINGUAL_CLIP_PROPERTIES



