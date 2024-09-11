import os
from io import BytesIO
from platform import architecture

import certifi
import clip
import numpy as np
import open_clip
import pycurl
import requests
import torch
import transformers
import validators
from PIL import Image, UnidentifiedImageError
from multilingual_clip import pt_multilingual_clip
from open_clip.pretrained import _pcfg, _slpcfg, _apcfg
from open_clip.transform import image_transform_v2, PreprocessCfg, merge_preprocess_dict
from requests.utils import requote_uri
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from marqo import marqo_docs
from marqo.api.exceptions import InternalError
from marqo.core.inference.models.abstract_clip_model import AbstractCLIPModel
from marqo.core.inference.models.open_clip_model_properties import OpenCLIPModelProperties, ImagePreprocessor
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError, ImageDownloadError
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.processing.custom_clip_utils import HFTokenizer, download_model
from marqo.s2_inference.types import *
from marqo.tensor_search.enums import ModelProperties, InferenceParams
from marqo.tensor_search.models.private_models import ModelLocation
from marqo.tensor_search.telemetry import RequestMetrics

logger = get_logger(__name__)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
BICUBIC = InterpolationMode.BICUBIC
DEFAULT_HEADERS = {'User-Agent': 'Marqobot/1.0'}
HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"

def get_allowed_image_types():
    return set(('.jpg', '.png', '.bmp', '.jpeg'))


def _convert_image_to_rgb(image: ImageType) -> ImageType:
    # Take a PIL.Image.Image and return its RGB version
    return image.convert("RGB")


def _get_transform(n_px: int, image_mean: List[float] = None, image_std: List[float] = None) -> torch.Tensor:
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


def format_and_load_CLIP_images(images: List[Union[str, ndarray, ImageType]], image_download_headers: dict) -> List[
    ImageType]:
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


def load_image_from_path(image_path: str, image_download_headers: dict, timeout_ms=3000,
                         metrics_obj: Optional[RequestMetrics] = None) -> ImageType:
    """Loads an image into PIL from a string path that is either local or a url

    Args:
        image_path (str): Local or remote path to image.
        image_download_headers (dict): header for the image download
        timeout_ms (int): timeout (in milliseconds), for the whole request
    Raises:
        ValueError: If the local path is invalid, and is not a url
        UnidentifiedImageError: If the image is irretrievable or unprocessable.

    Returns:
        ImageType: In-memory PIL image.
    """
    if os.path.isfile(image_path):
        img = Image.open(image_path)
    elif validate_url(image_path):
        if metrics_obj is not None:
            metrics_obj.start(f"image_download.{image_path}")
        try:
            img_io: BytesIO = download_image_from_url(image_path, image_download_headers, timeout_ms)
            img = Image.open(img_io)
        except ImageDownloadError as e:
            raise UnidentifiedImageError(str(e)) from e
        finally:
            if metrics_obj is not None:
                metrics_obj.stop(f"image_download.{image_path}")
    else:
        raise UnidentifiedImageError(f"Input str of {image_path} is not a local file or a valid url. "
                                     f"If you are using Marqo Cloud, please note that images can only be downloaded "
                                     f"from a URL and local files are not supported. "
                                     f"If you are running Marqo in a Docker container, you will need to use a Docker "
                                     f"volume so that your container can access host files. "
                                     f"For more information, please refer to: "
                                     f"{marqo_docs.indexing_images()}")

    return img


def validate_url(url: str) -> bool:
    """Validate a URL to ensure it is a valid URL. Returns True if the URL is valid or the encoded URL is valid.
    Args:
        url (str): URL to validate.
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if isinstance(url, str):
        return validators.url(url) or validators.url(encode_url(url))
    else:
        return False



def download_image_from_url(image_path: str, image_download_headers: dict, timeout_ms: int = 3000) -> BytesIO:
    """Download an image from a URL and return a PIL image using pycurl.

    Args:
        image_path (str): URL to the image.
        image_download_headers (dict): Headers for the image download.
        timeout_ms (int): Timeout in milliseconds, for the whole request.

    Returns:
        buffer (BytesIO): The image as a BytesIO object.

    Raises:
        ImageDownloadError: If the image download fails.
    """

    if not isinstance(timeout_ms, int):
        raise InternalError(f"timeout must be an integer but received {timeout_ms} of type {type(timeout_ms)}")

    try:
        encoded_url = encode_url(image_path)
    except UnicodeEncodeError as e:
        raise ImageDownloadError(f"Marqo encountered an error when downloading the image url {image_path}. "
                                 f"The url could not be encoded properly. Original error: {e}")
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.CAINFO, certifi.where())
    c.setopt(pycurl.URL, encoded_url)
    c.setopt(pycurl.WRITEDATA, buffer)
    c.setopt(pycurl.TIMEOUT_MS, timeout_ms)
    c.setopt(pycurl.FOLLOWLOCATION, 1)

    headers = DEFAULT_HEADERS.copy()
    headers.update(image_download_headers)
    c.setopt(pycurl.HTTPHEADER, [f"{k}: {v}" for k, v in headers.items()])

    try:
        c.perform()
        if c.getinfo(pycurl.RESPONSE_CODE) != 200:
            raise ImageDownloadError(f"image url `{image_path}` returned {c.getinfo(pycurl.RESPONSE_CODE)}")
    except pycurl.error as e:
        raise ImageDownloadError(f"Marqo encountered an error when downloading the image url {image_path}. "
                                 f"The original error is: {e}")
    finally:
        c.close()
    buffer.seek(0)
    return buffer


def encode_url(url: str) -> str:
    """
    Encode a URL to a valid format with only ASCII characters and reserved characters using percent-encoding.

    In version 2.8, we replaced the requests library with pycurl for image downloads. Consequently, we need to implement
    the URL encoding function ourselves. This function replicates the encoding behavior of the
    'requests.utils.requote_uri' function from the requests library.

    Args:
        url (str): The URL to encode.

    Returns:
        str: The encoded URL.

    Raises:
        UnicodeEncodeError: If the URL cannot be encoded properly.

    """
    return requests.utils.requote_uri(url)


def format_and_load_CLIP_image(image: Union[str, ndarray, ImageType, Tensor],
                               image_download_headers: dict) -> Union[ImageType, Tensor]:
    """standardizes the input to be a PIL image

    Args:
        image (Union[str, np.ndarray, ImageType, Tensor]): can be a local file, url, array or a tensor

    Raises:
        ValueError: _description_
        TypeError: _description_

    Returns:
        standardized the image:
            ImageType: PIL image if input is a string, an array or a PIL image
            Tensor: torch tensor if input is a torch tensor
    """
    # check for the input type
    if isinstance(image, str):
        img = load_image_from_path(image, image_download_headers)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, torch.Tensor):
        img = image
    elif isinstance(image, ImageType):
        img = image
    else:
        raise UnidentifiedImageError(f"input of type {type(image)} "
                                     f"did not match allowed types of str, np.ndarray, ImageType, Tensor")

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
            raise UnidentifiedImageError(
                f"local file [{thing}] extension {extension} does not match allowed file types of {_allowed}")
        else:
            # if it is not a local file and does not have an extension
            # check if url
            if validate_url(thing):
                return True
            else:
                return False

    # if it is an array, then it is an image
    elif isinstance(thing, (ImageType, ndarray, Tensor)):
        return True
    else:
        raise UnidentifiedImageError(f"expected type Image or str for inputs but received type {type(thing)}")


class CLIP:
    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(self, model_type: str = "ViT-B/32", device: str = None, embedding_dim: int = None,
                 truncate: bool = True, **kwargs) -> None:

        self.model_type = model_type

        if not device:
            raise InternalError("`device` is required for loading CLIP models!")
        self.device = device

        self.model = None
        self.tokenizer = None
        self.preprocess = None

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
        path = self.model_properties.get("localpath", None) or self.model_properties.get("url", None)

        if path is None and not model_location_presence:
            # We must load the model into CPU then transfer it to the desired device, always
            # The original method to load the openai clip model
            # https://github.com/openai/CLIP/issues/30
            self.model, self.preprocess = clip.load(self.model_type, device='cpu', jit=False,
                                                    download_root=ModelCache.clip_cache_path)
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
        model, preprocess = clip.load(name=self.model_path, device="cpu", jit=self.jit,
                                      download_root=ModelCache.clip_cache_path)
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

    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:

        if self.model is None:
            self.load()

        text = self.tokenizer(sentence, truncate=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def _preprocess_images(self, images: Union[str, ImageType, List[Union[str, ImageType, Tensor]], Tensor],
                           image_download_headers: Optional[Dict] = None) -> Tensor:
        """Preprocess the input image to be ready for the model.

        Args:
            images (Union[str, ImageType, List[Union[str, ImageType, Tensor]], Tensor]): input image,
            can be a str(url), a PIL image, or a tensor, or a list of them
            image_download_headers (Optional[Dict]): headers for the image download
        Return:
            Tensor: the processed image tensor with shape (batch_size, channel, n_px, n_px)
        """
        if self.model is None:
            self.load()
        if image_download_headers is None:
            image_download_headers = dict()

        # default to batch encoding
        if isinstance(images, list):
            image_input: List[Union[ImageType, Tensor]] \
                = format_and_load_CLIP_images(images, image_download_headers)
        else:
            image_input: List[Union[ImageType, Tensor]] = [format_and_load_CLIP_image(images, image_download_headers)]

        image_input_processed: Tensor = torch.stack([self.preprocess(_img).to(self.device) \
                                                         if not isinstance(_img, torch.Tensor) else _img \
                                                     for _img in image_input])
        return image_input_processed

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType, Tensor]], Tensor],
                     normalize=True, image_download_headers: Optional[Dict] = None) -> FloatTensor:
        """Encode the input image to a tensor representation.

        Args:
            images (Union[str, ImageType, List[Union[str, ImageType, Tensor]], Tensor]): input image,
            can be a str(url), a PIL image, or a tensor, or a list of them
            normalize (bool): whether to normalize the output tensor
            image_download_headers (Optional[Dict]): headers for the image download
        Return:
            FloatTensor: the encoded image tensor with shape (batch_size, embedding_dim)
        """
        self.image_input_processed: Tensor = self._preprocess_images(images, image_download_headers)
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
                raise UnidentifiedImageError(f"expected default='image' or default='text' but received {default}")

        if is_image:
            logger.debug('image')
            image_download_headers = kwargs.get("image_download_headers", dict())
            return self.encode_image(inputs, normalize=normalize, image_download_headers=image_download_headers)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=normalize)


class FP16_CLIP(CLIP):
    def __init__(self, model_type: str = "fp16/ViT-B/32", device: str = None, embedding_dim: int = None,
                 truncate: bool = True, **kwargs) -> None:
        super().__init__(model_type, device, embedding_dim, truncate, **kwargs)
        '''This class loads the provided clip model directly from cuda in float16 version. The inference time is halved
        with very minor accuracy drop. 
        '''

        if not self.device.startswith("cuda"):
            logger.warning(f"The fp16 clip model `{self.model_type} is loaded with device `{self.device}`."
                           f"FP16 clip model `{self.model_type}` is only available with device `cuda`.\n"
                           f"With current device `{self.device}`, the model will be loaded in `float32` mode. \n"
                           f"Please check you cuda availability or try the fp32 version `{self.model_type.replace('fp16/', '')}`"
                           f"Check `https://docs.marqo.ai/0.0.13/Models-Reference/dense_retrieval/#generic-clip-models` for more info.")

        self.model_name = self.model_type.replace("fp16/", "")

    def load(self) -> None:
        # https://github.com/openai/CLIP/issues/30
        self.model, self.preprocess = clip.load(self.model_name, device=self.device, jit=False,
                                                download_root=ModelCache.clip_cache_path)
        self.model = self.model.to(self.device)
        self.tokenizer = clip.tokenize
        self.model.eval()


class OPEN_CLIP(AbstractCLIPModel):
    def __init__(
            self,
            device: Optional[str] = None,
            model_properties: Optional[Dict] = None,
            model_auth: Optional[Dict] = None,
    ) -> None:

        super().__init__(device, model_properties, model_auth)

        # model_auth gets passed through add_docs and search requests:
        self.preprocess_config = None

    def _build_model_properties(self, model_properties: dict):
        return OpenCLIPModelProperties(**model_properties)

    def _load_necessary_components(self) -> None:
        """Load the open_clip model and tokenizer."""
        if self.model_properties.url is not None or self.model_properties.model_location is not None:
            self.model, self.preprocess = self._load_model_and_image_preprocessor_from_checkpoint()
            self.tokenizer = self._load_tokenizer_from_checkpoint()
        elif self.model_properties.name.startswith(HF_HUB_PREFIX):
            self.model, self.preprocess = self._load_model_and_image_preprocessor_from_hf_repo()
            self.tokenizer = self._load_tokenizer_from_hf_repo()
        elif self.model_properties.name.startswith(MARQO_OPEN_CLIP_REGISTRY_PREFIX):
            self.model, self.preprocess = self._load_model_and_image_preprocessor_from_open_clip_repo()
            self.tokenizer = self._load_tokenizer_from_open_clip_repo()
        else:
            raise InvalidModelPropertiesError(
                f"Marqo cannot load the provided open_clip model. "
                f"Check {marqo_docs.bring_your_own_model()} "
                f"for more details on the supported methods to open_clip model "
            )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _check_loaded_components(self):
        """Check if the open_clip model, tokenizer, and image preprocessor are loaded.

        Raises:
            RuntimeError: If the open_clip model, tokenizer, or image preprocessor is not loaded.
        """
        if self.model is None:
            raise RuntimeError("The open_clip model is not loaded. Please load the model before inference.")
        if self.tokenizer is None:
            raise RuntimeError("The open_clip tokenizer is not loaded. Please load the tokenizer before inference.")
        if self.preprocess is None:
            raise RuntimeError("The open_clip image preprocessor is not loaded. "
                               "Please load the image preprocessor before inference.")

    def _load_image_preprocessor(self) -> Callable:
        return image_transform_v2(self.preprocess_config)

    def _aggregate_image_preprocessor_config(self) -> PreprocessCfg:
        """Aggregate the image preprocessor configuration for the open_clip model."""

        if self.model_properties.image_preprocessor in [ImagePreprocessor.OpenCLIP, ImagePreprocessor.OpenAI]:
            base_image_preprocess_config = _pcfg()
        elif self.model_properties.image_preprocessor in [ImagePreprocessor.SigLIP]:
            base_image_preprocess_config = _slpcfg()
        elif self.model_properties.image_preprocessor in [ImagePreprocessor.CLIPA]:
            base_image_preprocess_config = _apcfg()
        else:
            raise ValueError(f"Invalid image preprocessor {self.model_properties.image_preprocessor}")

        aggregated_image_preprocess_config = PreprocessCfg(
            **merge_preprocess_dict(
                base_image_preprocess_config, self.model_properties.dict(exclude_none=True)
            )
        )

        return aggregated_image_preprocess_config

    def _load_model_and_image_preprocessor_from_checkpoint(self) -> Tuple[torch.nn.Module, Compose]:
        """Load the model and image preprocessor from a checkpoint file.

        The checkpoint file can be provided through a URL or a model_location object.
        """
        # Load the image preprocessor
        if self.model_properties.url and self.model_properties.model_location:
            raise InvalidModelPropertiesError(
                "Only one of url, model_location can be specified in 'model_properties' "
            )
        elif self.model_properties.model_location:
            self.model_path = self._download_from_repo()
        elif self.model_properties.url:
            self.model_path = download_model(url=self.model_properties.url)
        else:
            raise ValueError("The 'url' or 'model_location' is required in 'model_properties' "
                             "when loading a custom open_clip model through a URL or a model_location object")

        logger.info(f"The name of the custom clip model is {self.model_properties.name}. We use open_clip loader")

        try:
            self.preprocess_config = self._aggregate_image_preprocessor_config()
            preprocess = image_transform_v2(self.preprocess_config, is_train=False)
            model = open_clip.create_model(
                model_name=self.model_properties.name,
                jit=self.model_properties.jit,
                pretrained=self.model_path,
                precision=self.model_properties.precision,
                device=self.device,
                cache_dir=ModelCache.clip_cache_path
            )
            return model, preprocess
        except Exception as e:
            if (isinstance(e, RuntimeError) and "The file might be corrupted" in str(e)):
                try:
                    os.remove(self.model_path)
                except Exception as remove_e:
                    raise RuntimeError(
                        f"Marqo encountered an error while attempting to delete a corrupted file '{self.model_path}'. "
                        f"Please report this issue on Marqo's Github Repo and replace the problematic Marqo instance "
                        f"with a new one. \n "
                        f"Error message: `{str(remove_e)}`"
                    )
                raise InvalidModelPropertiesError(
                    f"Marqo encountered a corrupted file when loading open_clip file '{self.model_path}'. "
                    f"Marqo has removed this file from the disk. "
                    f"Some possible causes are: "
                    f"1. the file was not a valid open_clip checkpoint, "
                    f"2. the file was corrupted during download or incompletely downloaded, "
                    f"3. you may have tried to load a clip model even though model_properties['type'] is set to 'open_clip' "
                    f"Please check and update your model properties and retry. "
                    f"You can find more details at {marqo_docs.bring_your_own_model()}")
            # It is tricky to cacth the error when loading clip model using type = open_clip. Different pytorch version will raise different error.
            elif isinstance(e, (AttributeError, RuntimeError)) or (
                    "This could be because the operator doesn't exist for this backend" in str(e)):
                raise InvalidModelPropertiesError(
                    f"Marqo encountered an error when loading custom open_clip model '{self.model_properties.name}' with "
                    f"model properties = '{self.model_properties.dict()}'. "
                    f"The error message is {str(e)}. "
                    f"You may have tried to load a clip model even though model_properties['type'] is set to 'open_clip' "
                    f"Please check and update your model properties and retry. "
                    f"You can find more details at {marqo_docs.bring_your_own_model()}"
                )
            else:
                raise RuntimeError(
                    f"Marqo encountered an error when loading custom open_clip model {self.model_properties.name} with "
                    f"model properties = {self.model_properties.dict()}. "
                    f"The error message is {str(e)}. "
                    f"Please check and update your model properties and retry. "
                    f"You can find more details at {marqo_docs.bring_your_own_model()}"
                )

    def _load_model_and_image_preprocessor_from_hf_repo(self) -> Tuple[torch.nn.Module, Compose]:
        """Load the model and image preprocessor from a hf_repo.

        The hf_repo should be provided in the model properties, and it is a string starting with `hf-hub:`.
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_properties.name,
            device=self.device,
            cache_dir=ModelCache.clip_cache_path,
        )
        return model, preprocess

    def _load_model_and_image_preprocessor_from_open_clip_repo(self) -> Tuple[torch.nn.Module, Compose]:
        """Load the model and image preprocessor from the marqo model registry.

        The model name should be provided in the model properties, and it is a string starting with `open_clip/`.
        """
        architecture = self.model_properties.name.split("/", 3)[1]
        pretrained = self.model_properties.name.split("/", 3)[2]

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=architecture,
            pretrained=pretrained,
            device=self.device,
            cache_dir=ModelCache.clip_cache_path
        )
        return model, preprocess

    def _load_tokenizer_from_checkpoint(self) -> Callable:
        if not self.model_properties.tokenizer:
            return open_clip.get_tokenizer(self.model_properties.name)
        else:
            logger.info(f"Custom HFTokenizer is provided. Loading...")
            return HFTokenizer(self.model_properties.tokenizer)

    def _load_tokenizer_from_hf_repo(self) -> Callable:
        return open_clip.get_tokenizer(self.model_properties.name)

    def _load_tokenizer_from_open_clip_repo(self) -> Callable:
        return open_clip.get_tokenizer(self.model_properties.name.split("/", 3)[1])

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
        model_location: ModelLocation = self.model_properties.model_location
        download_model_params = {"repo_location": model_location}

        if model_location.auth_required:
            download_model_params['auth'] = self.model_properties.model_auth

        model_file_path = download_model(**download_model_params)
        if model_file_path is None or model_file_path == '':
            raise RuntimeError(
                'download_model() needs to return a valid filepath to the model! Instead, received '
                f' filepath `{model_file_path}`')
        return model_file_path

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                     image_download_headers: Optional[Dict] = None,
                     normalize=True) -> FloatTensor:

        self.image_input_processed: Tensor = self._preprocess_images(images, image_download_headers)

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = self.model.encode_image(self.image_input_processed).to(torch.float32)
            else:
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

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = self.model.encode_text(text).to(torch.float32)
            else:
                outputs = self.model.encode_text(text).to(torch.float32)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)


class MULTILINGUAL_CLIP(CLIP):
    def __init__(self, model_type: str = "multilingual-clip/ViT-L/14", device: str = None, embedding_dim: int = None,
                 truncate: bool = True, **kwargs) -> None:

        if not device:
            raise InternalError("`device` is required for loading MULTILINGUAL CLIP models!")

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
            self.visual_model, self.preprocess = clip.load(name=clip_name, device="cpu", jit=False,
                                                           download_root=ModelCache.clip_cache_path)
            self.visual_model = self.visual_model.to(self.device)
            self.visual_model = self.visual_model.visual

        elif self.visual_name.startswith("open_clip/"):
            clip_name = self.visual_name.replace("open_clip/", "")
            self.visual_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=clip_name.split("/")[0], pretrained=clip_name.split("/")[1], device=self.device)
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