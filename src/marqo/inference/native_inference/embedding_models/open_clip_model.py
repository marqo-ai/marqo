import os

import numpy as np
import open_clip
import torch
from PIL.Image import Image
from open_clip.pretrained import _pcfg, _slpcfg, _apcfg
from open_clip.transform import image_transform_v2, PreprocessCfg, merge_preprocess_dict
from pydantic import ValidationError
from torchvision.transforms import Compose

from marqo import marqo_docs
from marqo.exceptions import InternalError
from marqo.inference.model_download.model_download import download_model
from marqo.inference.native_inference.embedding_models.abstract_clip_model import AbstractCLIPModel
from marqo.inference.native_inference.embedding_models.abstract_clip_model import AbstractCLIPPreprocessor
from marqo.inference.native_inference.embedding_models.hf_tokenizer import HFTokenizer
from marqo.inference.native_inference.embedding_models.open_clip_model_properties import OpenCLIPModelProperties, \
    ImagePreprocessor
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.logging import get_logger
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation

logger = get_logger(__name__)

HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"


class OpenCLIPPreprocessor(AbstractCLIPPreprocessor):

    def __init__(self, tokenizer, image_preprocessor, device: str):
        super().__init__(tokenizer, image_preprocessor)
        self.device = device

    def _tokenize_text(self, inputs: list[str]) -> List[Tensor]:
        """
        Preprocess the text using the tokenizer.
        Args:
            inputs: A list of strings to preprocess.

        Returns:
            A list of preprocessed images in the form of tensors.
            Each tensor has the shape (N, M) where N is the batch_size,
             M is the length of the tokenized text.
        """
        return [self.tokenizer(text).to(self.device) for text in inputs]

    def _preprocess_image(self, inputs: list[Image]) -> List[Tensor]:
        """
        Preprocess the images using the image preprocessor.
        Args:
            inputs: A list of images to preprocess.

        Returns:
            A list of preprocessed images in the form of tensors.
            Each tensor has the shape (N, 3, H, W) where N is the batch_size,
             H and W are the height and width of the image.
        """
        # Need unsqueeze(0) to add the batch dimension
        return [self.image_preprocessor(image).unsqueeze(0).to(self.device) for image in inputs]


class OpenCLIPModel(AbstractCLIPModel):
    def __init__(
            self,
            device: Optional[str] = None,
            model_properties: Optional[Dict] = None,
            model_auth: Optional[ModelAuth] = None,
    ) -> None:

        super().__init__(device=device, model_properties=model_properties, model_auth=model_auth)

        self.model_properties = self._build_model_properties(model_properties)

        self.image_preprocessor_config = None

    def _build_model_properties(self, model_properties: dict) -> OpenCLIPModelProperties:
        """Convert the user input model_properties to OpenCLIPModelProperties."""
        try:
            return OpenCLIPModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {model_properties}. Original error: {e}") \
                from e

    def _load_necessary_components(self) -> None:
        """Load the open_clip model and tokenizer."""
        if self.model_properties.url is not None or self.model_properties.model_location is not None or \
                self.model_properties.localpath is not None:
            self.model, self.image_preprocessor = self._load_model_and_image_preprocessor_from_checkpoint()
            self.tokenizer = self._load_tokenizer_from_checkpoint()
        elif self.model_properties.name.startswith(HF_HUB_PREFIX):
            self.model, self.image_preprocessor = self._load_model_and_image_preprocessor_from_hf_repo()
            self.tokenizer = self._load_tokenizer_from_hf_repo()
        elif self.model_properties.name.startswith(MARQO_OPEN_CLIP_REGISTRY_PREFIX):
            self.model, self.image_preprocessor = self._load_model_and_image_preprocessor_from_open_clip_repo()
            self.tokenizer = self._load_tokenizer_from_open_clip_repo()
        else:
            raise InvalidModelPropertiesError(
                f"Marqo cannot load the provided open_clip model. "
                f"Check {marqo_docs.bring_your_own_model()} "
                f"for more details on the supported methods to open_clip model "
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocessor = OpenCLIPPreprocessor(self.tokenizer, self.image_preprocessor, device=self.device)

    def get_preprocessor(self) -> OpenCLIPPreprocessor:
        return self.preprocessor

    def _check_loaded_components(self):
        """Check if the open_clip model, tokenizer, and image preprocessor are loaded.

        Raises:
            RuntimeError: If the open_clip model, tokenizer, or image preprocessor is not loaded.
        """
        if self.model is None:
            raise RuntimeError("The open_clip model is not loaded. Please load the model before inference.")
        if self.tokenizer is None:
            raise RuntimeError("The open_clip tokenizer is not loaded. Please load the tokenizer before inference.")
        if self.image_preprocessor is None:
            raise RuntimeError("The open_clip image preprocessor is not loaded. "
                               "Please load the image preprocessor before inference.")

    def _load_image_preprocessor(self) -> Callable:
        return image_transform_v2(self.image_preprocessor_config)

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
        if self.model_properties.localpath:
            if os.path.exists(self.model_properties.localpath):
                self.model_path = self.model_properties.localpath
            else:
                raise InvalidModelPropertiesError(
                    f"The localpath '{self.model_properties.localpath}' does not exist. "
                    f"Please provide a valid localpath to load the model. If you are running Marqo in a container, "
                    f"make sure the localpath is mounted correctly."
                )
        elif self.model_properties.model_location:
            self.model_path = self._download_from_repo()
        elif self.model_properties.url:
            self.model_path = download_model(url=self.model_properties.url)
        else:
            raise InternalError("One of 'localpath', 'model_location', or 'url' must be provided to load the model.")

        logger.info(f"The name of the custom clip model is {self.model_properties.name}. We use open_clip loader")

        try:
            self.image_preprocessor_config = self._aggregate_image_preprocessor_config()
            preprocess = image_transform_v2(self.image_preprocessor_config, is_train=False)
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
            if self.model_properties.name.startswith(HF_HUB_PREFIX):
                return open_clip.get_tokenizer(self.model_properties.name)
            else:
                # Replace '/'with '-' to support old clip model name style
                return open_clip.get_tokenizer(self.model_properties.name.replace("/", "-"))
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
            download_model_params['auth'] = self.model_auth

        model_file_path = download_model(**download_model_params)
        if model_file_path is None or model_file_path == '':
            raise RuntimeError(
                'download_model() needs to return a valid filepath to the model! Instead, received '
                f' filepath `{model_file_path}`')
        return model_file_path

    def encode_image(self, images: List[Tensor], normalize=True) -> List[ndarray]:
        images = torch.cat(images, dim=0)

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = self.model.encode_image(images).to(torch.float32)
            else:
                outputs = self.model.encode_image(images).to(torch.float32)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode_text(self, text: List[Tensor], normalize=True) -> List[ndarray]:
        text = torch.cat(text, dim=0)

        if self.model is None:
            self.load()
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