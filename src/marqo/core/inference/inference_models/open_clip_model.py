import os

import open_clip
import torch
from open_clip.pretrained import _pcfg, _slpcfg, _apcfg
from open_clip.transform import image_transform_v2, PreprocessCfg, merge_preprocess_dict
from torchvision.transforms import Compose

from marqo import marqo_docs
from marqo.core.inference.inference_models.abstract_clip_model import AbstractCLIPModel
from marqo.core.inference.inference_models.open_clip_model_properties import OpenCLIPModelProperties, ImagePreprocessor
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.logger import get_logger
from marqo.core.inference.inference_models.hf_tokenizer import HFTokenizer
from marqo.core.inference.model_download import download_model
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelLocation

logger = get_logger(__name__)

HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"


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
        """Load the open_clip model and _tokenizer."""
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
        """Check if the open_clip model, _tokenizer, and image preprocessor are loaded.

        Raises:
            RuntimeError: If the open_clip model, _tokenizer, or image preprocessor is not loaded.
        """
        if self.model is None:
            raise RuntimeError("The open_clip model is not loaded. Please load the model before inference.")
        if self.tokenizer is None:
            raise RuntimeError("The open_clip _tokenizer is not loaded. Please load the _tokenizer before inference.")
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
        if not self.model_properties._tokenizer:
            return open_clip.get_tokenizer(self.model_properties.name)
        else:
            logger.info(f"Custom HFTokenizer is provided. Loading...")
            return HFTokenizer(self.model_properties._tokenizer)

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