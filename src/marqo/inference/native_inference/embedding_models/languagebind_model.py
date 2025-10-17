import torch
from PIL.Image import Image
from pydantic.v1 import ValidationError

from marqo.base_model import MarqoBaseModel
from marqo.core.exceptions import InternalError
from marqo.exceptions import InternalError
from marqo.inference.model_download.model_download import (
    download_model_from_hf,
    download_pretrained_from_s3,
    download_pretrained_from_url,
    extract_zip_file,
)
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import (
    AbstractPreprocessor,
)
from marqo.inference.native_inference.embedding_models.languagebind_model_properties import (
    LanguagebindModelProperties,
    ModalityLocation,
)
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError, MediaMismatchError
from marqo.s2_inference.languagebind import (
    LanguageBind,
    LanguageBindImageTokenizer,
    to_device,
    transform_dict,
)
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth


class LanguagebindPreprocessor(AbstractPreprocessor):
    def __init__(self, raw_preprocessor, device):
        super().__init__()
        self._preprocessor = raw_preprocessor
        self._device = device

    def preprocess(self, inputs: Union[List[str], List[Image]], modality: Modality):
        if modality == Modality.TEXT:
            return self._preprocess_text(inputs)
        elif modality == Modality.IMAGE:
            return self._preprocess_image(inputs)
        elif modality == Modality.VIDEO:
            return self._preprocess_video(inputs)
        elif modality == Modality.AUDIO:
            return self._preprocess_audio(inputs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _preprocess_text(self, inputs: List[str]) -> List[str]:
        return inputs  # No preprocessing needed for text

    def _preprocess_image(self, inputs: List[Image]) -> List[Tensor]:
        return [
            self._preprocessor["image"](image, return_tensors="pt")["pixel_values"].to(
                self._device
            )
            for image in inputs
        ]

    def _preprocess_video(self, inputs) -> List[Tensor]:
        return [
            self._preprocessor["video"](video, return_tensors="pt")["pixel_values"].to(
                self._device
            )
            for video in inputs
        ]

    def _preprocess_audio(self, inputs):
        return [
            self._preprocessor["audio"](audio, return_tensors="pt")["pixel_values"].to(
                self._device
            )
            for audio in inputs
        ]


class CLIPType(MarqoBaseModel):
    """A wrapper class that is used to store the model location for each modality.
    The key is the modality and the value is the location of the model.
    A location can either be a HuggingFace repo ID or a directory containing the model files.
    """

    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None


class LanguagebindModel(AbstractEmbeddingModel):
    DEFAULT_TOKENIZER_REPO = "lb203/LanguageBind_Image"

    MODEL_NAME_CLIP_TYPE_MAPPING = {
        "LanguageBind/Video_V1.5_FT_Audio_FT_Image": CLIPType(
            video="LanguageBind/LanguageBind_Video_V1.5_FT",
            audio="LanguageBind/LanguageBind_Audio_FT",
            image="LanguageBind/LanguageBind_Image",
        ),
        "LanguageBind/Video_V1.5_FT_Audio_FT": CLIPType(
            video="LanguageBind/LanguageBind_Video_V1.5_FT",
            audio="LanguageBind/LanguageBind_Audio_FT",
        ),
        "LanguageBind/Video_V1.5_FT_Image": CLIPType(
            video="LanguageBind/LanguageBind_Video_V1.5_FT",
            image="LanguageBind/LanguageBind_Image",
        ),
        "LanguageBind/Audio_FT_Image": CLIPType(
            audio="LanguageBind/LanguageBind_Audio_FT",
            image="LanguageBind/LanguageBind_Image",
        ),
        "LanguageBind/Audio_FT": CLIPType(audio="LanguageBind/LanguageBind_Audio_FT"),
        "LanguageBind/Video_V1.5_FT": CLIPType(
            video="LanguageBind/LanguageBind_Video_V1.5_FT"
        ),
    }

    def __init__(
        self,
        device: Optional[str] = None,
        model_properties: Optional[Dict] = None,
        model_auth: Optional[ModelAuth] = None,
    ) -> None:
        super().__init__(
            model_properties=model_properties, device=device, model_auth=model_auth
        )

        self.model_properties = self._build_model_properties(model_properties)
        self.preprocess_config = None

        self._model = None
        self._tokenizer = None
        self._preprocessors = None
        self._clip_type = None

    def _build_model_properties(self, model_properties):
        try:
            return LanguagebindModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {e}")

    def _load_necessary_components(self):
        self._clip_type = self._generate_clip_type()
        self._load_model()
        self._load_tokenizer()
        self._load_preprocessor()

        self._preprocessor = LanguagebindPreprocessor(self._preprocessors, self.device)

        self._model = self._model.to(self.device)
        self._model.eval()

    def get_preprocessors(self) -> dict:
        return self._preprocessors

    def _check_loaded_components(self):
        if self._model is None:
            raise InternalError("Model was not loaded properly")
        if self._tokenizer is None:
            raise InternalError("Tokenizer was not loaded properly")
        if self._preprocessors is None:
            raise InternalError("Preprocessors were not loaded properly")

    def _load_model(self):
        try:
            token = (
                self.model_auth.hf.token
                if (self.model_auth and self.model_auth.hf)
                else None
            )
            self._model = LanguageBind(
                self._clip_type.dict(exclude_none=True),
                cache_dir=ModelCache.languagebind_cache_path,
                token=token,
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the Languagebind model, "
                f"modelProperties={self.model_properties}. "
                f" Original error message: {e}"
            ) from e

    def _generate_clip_type(self) -> CLIPType:
        """
        Generate a CLIPType object that contains the model location for each modality.

        Returns:
            A CLIPType object that contains the model location for each modality.
        """
        if self.model_properties.name:
            # Loading from a registered Languagebind model
            if self.model_properties.name not in self.MODEL_NAME_CLIP_TYPE_MAPPING:
                raise InvalidModelPropertiesError(
                    f"Model name '{self.model_properties.name}' is not a registered Languagebind model."
                    f"If you are loading a custom model, please provide the modelLocation and remove the 'name' field"
                )
            clip_type = self.MODEL_NAME_CLIP_TYPE_MAPPING[self.model_properties.name]
        elif self.model_properties.modelLocation:
            # Custom model loading
            clip_type_dict = {"image": None, "video": None, "audio": None}
            for modality in self.model_properties.supportedModalities:
                if modality == Modality.TEXT or modality == "text":
                    continue
                model_location: ModalityLocation = getattr(
                    self.model_properties.modelLocation, modality
                )
                if model_location is None:
                    continue
                elif model_location.hf and not model_location.hf.filename:
                    clip_type_dict[modality.value] = model_location.hf.repo_id
                elif (
                    (model_location.hf and model_location.hf.filename)
                    or model_location.s3
                    or model_location.url
                ):
                    downloaded_zip_file = self._download_languagebind_model(
                        model_location
                    )
                    clip_type_dict[modality.value] = extract_zip_file(
                        downloaded_zip_file
                    )
                else:
                    raise InternalError(
                        f"Invalid model location {model_location} provided for modality {modality}"
                    )
            clip_type = CLIPType(**clip_type_dict)
        else:
            raise InvalidModelPropertiesError(
                "Invalid model properties provided. Either 'name' or "
                "'modelLocation' must be provided."
            )
        return clip_type

    def _load_tokenizer(self):
        if self.model_properties.modelLocation:
            self._load_custom_tokenizer()
        else:
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                self.DEFAULT_TOKENIZER_REPO,
                cache_dir=ModelCache.languagebind_cache_path,
            )

    def _load_custom_tokenizer(self):
        """Custom tokenizer loading. The tokenizer can be loaded in two ways:

        1. A huggingface repo, e.g., 'lb203/LanguageBind_Image'
        2. A directory containing the tokenizer files
        """
        tokenizer_location: ModalityLocation = (
            self.model_properties.modelLocation.tokenizer
        )
        if tokenizer_location is None:
            # Use the default tokenizer repo
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                self.DEFAULT_TOKENIZER_REPO,
                cache_dir=ModelCache.languagebind_cache_path,
            )
        elif tokenizer_location.hf and (not tokenizer_location.hf.filename):
            # Loading from a HuggingFace repo
            token = (
                self.model_auth.hf.token
                if (self.model_auth and self.model_auth.hf)
                else None
            )
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                tokenizer_location.hf.repo_id,
                cache_dir=ModelCache.languagebind_cache_path,
                token=token,
            )
        elif (
            (tokenizer_location.hf and tokenizer_location.hf.filename)
            or tokenizer_location.s3
            or tokenizer_location.url
        ):
            # Loading from a directory provided by a zip file
            downloaded_zip_file = self._download_languagebind_model(tokenizer_location)
            extracted_dir = extract_zip_file(downloaded_zip_file)

            try:
                self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                    extracted_dir, cache_dir=ModelCache.languagebind_cache_path
                )
            except (OSError, ValueError, RuntimeError) as e:
                raise InvalidModelPropertiesError(
                    f"Marqo encountered an error loading the Languagebind tokenizer, "
                    f"modelProperties={self.model_properties}. "
                    f" Original error message = {e}"
                ) from e
        else:
            raise InternalError(
                f"Invalid tokenizer location provided for tokenizer: "
                f"{tokenizer_location}"
            )

    def _load_preprocessor(self):
        """Load the preprocessors for each modality.

        It is a dictionary where the key is the modality and the value is the preprocessor function.
        """
        self._preprocessors = {
            c: transform_dict[c](self._model.modality_config[c])
            for c in self._clip_type.dict(exclude_none=True).keys()
        }

    def encode(
        self,
        inputs: Union[List[str], List[Tensor]],
        modality: Modality,
        normalize: bool = True,
    ) -> List[ndarray]:
        if modality not in self.model_properties.supportedModalities:
            raise MediaMismatchError(
                f"The provided modality {modality} is not supported by the model. This model "
                f"supports the following modalities: {self.model_properties.supportedModalities}"
            )

        if modality == Modality.TEXT:
            return self._encode_text(inputs, normalize)
        elif modality == Modality.IMAGE:
            return self._encode_image(inputs, normalize)
        elif modality == Modality.VIDEO:
            return self._encode_video(inputs, normalize)
        elif modality == Modality.AUDIO:
            return self._encode_audio(inputs, normalize)
        else:
            raise NotImplementedError(
                f"Encoding for modality {modality} is not implemented yet. "
            )

    def _encode_text(self, text: list[str], normalize=True) -> List[ndarray]:
        formated_input = dict()
        processed_text = to_device(
            self._tokenizer(
                text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            self.device,
        )
        formated_input["language"] = processed_text

        with torch.no_grad():
            outputs = self._model(formated_input)["language"]

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            if outputs.shape != _shape_before:
                raise InternalError(
                    "Normalization changed the shape of the output tensor."
                )

        return self._convert_output(outputs)

    def _encode_image(self, inputs: List[Tensor], normalize: bool = True):
        """
        Args:
            inputs: A list of processed image tensors.
            normalize: Whether to normalize the output.
        """
        # Format the inputs to the required data structure for the model
        formated_input = {"image": {"pixel_values": torch.cat(inputs, dim=0)}}

        with torch.no_grad():
            outputs = self._model(formated_input)["image"]

        if normalize:
            outputs /= self.normalize(outputs)
            if (
                outputs.shape != outputs.shape
            ):  # Check if normalization changed the shape
                raise InternalError(
                    "Normalization changed the shape of the output tensor."
                )
        return self._convert_output(outputs)

    def _encode_video(self, inputs: List[Tensor], normalize: bool = True):
        """
        Args:
            inputs: A list of processed video tensors.
            normalize: Whether to normalize the output.
        """
        formated_input = {"video": {"pixel_values": torch.cat(inputs, dim=0)}}

        with torch.no_grad():
            outputs = self._model(formated_input)["video"]

        if normalize:
            outputs /= self.normalize(outputs)
            if (
                outputs.shape != outputs.shape
            ):  # Check if normalization changed the shape
                raise InternalError(
                    "Normalization changed the shape of the output tensor."
                )
        return self._convert_output(outputs)

    def _encode_audio(self, inputs: List[Tensor], normalize: bool = True):
        """
        Args:
            inputs: A list of processed audio tensors.
            normalize: Whether to normalize the output.
        """
        formated_input = {"audio": {"pixel_values": torch.cat(inputs, dim=0)}}

        with torch.no_grad():
            outputs = self._model(formated_input)["audio"]

        if normalize:
            outputs /= self.normalize(outputs)
            if (
                outputs.shape != outputs.shape
            ):  # Check if normalization changed the shape
                raise InternalError(
                    "Normalization changed the shape of the output tensor."
                )
        return self._convert_output(outputs)

    def _download_languagebind_model(self, modality_location: ModalityLocation) -> str:
        """Download the Languagebind model zip file via a given location. The location is a ModalityLocation object.
        We have 3 possible locations:
        1. S3Location: a zip file in an S3 bucket.
        2. HFLocation: a zip file in the HuggingFace repo.
        3. URL: a direct download link.

        Args:
            modality_location: The location of the Languagebind model.

        Returns:
            The path of the downloaded Languagebind model zip file.
        """
        if modality_location.url:
            return download_pretrained_from_url(
                modality_location.url, cache_dir=ModelCache.languagebind_cache_path
            )
        elif modality_location.s3:
            download_kwargs = {
                "location": modality_location.s3,
                "download_dir": ModelCache.languagebind_cache_path,
            }
            if self.model_auth and self.model_auth.s3:
                download_kwargs["auth"] = self.model_auth.s3
            return download_pretrained_from_s3(**download_kwargs)
        elif modality_location.hf:
            download_kwargs = {
                "location": modality_location.hf,
                "download_dir": ModelCache.languagebind_cache_path,
            }
            if self.model_auth and self.model_auth.hf:
                download_kwargs["auth"] = self.model_auth.hf
            return download_model_from_hf(**download_kwargs)
        else:
            raise InternalError("Invalid modality location object provided.")

    def _convert_output(self, output: Tensor) -> List[ndarray]:
        if self.device == "cpu":
            return [single_ndarray for single_ndarray in output.numpy()]
        elif self.device.startswith("cuda"):
            return [single_ndarray for single_ndarray in output.cpu().numpy()]

    def normalize(self, outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def get_preprocessor(self):
        return self._preprocessor
