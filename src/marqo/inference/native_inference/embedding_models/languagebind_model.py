import os
import tempfile
from contextlib import contextmanager

import torch
from pydantic import ValidationError

from marqo.base_model import MarqoBaseModel
from marqo.core.exceptions import InternalError
from marqo.exceptions import InternalError
from marqo.inference.media_download_and_preprocess.image_download import format_and_load_CLIP_images, \
    download_media_from_url
from marqo.inference.model_download.model_download import (download_model_from_hf, download_pretrained_from_url,
                                                           download_pretrained_from_s3, extract_zip_file)
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.native_inference.embedding_models.languagebind_model_properties import LanguagebindModelProperties, \
    ModalityLocation
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError, MediaMismatchError
from marqo.s2_inference.languagebind import LanguageBindImageTokenizer, LanguageBind, transform_dict, to_device
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor


class LanguagebindPreprocessor(AbstractPreprocessor):

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def preprocess(self, inputs, modality: Modality):
        return self.preprocessor(inputs)



class CLIPType(MarqoBaseModel):
    """A wrapper class that is used to store the model location for each modality.
    The key is the modality and the value is the location of the model.
    A location can either be a HuggingFace repo ID or a directory containing the model files.
    """
    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None


class LanguagebindModel(AbstractEmbeddingModel):
    DEFAULT_TOKENIZER_REPO = 'lb203/LanguageBind_Image'

    MODEL_NAME_CLIP_TYPE_MAPPING = {
        "LanguageBind/Video_V1.5_FT_Audio_FT_Image": CLIPType(
            video='LanguageBind/LanguageBind_Video_V1.5_FT',
            audio='LanguageBind/LanguageBind_Audio_FT',
            image='LanguageBind/LanguageBind_Image'
        ),
        "LanguageBind/Video_V1.5_FT_Audio_FT": CLIPType(
            video='LanguageBind/LanguageBind_Video_V1.5_FT',
            audio='LanguageBind/LanguageBind_Audio_FT'
        ),
        "LanguageBind/Video_V1.5_FT_Image": CLIPType(video='LanguageBind/LanguageBind_Video_V1.5_FT',
                                                     image='LanguageBind/LanguageBind_Image'),
        "LanguageBind/Audio_FT_Image": CLIPType(audio='LanguageBind/LanguageBind_Audio_FT',
                                                image='LanguageBind/LanguageBind_Image'),
        "LanguageBind/Audio_FT": CLIPType(audio='LanguageBind/LanguageBind_Audio_FT'),
        "LanguageBind/Video_V1.5_FT": CLIPType(video='LanguageBind/LanguageBind_Video_V1.5_FT'),
    }

    def __init__(
            self,
            device: Optional[str] = None,
            model_properties: Optional[Dict] = None,
            model_auth: Optional[ModelAuth] = None,
    ) -> None:

        super().__init__(model_properties=model_properties, device=device, model_auth=model_auth)

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
            raise InvalidModelPropertiesError(
                f"Invalid model properties: {e}"
            )

    def _load_necessary_components(self):
        self._clip_type = self._generate_clip_type()
        self._load_model()
        self._load_tokenizer()
        self._load_preprocessor()

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
            token = self.model_auth.hf.token if (self.model_auth and self.model_auth.hf) else None
            self._model = LanguageBind(
                self._clip_type.dict(exclude_none=True),
                cache_dir=ModelCache.languagebind_cache_path,
                token = token
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
            clip_type_dict = {
                "image": None,
                "video": None,
                "audio": None
            }
            for modality in self.model_properties.supportedModalities:
                if modality == Modality.TEXT or modality == "text":
                    continue
                model_location: ModalityLocation = getattr(self.model_properties.modelLocation, modality)
                if model_location is None:
                    continue
                elif model_location.hf and not model_location.hf.filename:
                    clip_type_dict[modality.value] = model_location.hf.repo_id
                elif (model_location.hf and model_location.hf.filename) or model_location.s3 or model_location.url:
                    downloaded_zip_file = self._download_languagebind_model(model_location)
                    clip_type_dict[modality.value] = extract_zip_file(downloaded_zip_file)
                else:
                    raise InternalError(
                        f"Invalid model location {model_location} provided for modality {modality}"
                    )
            clip_type = CLIPType(**clip_type_dict)
        else:
            raise InvalidModelPropertiesError("Invalid model properties provided. Either 'name' or "
                                              "'modelLocation' must be provided.")
        return clip_type

    def _load_tokenizer(self):
        if self.model_properties.modelLocation:
            self._load_custom_tokenizer()
        else:
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                self.DEFAULT_TOKENIZER_REPO, cache_dir=ModelCache.languagebind_cache_path
            )

    def _load_custom_tokenizer(self):
        """Custom tokenizer loading. The tokenizer can be loaded in two ways:

        1. A huggingface repo, e.g., 'lb203/LanguageBind_Image'
        2. A directory containing the tokenizer files
        """
        tokenizer_location: ModalityLocation = self.model_properties.modelLocation.tokenizer
        if tokenizer_location is None:
            # Use the default tokenizer repo
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                self.DEFAULT_TOKENIZER_REPO, cache_dir=ModelCache.languagebind_cache_path,
            )
        elif tokenizer_location.hf and (not tokenizer_location.hf.filename):
            # Loading from a HuggingFace repo
            token = self.model_auth.hf.token if (self.model_auth and self.model_auth.hf) else None
            self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                tokenizer_location.hf.repo_id, cache_dir=ModelCache.languagebind_cache_path, token=token
            )
        elif ((tokenizer_location.hf and tokenizer_location.hf.filename) or tokenizer_location.s3 or
              tokenizer_location.url):
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
                    f" Original error message = {e}") from e
        else:
            raise InternalError(f"Invalid tokenizer location provided for tokenizer: "
                                              f"{tokenizer_location}")

    def _load_preprocessor(self):
        """Load the preprocessors for each modality.

        It is a dictionary where the key is the modality and the value is the preprocessor function.
        """
        self._preprocessors = {c: transform_dict[c](self._model.modality_config[c]) for c in
                               self._clip_type.dict(exclude_none=True).keys()}

    def encode(self, content, modality, media_download_headers: Optional[Dict] = None, normalize=True, **kwargs):
        if media_download_headers is None:
            media_download_headers = dict()

        if modality not in self.model_properties.supportedModalities:
            raise MediaMismatchError(f"The provided modality {modality} is not supported by the model. This model "
                                     f"supports the following modalities: {self.model_properties.supportedModalities}")

        if modality == Modality.TEXT:
            return self._encode_text(content, normalize)
        elif modality == Modality.IMAGE:
            return self._encode_image(content, normalize, media_download_headers)
        elif modality == Modality.VIDEO:
            return self._encode_video(content, normalize, media_download_headers)
        elif modality == Modality.AUDIO:
            return self._encode_audio(content, normalize, media_download_headers)

    def _encode_text(self, text: Union[str, list[str]], normalize=True):
        formated_input = dict()
        processed_text = to_device(
            self._tokenizer(text, max_length=77, padding='max_length',
                            truncation=True, return_tensors='pt'), self.device
        )
        formated_input['language'] = processed_text

        with torch.no_grad():
            outputs = self._model(formated_input)["language"]

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            if outputs.shape != _shape_before:
                raise InternalError("Normalization changed the shape of the output tensor.")
        return self._convert_output(outputs)

    def _preprocess_image(self, content: Union[str, list[str], dict[str, Tensor]],
                          media_download_headers: dict) -> dict:
        """
        Preprocess the image content to be encoded. It can be represented in the following ways:
            - A str: URL of the image from a search query.
            - A list of str: URLs of images for batch inference.
            - A dictionary of str and Tensor: Key should be "pixel_values", and the value should be the preprocessed image tensors.

        Args:
            content: The image content to be preprocessed.
            media_download_headers: Headers for downloading private image content.

        Returns:
            A dictionary containing the preprocessed image tensors in the format:
            > {"pixel_values": torch.Tensor}, where the tensor is of shape [N, 3, 224, 224] and N is the number of images.
        """

        def process_image_dict(image):
            if "pixel_values" not in image:
                raise InternalError(f"Invalid image input format: {image}")
            return image["pixel_values"]

        # Normalize content to a list
        content = [content] if isinstance(content, str) else content
        if not isinstance(content, list):
            raise InternalError(f"Invalid image input format: {content}")

        if len(content) == 0:
            raise InternalError(f"Invalid image input format: {content}. The input list is empty.")

        # Process content based on its type
        if isinstance(content[0], str):
            images = format_and_load_CLIP_images(content, media_download_headers=media_download_headers)
            processed_images = self._preprocessors["image"](images, return_tensors='pt')["pixel_values"]
        elif isinstance(content[0], dict):
            processed_images = torch.stack([process_image_dict(image).squeeze(0) for image in content], dim=0)
        else:
            raise InternalError(f"Invalid image input format: {content}")

        return {"pixel_values": processed_images}

    def _encode_image(self, content: Union[str, list[str], dict[str, Tensor]], normalize: bool,
                      media_download_headers: dict):
        """
        Args:
            content: The image content to be encoded. It can be represented in the following ways:
                - A str: URL of the image from a search query.
                - A list of str: URLs of images for batch inference.
                - A dictionary of str and Tensor: Key should be "pixel_values", and the value should be the preprocessed image tensors.
            normalize: Whether to normalize the output.
            media_download_headers: Headers for downloading private image content.
        """
        processed_images = self._preprocess_image(content, media_download_headers)
        formated_input = {"image": to_device(processed_images, self.device)}

        # Perform inference
        with torch.no_grad():
            outputs = self._model(formated_input)["image"]

        # Normalize outputs if required
        if normalize:
            outputs /= self.normalize(outputs)
            if outputs.shape != outputs.shape:  # Check if normalization changed the shape
                raise InternalError("Normalization changed the shape of the output tensor.")

        return self._convert_output(outputs)

    def _preprocess_video(self, content, media_download_headers) -> dict:
        """
        Preprocess the video content to be encoded. It can be represented in the following ways:
            - A str: URL of the video from a search query
            - A list of str: URLs of videos for batch inference
            - A dictionary of str and Tensor: Key should be "pixel_values", and the value should be the
              preprocessed video tensors from the add_document method.

        Args:
            content: The video content to be encoded.
            media_download_headers: Headers for downloading private video content.

        Returns:
            A dictionary containing the preprocessed video tensors in the format:
            > {"pixel_values": torch.Tensor}, where the tensor is of shape [N, 3, 8, 224, 224] and N is the number of videos.
        """

        def process_video_url(video_url):
            with self._temp_file(".mp4") as temp_filename:
                self._download_content(video_url, temp_filename, media_download_headers, Modality.VIDEO)
                return self._preprocessors["video"]([temp_filename], return_tensors='pt')["pixel_values"][0]

        def process_video_dict(video):
            if "pixel_values" not in video:
                raise InternalError(f"Invalid video input format: {video}")
            return video["pixel_values"][0]

        # Normalize content to a list
        content = [content] if isinstance(content, str) else content
        if not isinstance(content, list):
            raise InternalError(f"Invalid video input format: {content}")

        if len(content) == 0:
            raise InternalError(f"Invalid image input format: {content}. The input list is empty.")

        # Process content based on its type
        if isinstance(content[0], str):
            processed_videos = [process_video_url(video_url) for video_url in content]
        elif isinstance(content[0], dict):
            processed_videos = [process_video_dict(video) for video in content]
        else:
            raise InternalError(f"Invalid video input format: {content}")

        # Stack processed video tensors
        processed_videos = {"pixel_values": torch.stack(processed_videos, dim=0)}
        return processed_videos

    def _encode_video(self, content, normalize, media_download_headers):
        """
        Args:
            content: The video content to be encoded. It can be represented in the following ways:
                - A str: URL of the video from a search query
                - A list of str: URLs of videos for batch inference
                - A dictionary of str and Tensor: Key should be "pixel_values", and the value should be
                  the preprocessed video tensors from the add_document method.
        """
        processed_videos = self._preprocess_video(content, media_download_headers)
        formated_input = {"video": to_device(processed_videos, device=self.device)}

        # Perform inference
        with torch.no_grad():
            outputs = self._model(formated_input)["video"]

        # Normalize outputs if required
        if normalize:
            outputs /= self.normalize(outputs)
            if outputs.shape != outputs.shape:  # Check if normalization changed the shape
                raise InternalError("Normalization changed the shape of the output tensor.")

        return self._convert_output(outputs)

    def _preprocess_audio(self, content, media_download_headers) -> dict:
        """
        Preprocess the audio content to be encoded. It can be represented in the following ways:
            - A str, which is the URL of the audio from search query
            - A list of str, which are the URLs of the audios from search query (normally weighted queries that do
                inference in batch)
            - A dictionary of str and Tensor, the key should be "pixel_values" and the value should be the preprocessed
            audio tensors. They are preprocessed audio tensors from add_document method.

        Args:
            content: the audio content to be encoded
            media_download_headers: the headers to be used for downloading the private audio content
        Returns:
            A dictionary containing the preprocessed audio tensors in the format
            > {"pixel_values": torch.Tensor}, where the tensor is of shape [N, 3, 112, 1036] and N is the number of audios
        """

        def process_audio_url(audio_url):
            with self._temp_file(".mp4") as temp_filename:
                self._download_content(audio_url, temp_filename, media_download_headers, Modality.AUDIO)
                return self._preprocessors["audio"]([temp_filename], return_tensors='pt')["pixel_values"][0]

        def process_audio_dict(audio):
            if "pixel_values" not in audio:
                raise InternalError(f"Invalid audio input format: {audio}")
            return audio["pixel_values"][0]

        # Normalize content to a list
        content = [content] if isinstance(content, str) else content
        if not isinstance(content, list):
            raise InternalError(f"Invalid audio input format: {content}")

        if len(content) == 0:
            raise InternalError(f"Invalid image input format: {content}. The input list is empty.")

        # Process content based on its type
        if isinstance(content[0], str):
            processed_audios = [process_audio_url(audio_url) for audio_url in content]
        elif isinstance(content[0], dict):
            processed_audios = [process_audio_dict(audio) for audio in content]
        else:
            raise InternalError(f"Invalid audio input format: {content}")

        # Stack processed audio tensors
        processed_audios = {"pixel_values": torch.stack(processed_audios, dim=0)}
        return processed_audios

    def _encode_audio(self, content, normalize, media_download_headers):
        """
        Args:
            content: The audio content to be encoded. It can be represented in the following ways:
                - A str: URL of the audio from a search query
                - A list of str: URLs of audios for batch inference
                - A list of dict: Each dict contains the key "pixel_values" with preprocessed audios
        """
        processed_audios = self._preprocess_audio(content, media_download_headers)
        formated_input = {"audio": to_device(processed_audios, device=self.device)}

        # Perform inference
        with torch.no_grad():
            outputs = self._model(formated_input)["audio"]

        # Normalize outputs if required
        if normalize:
            outputs /= self.normalize(outputs)
            if outputs.shape != outputs.shape:  # Check if normalization changed the shape
                raise InternalError("Normalization changed the shape of the output tensor.")

        return self._convert_output(outputs)

    def _download_content(self, url, filename, media_download_headers: Optional[Dict] = None,
                          modality: Optional[str] = None):
        # 3 seconds for images, 20 seconds for audio and video
        timeout_ms = 3000 if filename.endswith(('.png', '.jpg', '.jpeg')) else 20000

        buffer = download_media_from_url(url, media_download_headers, timeout_ms, modality)

        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())

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
            return download_pretrained_from_url(modality_location.url, cache_dir=ModelCache.languagebind_cache_path)
        elif modality_location.s3:
            download_kwargs = {'location': modality_location.s3, 'download_dir': ModelCache.languagebind_cache_path}
            if self.model_auth and self.model_auth.s3:
                download_kwargs['auth'] = self.model_auth.s3
            return download_pretrained_from_s3(**download_kwargs)
        elif modality_location.hf:
            download_kwargs = {'location': modality_location.hf, 'download_dir': ModelCache.languagebind_cache_path}
            if self.model_auth and self.model_auth.hf:
                download_kwargs['auth'] = self.model_auth.hf
            return download_model_from_hf(**download_kwargs)
        else:
            raise InternalError("Invalid modality location object provided.")

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    def normalize(self, outputs):
        return outputs.norm(dim=-1, keepdim=True)

    @contextmanager
    def _temp_file(self, suffix):
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def get_preprocessor(self):
        raise NotImplementedError