import os
from marqo.tensor_search.enums import ModelProperties, InferenceParams
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
import validators
import requests
import numpy as np
from transformers import ClapModel, ClapProcessor
# from msclap import CLAP as MSCLAP
import librosa
import soundfile as sf
import torch
import io
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.errors import (
    InvalidModelPropertiesError,
)
from marqo.s2_inference.configs import ModelCache
from marqo.errors import InternalError
from marqo.tensor_search.telemetry import RequestMetrics, RequestMetricsStore

logger = get_logger(__name__)

SR = 48000


def get_allowed_image_types():
    return set(".wav")


def _is_audio(inputs: Union[str, List[Union[str, ImageType, ndarray]]]) -> bool:
    # some logic to determine if something is an image or not
    # assume the batch is the same type
    # maybe we use something like this https://github.com/ahupp/python-magic

    _allowed = get_allowed_image_types()

    # we assume the batch is this way if a list
    # otherwise apply over each element
    if isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("received empty list, expected at least one element.")

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
            raise ValueError(
                f"local file [{thing}] extension {extension} does not match allowed file types of {_allowed}"
            )
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
        raise ValueError(
            f"expected type Image or str for inputs but received type {type(thing)}"
        )


def format_and_load_CLAP_audio(
    audio: Union[str, ndarray], download_headers: dict
) -> np.ndarray:
    # check for the input type
    if isinstance(audio, str):
        audio_data = load_audio_from_path(audio, download_headers)
    elif isinstance(audio, np.ndarray):
        audio_data = audio
    else:
        raise ValueError(
            f"input of type {type(audio)} did not match allowed types of str, np.ndarray"
        )

    return audio_data


def format_and_load_CLAP_audios(
    audios: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
    download_headers: Optional[Dict] = None,
) -> np.ndarray:
    if not isinstance(audios, list):
        raise TypeError(f"expected list but received {type(audios)}")

    results = []
    for audio in audios:
        results.append(format_and_load_CLAP_audio(audio, download_headers))

    return results


def load_audio_from_path(
    audio_path: str,
    download_headers: dict,
    timeout=15,
    metrics_obj: Optional[RequestMetrics] = None,
) -> ImageType:
    if os.path.isfile(audio_path):
        audio = librosa.load(audio_path, sr=SR)
    elif validators.url(audio_path):
        try:
            if metrics_obj is not None:
                metrics_obj.start(f"audio_download.{audio_path}")

            with requests.get(
                audio_path, stream=True, timeout=timeout, headers=download_headers
            ) as resp:
                if not resp.ok:
                    raise ValueError(
                        f"audio url `{audio_path}` returned {resp.status_code}. Reason: {resp.reason}"
                    )

                audio, _ = librosa.load(io.BytesIO(resp.content), sr=SR)

            if metrics_obj is not None:
                metrics_obj.stop(f"audio_download.{audio_path}")

        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            raise ValueError(
                f"audio url `{audio_path}` is unreachable, perhaps due to timeout. "
                f"Timeout threshold is set to {timeout} seconds."
                f"\nConnection error type: `{e.__class__.__name__}`"
            )
    else:
        raise ValueError(
            f"input str of `{audio_path}` is not a local file or a valid url"
        )

    return audio


class CLAP:
    def __init__(
        self,
        model_name: str,
        device: str = None,
        embedding_dim: int = None,
        truncate: bool = True,
        **kwargs,
    ):
        self.model_name = model_name

        if not device:
            raise InternalError("`device` is required for loading CLIP models!")

        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_dimension = embedding_dim
        self.truncate = truncate
        self.model_properties: dict = kwargs.get("model_properties", dict())
        self.sample_rate = self.model_properties.get("sample_rate", 48000)
        # model_auth gets passed through add_docs and search requests:
        self.model_auth = kwargs.get(InferenceParams.model_auth, None)

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def _convert_output(self, output):
        if self.device == "cpu":
            return output.numpy()
        elif self.device.startswith("cuda"):
            return output.cpu().numpy()

    def load(self) -> None:
        raise NotImplementedError

    def encode_text(
        self, sentence: Union[str, List[str]], normalize=True
    ) -> FloatTensor:
        raise NotImplementedError

    def encode_audio(
        self,
        audios: Union[str, np.ndarray, List[Union[str, ImageType]]],
        normalize=True,
        download_headers: Optional[Dict] = None,
    ) -> FloatTensor:
        raise NotImplementedError

    def encode(
        self,
        inputs: Union[str, ImageType, List[Union[str, ImageType]]],
        default: str = "text",
        normalize=True,
        **kwargs,
    ) -> FloatTensor:
        raise NotImplementedError


# class MSCLAP(CLAP):

#     """
#     conveniance class wrapper to make clip work easily for both text and image encoding
#     """

#     def __init__(
#         self,
#         model_name: str,
#         device: str = None,
#         embedding_dim: int = None,
#         truncate: bool = True,
#         **kwargs,
#     ) -> None:
#         super().__init__(model_name, device, embedding_dim, truncate, **kwargs)

#     def load(self) -> None:
#         model_location_presence = (
#             ModelProperties.model_location in self.model_properties
#         )

#         path = self.model_properties.get(
#             "localpath", None
#         ) or self.model_properties.get("url", None)

#         if path is None and not model_location_presence:
#             self.model = MSCLAP(version=self.model_name, use_cuda=self.device != "cpu")
#             self.processor = ClapProcessor.from_pretrained(self.model_name)
#         else:
#             raise InvalidModelPropertiesError("Custom CLAP not supported")

#     def encode_text(
#         self, sentence: Union[str, List[str]], normalize=True
#     ) -> FloatTensor:
#         pass

#     def encode_audio(
#         self,
#         audios: Union[str, np.ndarray, List[Union[str, ImageType]]],
#         normalize=True,
#         download_headers: Optional[Dict] = None,
#     ) -> FloatTensor:
#         pass

#     def encode(
#         self,
#         inputs: Union[str, ImageType, List[Union[str, ImageType]]],
#         default: str = "text",
#         normalize=True,
#         **kwargs,
#     ) -> FloatTensor:
#         pass


class LAION_CLAP(CLAP):

    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        embedding_dim: int = None,
        truncate: bool = True,
        **kwargs,
    ) -> None:
        # super().__init__(model_name, device, embedding_dim, truncate, **kwargs)
        super().__init__(model_name, device, embedding_dim, truncate, **kwargs)

    def load(self) -> None:
        model_location_presence = (
            ModelProperties.model_location in self.model_properties
        )

        path = self.model_properties.get(
            "localpath", None
        ) or self.model_properties.get("url", None)

        if path is None and not model_location_presence:
            self.model = (
                ClapModel.from_pretrained(self.model_name).eval().to(self.device)
            )
            self.processor = ClapProcessor.from_pretrained(self.model_name)
        else:
            raise InvalidModelPropertiesError("Custom CLAP not supported")

    def encode_text(
        self, sentence: Union[str, List[str]], normalize=True
    ) -> FloatTensor:
        with torch.no_grad():
            inputs = self.processor(text=sentence, return_tensors="pt")
            outputs = self.model.get_text_features(**inputs)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_audio(
        self,
        audios: Union[str, np.ndarray, List[Union[str, ImageType]]],
        normalize=True,
        download_headers: Optional[Dict] = None,
    ) -> FloatTensor:
        if self.model is None:
            self.load()
        if download_headers is None:
            download_headers = dict()

        # default to batch encoding
        if isinstance(audios, list):
            audio_input = format_and_load_CLAP_audios(audios, download_headers)
        else:
            audio_input = [format_and_load_CLAP_audios(audios, download_headers)]

        with torch.no_grad():
            inputs = self.processor(audios=audio_input, return_tensors="pt")
            outputs = self.model.get_audio_features(**inputs)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode(
        self,
        inputs: Union[str, ImageType, List[Union[str, ImageType]]],
        default: str = "text",
        normalize=True,
        **kwargs,
    ) -> FloatTensor:
        infer = kwargs.pop("infer", True)

        if infer and _is_audio(inputs):
            is_image = True
        else:
            is_image = False
            if default == "text":
                is_image = False
            elif default == "audio":
                is_image = True
            else:
                raise ValueError(
                    f"expected default='audio' or default='text' but received {default}"
                )

        if is_image:
            logger.debug("image")
            download_headers = kwargs.get("download_headers", dict())
            return self.encode_audio(
                inputs, normalize=normalize, download_headers=download_headers
            )
        else:
            logger.debug("text")
            return self.encode_text(inputs, normalize=normalize)
