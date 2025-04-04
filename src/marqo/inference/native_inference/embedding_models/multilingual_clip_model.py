from typing import List

import open_clip
import torch
import transformers
from PIL.Image import Image
from multilingual_clip import pt_multilingual_clip
from numpy import ndarray
from pydantic import ValidationError

from marqo.exceptions import InternalError
from marqo.inference.native_inference.embedding_models.abstract_clip_model import AbstractCLIPModel
from marqo.inference.native_inference.embedding_models.abstract_clip_model import AbstractCLIPPreprocessor
from marqo.inference.native_inference.embedding_models.multilingual_clip_model_properties import \
    MultilingualCLIPModelProperties
from marqo.logging import get_logger
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth

logger = get_logger(__name__)


class MultiLingualCLIPTokenizerWrapper:
    """
    A wrapper class for the tokenizer used in the multilingual CLIP model.

    The original implementation of the tokenizer can not move the tensors to the device. So we need to
    implement a wrapper class to move the tensors to the device.
    """

    def __init__(self, tokenizer, device: str):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, inputs: list[str]):
        return self.tokenize(inputs)

    def tokenize(self, inputs: list[str]):
        return self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.device)


class MultiLingualCLIPPreprocessor(AbstractCLIPPreprocessor):
    def __init__ (self, tokenizer, image_preprocessor, device: str):
        super().__init__(tokenizer=tokenizer, image_preprocessor=image_preprocessor)
        self.device = device

    def _tokenize_text(self, inputs: list[str]) -> List[str]:
        return inputs

    def _preprocess_image(self, inputs: list[Image]) -> List[Tensor]:
        return [self.image_preprocessor(image).unsqueeze(0).to(self.device) for image in inputs]


class MultiLingualCLIPModel(AbstractCLIPModel):
    """
    A class representing a multilingual CLIP model.
    This class inherits from the AbstractCLIPModel and implements the required methods.
    """
    def __init__(self, device: str, model_properties: dict, model_auth: Optional[ModelAuth] = None):
        super().__init__(device=device, model_properties=model_properties, model_auth=model_auth)

        self.model_properties = self._build_model_properties(model_properties)
        self._tokenizer = None
        self._image_preprocessor = None
        self._textual_model = None
        self._visual_model = None
        self._preprocessor = None

    def _build_model_properties(self, model_properties: dict) -> MultilingualCLIPModelProperties:
        """Convert the user input model_properties to MultilingualCLIPModelProperties."""
        try:
            return MultilingualCLIPModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {model_properties}. Original error: {e}") \
                from e

    def _load_necessary_components(self):
        self._visual_model, _, self._image_preprocessor = open_clip.create_model_and_transforms(
            model_name=self.model_properties.visual_model.split("/")[1],
            pretrained=self.model_properties.visual_model.split("/")[2],
            device=self.device
        )

        self._textual_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            self.model_properties.textual_model, self.device
        )
        self._tokenizer = MultiLingualCLIPTokenizerWrapper(
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_properties.textual_model),
            device= self.device
        )

        self._preprocessor = MultiLingualCLIPPreprocessor(
            tokenizer=self.tokenizer,
            image_preprocessor=self.image_preprocessor,
            device=self.device
        )

        self._textual_model.eval()
        self._visual_model.eval()

    def _check_loaded_components(self):
        if any(
                [
                    self._visual_model is None,
                    self._textual_model is None,
                    self._image_preprocessor is None,
                    self._tokenizer is None,
                ]
        ):
            raise InternalError("Model components are not loaded correctly.")

    def encode_text(self, inputs: List[str], normalize: bool = True) -> List[ndarray]:
        """
        Encode text inputs into embeddings.

        Args:
            inputs (List): A list of text inputs to be encoded.
            normalize (bool): Whether to normalize the embeddings. Defaults to True.

        Returns:
            List[ndarray]: A list of encoded text embeddings.
        """
        with torch.no_grad():
            outputs = self._textual_model.forward(inputs, self._tokenizer)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, inputs: List[Tensor], normalize: bool = True) -> List[ndarray]:
        """
        Args:
            inputs:
            normalize:

        Returns:

        """
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = self._visual_model.encode_image(inputs).to(torch.float32)
            else:
                outputs = self._visual_model.encode_image(inputs).to(torch.float32)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def get_preprocessor(self):
        return self._preprocessor

