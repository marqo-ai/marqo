from abc import abstractmethod

import numpy as np
from PIL import UnidentifiedImageError

from marqo.core.exceptions import InternalError
from marqo.core.inference.inference_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.s2_inference.types import *
from marqo.core.inference.inference_models.image_download import (_is_image, format_and_load_CLIP_images,
                                                                  format_and_load_CLIP_image)
from marqo.s2_inference.logger import get_logger
import torch
from marqo.core.inference.enums import Modality
from marqo.s2_inference.errors import UnsupportedModalityError

logger = get_logger(__name__)


class AbstractCLIPModel(AbstractEmbeddingModel):
    """Abstract base class for CLIP models.

    Attributes:
        device (str): The device to load the model on, typically 'cpu' or 'cuda'.
        model_properties (dict): A dictionary containing additional properties or configurations
            specific to the model. Defaults to an empty dictionary if not provided.
        model: The actual CLIP model instance, initialized to `None` and to be set by subclasses.
        tokenizer: The tokenizer associated with the model, initialized to `None` and to be set by subclasses.
        preprocess: The preprocessing pipeline for the model, initialized to `None` and to be set by subclasses.
    """

    def __init__(self, device: Optional[str] = None, model_properties: Optional[dict] = None,
                 model_auth: Optional[dict] = None):
        """Instantiate the abstract CLIP model.

        Args:
            device (str): The device to load the model on, typically 'cpu' or 'cuda'.
            model_properties (dict): A dictionary containing additional properties or configurations
                specific to the model. Defaults to an empty dictionary if not provided.
            model_auth (dict): The authentication information for the model. Defaults to `None` if not provided
        """

        super().__init__(model_properties, device, model_auth)

        self.model = None
        self.tokenizer = None
        self.preprocess = None

    @abstractmethod
    def encode_text(self, inputs: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def encode_image(self, inputs, normalize: bool = True, image_download_headers: dict = None) -> np.ndarray:
        pass

    def _validate_and_set_modality(self, modality: Optional[Modality] = None) -> Modality:
        if modality is None:
            return Modality.TEXT
        elif modality in [Modality.TEXT, Modality.IMAGE]:
            return modality
        else:
            raise UnidentifiedImageError(
                f"The model expected modality to be one of {Modality.TEXT} "
                f"or {Modality.IMAGE} but received {modality}."
        )

    def _validate_content_type(self, content: Any, modality: Modality) -> None:
        """Validate if the provided content type is valid for the specific model and if it matches the modality.

        Args:
            content (Any): The content to validate.
            modality (Modality): The modality of the content.

        Raises:
            ValueError: If the content type is not valid.
        """

        # TODO: Implement this method
        pass


    def _encode(self, content: Union[str, ImageType, List[str], List[ImageType], Tensor],
                modality: Modality, normalize: bool = True) -> np.ndarray:
        """Encode the given content.

        Args:
            content (): The content to encode.
            modality (Modality): The modality of the content.
            normalize (bool): Whether to normalize the output embeddings.

        Returns:
            np.ndarray: The encoded content.
        """
        if modality == Modality.TEXT:
            return self.encode_text(content, normalize)
        elif modality == Modality.IMAGE:
            return self.encode_image(content, normalize)
        else:
            raise InternalError(f"Unsupported modality: {modality}")

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

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