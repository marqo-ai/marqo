from abc import abstractmethod

from PIL import UnidentifiedImageError

from marqo.core.inference.models.abstract_model import AbstractModel
from marqo.s2_inference.types import *
from marqo.core.inference.models.utils import _is_image, format_and_load_CLIP_images, format_and_load_CLIP_image
from marqo.s2_inference.logger import get_logger
import torch

logger = get_logger(__name__)


class AbstractCLIPModel(AbstractModel):
    """Abstract base class for CLIP models.

    Attributes:
        model_tag (str): The tag of the model. It is used to identify the model in the model registry.
        device (str): The device to load the model on, typically 'cpu' or 'cuda'.
        embedding_dim (int, optional): The dimensionality of the model's embeddings. If not provided,
            it should be inferred from the model.
        truncate (bool): Indicates whether the text should be truncated to a smaller size in the tokenizer.
        model_properties (dict): A dictionary containing additional properties or configurations
            specific to the model. Defaults to an empty dictionary if not provided.
        model: The actual CLIP model instance, initialized to `None` and to be set by subclasses.
        tokenizer: The tokenizer associated with the model, initialized to `None` and to be set by subclasses.
        preprocess: The preprocessing pipeline for the model, initialized to `None` and to be set by subclasses.
    """

    def __init__(self, model_type: str,
                 device: Optional[str] = None,
                 embedding_dim: Optional[int] = None,
                 truncate: bool = True,
                 model_properties: Optional[dict] = None,
                 **kwargs):
        """Instantiate the abstract CLIP model.

        Args:
            model_type (str): The type of the model.
            device (str): The device to load the model on.
            embedding_dim (int): The embedding dimension of the model.
            truncate (bool): Whether to truncate the model.
            model_properties (dict): The properties of the model.
        """
        self.model_tag = model_type

        if not device:
            raise ValueError("`device` is required for loading CLIP models!")
        self.device = device

        self.embedding_dim = embedding_dim
        self.truncate = truncate

        self.model_properties = model_properties
        if self.model_properties is None:
            self.model_properties = dict()

        self.model = None
        self.tokenizer = None
        self.preprocess = None

    @abstractmethod
    def load_tokenizer(self):
        """Load tokenizer."""
        pass

    @abstractmethod
    def encode_text(self, inputs: Union[str, List[str]], normalize: bool = True) -> FloatTensor:
        pass

    @abstractmethod
    def encode_image(self, inputs, normalize: bool = True, image_download_headers: dict = None) -> FloatTensor:
        pass

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]],
               default: str = 'text', normalize=True, **kwargs) -> FloatTensor:
        infer = kwargs.pop('infer', True)

        if infer and _is_image(inputs):
            is_image = True
        else:
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