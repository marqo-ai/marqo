import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
import cv2

from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray, Any, Literal
from marqo.s2_inference.errors import ModelLoadError

logger = get_logger(__name__)


def _load_DINO_model(arch: Literal['vit_small', 'vit_base'], device: str, patch_size: int = None,
                    image_size: Tuple = (224, 224)) -> Tuple[Any, Any]:
    """ loads a vit dino model
       # arch -> ['vit_tiny', 'vit_small', 'vit_base']
    Args:
        arch (str): the model architecture
        device (str): cpu or cuda
        patch_size (_type_, optional): the number of patches for the model. note this needs to exactly match
                                        the number the model was trained with.

    Raises:
        KeyError: _description_
        KeyError: _description_
        RuntimeError: _description_

    Returns:
        Tuple[Any, Any]: the model and transform for the inputs
    """
  
    allowed_archs = ('vit_small', 'vit_base')
    allowed_patches = (8, 16)
    if arch not in allowed_archs:
        raise KeyError(f"{arch} not found in {allowed_archs}")

    if patch_size not in allowed_patches:
        raise KeyError(f"{patch_size} not found in {allowed_patches}")

    if arch == "vit_small":
        # TODO add onnx support
        if patch_size == 8:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

        if patch_size == 16:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    elif arch == "vit_base":
        if patch_size == 8:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        if patch_size == 16:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    else:
        raise ModelLoadError(f"unknown model of arch:{arch} and patch_size:{patch_size}")

    model.eval()
    model.to(device)

    return model, _get_DINO_transform(image_size=image_size)

def _get_DINO_transform(image_size: Tuple = (224, 224)) -> Any:
    """gets the preprocessing transform for dino models

    Args:
        image_size (Tuple, optional): The final size of the image before going to the model

    Returns:
        Any: _description_
    """
    
    # do not modify these unless you know exactly what you are doing!
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    return pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

def DINO_inference(model: Any, transform: Any, img: ImageType = None, 
                        patch_size: int = None, device: str = "cuda") -> FloatTensor:
    """runs inference for a model, transform and image

    Args:
        model (Any): ('vit_small', 'vit_base')
        transform (Any): _get_DINO_transform
        img (ImageType, optional): the image to infer on. Defaults to None.
        patch_size (int, optional): the patch size the model architecture uses. Defaults to None.
        device (str, optional): device for the model to run on. Defaults to "cuda".

    Returns:
        FloatTensor: returns N x w x h tensor
    """
    
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    with torch.no_grad():
        attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

def _rescale_image(image: Union[ndarray, ImageType]) -> ndarray:
    """rescales the image to be between 0-255
        assumes positive values as it does not correct a bias

    Args:
        image (ndarray): the input image to rescale

    Returns:
        ndarray: the rescaled image
    """

    if isinstance(image, ImageType):
        image = np.array(image.convert('RGB'))

    image = image*1.0
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image

def attention_to_bboxs(image: ndarray) -> List[Tuple]:
    """turns attention maps into classless bounding boxes
    expects a single dim attention e.g, attentions.size = (x, y)
    Args:
        image (ndarray): the greyscale image to use to generate bounding boxes for

    Returns:
        List[Tuple]: List of tuples for the bounding boxes
    """
    
    image = _rescale_image(image)
    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bboxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        x1 = x
        x2 = x + w
        y1 = y        
        y2 = y + h
        box = (x1, y1, x2, y2)
        bboxes.append(box)

    return bboxes
