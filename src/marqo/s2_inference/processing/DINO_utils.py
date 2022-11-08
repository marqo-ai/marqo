import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import cv2

from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray, Any

logger = get_logger('DINO')


def _load_DINO_model(arch: str, device: str, patch_size=None) -> Tuple[Any, Any]:
    """ loads a vit dino model
       # arch -> ['vit_tiny', 'vit_small', 'vit_base']
    Args:
        arch (str): _description_
        device (str): _description_
        patch_size (_type_, optional): _description_. Defaults to None.

    Raises:
        KeyError: _description_
        KeyError: _description_
        RuntimeError: _description_

    Returns:
        Tuple[Any, Any]: _description_
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
        raise RuntimeError("wrong")

    model.eval()
    model.to(device)

    return model, _get_DINO_transform()

def _get_DINO_transform(image_size: Tuple = (224, 224)) -> Any:
    """gets the preprocessing transform for dino models

    Args:
        image_size (Tuple, optional): _description_. Defaults to (224, 224).

    Returns:
        Any: _description_
    """
    return pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def DINO_inference(model: Any, transform: Any, img: ImageType = None, 
                        patch_size: int = None, device: str = "cpu") -> FloatTensor:
    """runs inference for a model, transform and image

    Args:
        model (Any): _description_
        transform (Any): _description_
        img (ImageType, optional): _description_. Defaults to None.
        patch_size (int, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        FloatTensor: _description_
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

# def PIL_to_cv2(pil_image: ImageType) -> ndarray:
#     """switches between PIL and cv2 -  assumes BGR for cv2

#     Args:
#         pil_image (ImageType): _description_

#     Returns:
#         ndarray: _description_
#     """
#     open_cv_image = np.array(pil_image) 
#     # Convert RGB to BGR 
#     return open_cv_image[:, :, ::-1]

def _rescale_image(image: ndarray) -> ndarray:
    """rescales the image to be between 0-255
        assumes positive values as it does not correct a bias

    Args:
        image (ndarray): _description_

    Returns:
        ndarray: _description_
    """
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image


def attention_to_bboxs(image: ImageType) -> List[Tuple]:
    """turns attention maps into classless bounding boxes

    Args:
        image (ImageType): _description_

    Returns:
        List[Tuple]: _description_
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
