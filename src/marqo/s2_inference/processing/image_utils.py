import requests

import cv2
import numpy as np
import onnxruntime
from torchvision import transforms

from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.s2_inference.errors import ChunkerMethodProcessError

logger = get_logger(__name__)


def get_default_size() -> Tuple:
    """this sets the default image size used for inference for the chunker

    Returns:
        Tuple: _description_
    """
    return (240,240)

def _PIL_to_opencv(pil_image: ImageType) -> ndarray:
    """switches between PIL and cv2 -  assumes BGR for cv2

    Args:
        pil_image (ImageType): _description_

    Raises:
        TypeError: _description_

    Returns:
        ndarray: _description_
    """
    if isinstance(pil_image, ImageType):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    raise TypeError(f"expected a PIL image but received {type(pil_image)}")

def _keep_topk(boxes_xyxy: Union[ndarray, FloatTensor], k: int = 10) -> Union[ndarray, FloatTensor]:
    """keeps first k elements

    Args:
        boxes_xyxy (Union[ndarray, FloatTensor]): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        Union[ndarray, FloatTensor]: _description_
    """
    if k == 0:
        return []

    if len(boxes_xyxy) <= k:
        return boxes_xyxy

    return boxes_xyxy[:k]

def _get_onnx_provider(device: str) -> str:
    """determine where the model should run based on specified device
    """
    onnxproviders = onnxruntime.get_available_providers()
    logger.info(f"device:{device} and available providers {onnxproviders}")
    if device == 'cpu':
        fast_onnxprovider = 'CPUExecutionProvider'
    else:
        if 'CUDAExecutionProvider' not in onnxproviders:
            fast_onnxprovider = 'CPUExecutionProvider'
        else:
            fast_onnxprovider = 'CUDAExecutionProvider'

    logger.info(f"onnx_provider:{fast_onnxprovider}")
    return fast_onnxprovider

def load_rcnn_image(image_name: str, size: Tuple = (320,320)) -> Tuple[ImageType, FloatTensor, Tuple[int, int]]:
    """this is the loading and processing for the input

    Args:
        image_name (str): _description_
        size (Tuple, optional): _description_.

    Returns:
        Tuple[ImageType, FloatTensor, Tuple[int, int]]: _description_
    """
    
    if isinstance(image_name, ImageType):
        image = image_name 
    elif isinstance(image_name, str):
        image = load_image_from_path(image_name)
    else:
        raise TypeError(f"received {type(image_name)} but expected a string or PIL image")

    original_size = image.size
    image = image.convert('RGB').resize(size)
    image_pt = transforms.ToTensor()(image)

    return image, image_pt,original_size

def calc_area(bboxes: Union[List[List], FloatTensor, ndarray], size: Union[None, Tuple[int, int]] = None) -> List[float]:
    """calculates the fractional area of a rectangle given 4 numbers (2points)
    (x1, y1, x2, y2) and the original size

    Args:
        bboxes (Union[List[List], FloatTensor, ndarray]): _description_
        size (Tuple[int, int]): _description_

    Returns:
        List[Float]: _description_
    """

    if size is None:
        A = 1.0
    else:
        A = size[0]*size[1]*1.0
    areas = [(bb[2]-bb[0])*(bb[3]-bb[1])/A for bb in bboxes]

    return areas

def filter_boxes(bboxes: Union[FloatTensor, ndarray], max_aspect_ratio: int = 4, 
                    min_area: int = 40*40, min_k: int = 3) -> List[int]:
    """filters a list of bounding boxes given as the 4-tuple (x1, y1, x2, y2)
    by area and aspect ratio

    Args:
        bboxes (Union[FloatTensor, ndarray]): _description_
        max_aspect_ratio (int, optional): _description_. Defaults to 4.
        min_area (int, optional): _description_. Defaults to 40*40.
        min_k : always keep this many
    Returns:
        List[ind]: _description_
    """
    inds = []
    for ind,bb in enumerate(bboxes):
        w, h = (bb[2] - bb[0]), (bb[3] - bb[1])
        area = w*h
        aspect = max(w,h)/min(w,h)
        if area > min_area and aspect < max_aspect_ratio:
            inds.append(ind)
    
    return inds

def rescale_box(box: Union[List[float], ndarray, FloatTensor], from_size: Tuple, to_size: Tuple) -> Tuple:
    """rescales a bounding box between two different image sizes

    Args:
        box (Union[List[float], ndarray, FloatTensor]): _description_
        from_size (Tuple): _description_
        to_size (Tuple): _description_

    Returns:
        Tuple: _description_
    """
    Fy = to_size[1]/from_size[1]
    Fx = to_size[0]/from_size[0]

    x1, y1, x2, y2 = box

    x1_n = x1*Fx
    x2_n = x2*Fx

    y1_n = y1*Fy
    y2_n = y2*Fy

    return (x1_n, y1_n, x2_n, y2_n)

def generate_boxes(image_size: Tuple[int, int], hn: int, wn: int, overlap: bool = False) -> List[Tuple]:
    """does a simple bounding box generation based on the desired number in the 
    horizontal and vertical directions

    Args:
        image_size (Tuple[int, int]): _description_
        hn (int): _description_
        wn (int): _description_

    Returns:
        List[Tuple]: _description_
    """
    img_width, img_height = image_size

    height = img_height // hn

    width = img_width // wn

    bboxes = []
    for i in range(0,img_height, height):
        for j in range(0,img_width, width):
            p1 = j+width
            p2 = i+height
            box = (j, i, p1, p2)
            if p1 > img_width or p2 > img_height:
                continue
            bboxes.append(box)

            if overlap:
                p3 = p1 + width//2
                p4 = p2 + height//2
                box = (j + width//2, i + height//2, p3, p4)
                if p3 > img_width or p4 > img_height:
                    continue
                bboxes.append(box)


    return bboxes

def str2bool(string: str) -> bool:
    """converts a string into a bool

    Args:
        string (str): _description_

    Returns:
        bool: _description_
    """
    return string.lower() in ("true", "1", "t")

def replace_small_boxes(boxes: Union[List[List[float]], List[Tuple[float]], ndarray], 
                min_area: float = 40*40, new_size: Tuple = (100,100)) -> List[Tuple]:
    """replaces bounding boxes given as (x0,y0,x1,y1) and any below min_area
    are replaced with boxes of size new_size
    Args:
        boxes (Union[List[List[float]], List[Tuple[float]], ndarray]): _description_
        min_area (float, optional): _description_. Defaults to 40*40.
        new_size (Tuple, optional): _description_. Defaults to (100,100).

    Returns:
        List[Tuple]: _description_
    """
    new_boxes = []
    for box in boxes:
        area = (box[2]-box[0])*(box[3] - box[1])
        if area < min_area:
            xc = (box[2]-box[0])/2 + box[0]
            yc = (box[3]-box[1])/2 + box[1]
            box = (xc-new_size[0]/2, yc-new_size[1]/2, xc+new_size[0]/2, yc+new_size[1]/2)
        new_boxes.append(box)

    return new_boxes

def clip_boxes(boxes: Union[List, ndarray], xmin: int, ymin: int, xmax: int, ymax: int) -> List[Tuple]:
    """given a list or iterable 4-tuple of floats or ints, clip these to be within the limits

    Args:
        boxes (Union[List, ndarray]): _description_
        xmin (int): _description_
        ymin (int): _description_
        xmax (int): _description_
        ymax (int): _description_

    Returns:
        List[Tuple]: _description_
    """
    new_boxes = []
    for box in boxes:
        
        x1, y1, x2, y2 = box
        x1 = np.clip(x1, xmin, xmax) 
        x2 = np.clip(x2, xmin, xmax) 
        y1 = np.clip(y1, ymin, ymax) 
        y2 = np.clip(y2, ymin, ymax) 
        box = (x1, y1, x2, y2)
        new_boxes.append(box)
        
    return new_boxes
    
def patchify_image(image: ImageType, bboxes: Union[List[float], FloatTensor, ndarray]) -> List[ImageType]:
    """given a list of 4-tuple rectangles (x1, y1, x2, y2) return a list of 
    cropped images
    See PIL documentation for coord system

    Args:
        image (ImageType): _description_
        bboxes (Union[List[float], FloatTensor, ndarray]): _description_

    Returns:
        List[ImageType]: _description_
    """
    return [image.crop(bb) for bb in bboxes]

def _process_patch_method(method: str) -> Tuple[str, dict]:
    """processes a method 'url' into the base method and paramters

    Args:
        method (str): 'simple', 'simple?hn=3', 'overlap', 'overlap?hn=3&wn=4'

    Raises:
        ChunkerMethodProcessError: _description_

    Returns:
        Tuple[str, dict]: _description_
    """
    req = requests.utils.urlparse(method)
    query = req.query
    path = req.path

    params = dict()

    if len(query) == 0:
        return path, params

    try:
        params = dict(x.split('=') for x in query.split('&'))
    except: 
        raise ChunkerMethodProcessError(message=f"could not pass parameters for string {query} from full path {method}")

    return path, params