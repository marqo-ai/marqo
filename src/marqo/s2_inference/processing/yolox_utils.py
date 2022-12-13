import cv2
import numpy as np
import onnxruntime
import huggingface_hub

from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray, Callable
from marqo.s2_inference.processing.image_utils import _get_onnx_provider

logger = get_logger(__name__)

def get_default_yolox_model() -> Dict:
    """this is for the marqo-yolox onnx weights.
        this points to the huggingface model repo

    Returns:
        Dict: _description_
    """
    return {
        "repo_id":'Marqo/marqo-yolo-v2',
        "filename": 'yolox_s.onnx',
    }

def _download_yolox(repo_id : str, filename: str) -> str:
    """ downloads the model artefacts from hf

    Args:
        repo_name (str): 
        filename (str): 

    Returns:
        str: the local path to the weights
    """
    return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

def preprocess_yolox(img: ndarray, input_size: Tuple, swap: Tuple = (2, 0, 1)) -> Tuple[ndarray, float]:
    """prepares an image for yolox inference

    Args:
        img (ndarray): 
        input_size (Tuple): 
        swap (Tuple, optional): . Defaults to (2, 0, 1).

    Returns:
        Tuple[ndarray, float]: 
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    return padded_img, r

def load_yolox_onnx(model_name: str, device: str) -> Tuple[onnxruntime.InferenceSession, preprocess_yolox]:
    """ loads the yolox onnx model

    Args:
        model_name (str): _description_
        device (str): _description_

    Returns:
        Tuple[onnxruntime.InferenceSession, preprocess_yolox]: _description_
    """
    fast_onnxprovider = _get_onnx_provider(device)

    sess_options = onnxruntime.SessionOptions()

    session = onnxruntime.InferenceSession(model_name, sess_options=sess_options, providers=[fast_onnxprovider])
    session.disable_fallback()
    logger.info(f"using yolox onnx provider:{session.get_providers()} after requesting {fast_onnxprovider}")
    preprocess = preprocess_yolox

    return session, preprocess

def demo_postprocess(outputs: ndarray, img_size: Tuple[int, int], p6: bool = False) -> ndarray:
    """yolox post processing function
        formats the raw outputs into scores and bounding boxes
    Args:
        outputs (ndarray): the outputs from the yolox model inference
        img_size (Tuple[int, int]): the size of the input image
        p6 (bool, optional): model architecture parameter. marqo-yolo v1 and v2 should be False. 
                            check the model architecture for anything else. Defaults to False.

    Returns:
        ndarray: _description_
    """
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def _infer_yolox(session: onnxruntime.InferenceSession, preprocess: preprocess_yolox, 
                    opencv_image: ndarray, input_shape: Tuple[int, int]) -> Tuple[ndarray, float]:
    """inference for onnx yolox

    Args:
        session (onnxruntime.InferenceSession): the onnx session of the model
        preprocess (preprocess_yolox): the preprocess function for the input image
        opencv_image (ndarray):the opencv formatted input image 
        input_shape (Tuple[int, int]): the shape of the input image

    Returns:
        Tuple[ndarray, float]: _description_
    """
    img, ratio = preprocess(opencv_image, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)

    return output, ratio

def _process_yolox(output: ndarray, ratio: float, size: Tuple = (384, 384)) -> Tuple[ndarray, ndarray]:
    """takes the outputs and processes them 

    Args:
        output (ndarray): processes the outpus of the yolox inference
        ratio (float): the ratio between original and inferred sizes.
        size (Tuple, optional): . Defaults to (384, 384).

    Returns:
        Tuple[ndarray, ndarray]: _description_
    """
    predictions = demo_postprocess(output[0], size)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] 

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    return boxes_xyxy, scores

