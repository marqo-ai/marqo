import torch
import torchvision

from marqo.s2_inference.types import *

from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

def load_pytorch(model_name: str, device: str):
    """loads the pytorch based object detector

    Args:
        model_name (str): _description_
        device (str): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    if model_name in ('frcnn', 'faster_rcnn'):
        model, preprocess = load_pytorch_rcnn()
        model = model.to(device)
        model.eval()
    else:
        raise RuntimeError("incorrect model specified")

    return model, preprocess

def load_pretrained_mobilenet():
    """"
    loads marqo trained model
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(device='cpu', num_classes=1204,
    box_score_thresh=0.0001, box_nms_thresh=0.01, 
                            rpn_pre_nms_top_n_test=200, 
                            box_detections_per_img=100,
                            rpn_post_nms_top_n_test=100, min_size=320)

    checkpoint_file = 'awesome_mode.pth'
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    transform = weights.transforms()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, transform

def load_pretrained_mobilenet320():
    """"
    loads marqo trained model
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(device='cpu', num_classes=1204,
    box_score_thresh=0.0001, box_nms_thresh=0.01, 
                            rpn_pre_nms_top_n_test=200, 
                            box_detections_per_img=100,
                            rpn_post_nms_top_n_test=100, min_size=320)

    checkpoint_file = 'model_17.pth'
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    transform = weights.transforms()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, transform

def load_pytorch_fcos():
    """this loads the pytorch fcos model

    Returns:
        _type_: _description_
    """
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT

    model = torchvision.models.detection.fcos_resnet50_fpn(weights=weights,
                        **get_default_rcnn_params())
    model.eval()

    preprocess = weights.transforms()
    
    return model, preprocess

def load_pytorch_rcnn():
    """loads the pytorch faster rcnn model

    Returns:
        _type_: _description_
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, 
                         **get_default_rcnn_params()
                        )
    # required for detector models otherwise they require targets for inference
    model.eval()

    # preprocessor lives in the 'weights'
    preprocess = weights.transforms()
    
    return model, preprocess

def get_default_rcnn_params() -> Dict:
    """sets the default params for a faster-rcnn in pytorch

    Returns:
        Dict: _description_
    """
    return {'box_score_thresh':0.0001, 
            'box_nms_thresh':0.01, 
            'rpn_pre_nms_top_n_test':200, 
            'box_detections_per_img':100,
            'rpn_post_nms_top_n_test':100, 
            'min_size':320,
    }

# TODO add onnx support https://pytorch.org/vision/0.12/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

