import copy
import requests
from functools import partial

import PIL
import numpy as np
import torch
import torchvision

from marqo.s2_inference.s2_inference import available_models
from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray
from marqo.s2_inference.clip_utils import format_and_load_CLIP_image, _load_image_from_path
from marqo.s2_inference.errors import ChunkerError, ChunkerMethodProcessError

from marqo.s2_inference.processing.DINO_utils import _load_DINO_model,attention_to_bboxs,DINO_inference
from marqo.s2_inference.processing.pytorch_utils import load_pytorch
from marqo.s2_inference.processing.yolox_utils import (
   _process_yolox,
    _infer_yolox, 
    load_yolox_onnx,
)

from marqo.s2_inference.processing.image_utils import (
    load_rcnn_image, 
    replace_small_boxes,
    _keep_topk,
    rescale_box,
    clip_boxes,
    _PIL_to_opencv, 
    str2bool,
    get_default_size,
    _process_patch_method,
    patchify_image,
    filter_boxes,
    calc_area,
    generate_boxes
)

logger = get_logger('image_chunks')


    
def chunk_image(image: Union[str, ImageType], device: str, 
                        method: str = 'simple', size=get_default_size()) -> Tuple[List[ImageType], ndarray]:
    """_summary_
    wrapper function to do the chunking and return the patches and their bounding boxes
    in the original coordinates system

    Args:
        image (Union[str, ImageType]): _description_
        device (str): _description_
        method (str, optional): _description_. Defaults to 'simple'.
        size (_type_, optional): _description_. Defaults to get_default_size().

    Raises:
        TypeError: _description_
        ValueError: _description_
        ChunkerError: Raises ChunkerError, if the chunker can't work for some reason

    Returns:
        Tuple[List[ImageType], ndarray]: _description_
    """

    HN = 3
    WN = 3

    if method in [None, 'none', '', "None", ' ']:
        if isinstance(image, str):
            return [image],[image]      
        elif isinstance(image, ImageType):
            return [image], [(0, 0, image.size[0], image.size[1])]
        else:
            raise TypeError(f'only pointers to an image or a PIL image are allowed. received {type(image)}')
    
    # get the paramters from the method 'url'
    method, params = _process_patch_method(method)
    logger.info(f"found method={method} and params={params}")

    # format the paramters to pass through
    hn = int(params.get('hn', HN))
    wn = int(params.get('wn', WN))
    nms = str2bool(params.get('nms', 'True'))
    filter_bb = str2bool(params.get('filter_bb', 'True'))

    if method == 'simple':
        patch = PatchifySimple(size=size, hn=hn, wn=wn)

    elif method == 'overlap':
        patch = PatchifySimple(size=size, hn=hn, wn=wn, 
        overlap=True)
    
    elif method in ['fastercnn', 'frcnn']:
        patch = PatchifyPytorch(device=device, size=size, nms=nms, filter_bb=filter_bb)

    elif method in ['overlap/fastercnn', 'overlap/frcnn', 
                        'fastercnn/overlap', 'frcnn/overlap']:
        patch = PatchifyPytorch(device=device, size=size, 
                    prior=True, hn=hn, wn=wn, nms=nms, filter_bb=filter_bb)

    elif method in ['marqo-yolo', 'yolox']:
        patch = PatchifyYolox(device=device, size=size)
    
    elif method in ['dino/v1', 'dino/v2']:
        if 'v1' in method:
            patch = PatchifyViT(device=device, filter_bb=True, size=size,
                        attention_method='abs', nms=True, replace_small=True)
        else:
            patch = PatchifyViT(device=device, filter_bb=True, size=size,
                        attention_method='pos', nms=True, replace_small=True)
    else:
        raise ValueError(f"unexpected image chunking type. found {method}")
    try:
        patch.infer(image)
        patch.process()
    except PIL.UnidentifiedImageError as e:
        raise ChunkerError from e
    return patch.patches,patch.bboxes_orig

class PatchifySimple:
    """class to do the patching
    """
    def __init__(self, size=(512, 512), hn=3, wn=3, overlap=False, **kwargs):

        self.size = size
        self.hn = hn
        self.wn = wn
        self.overlap = overlap

    def infer(self, image):

        self.image = format_and_load_CLIP_image(image)
        self.original_size = self.image.size
        self.image_resized = self.image.resize(self.size)
        self.bboxes_simple = generate_boxes(self.size, self.hn, self.wn, overlap=self.overlap)

    def process(self):
        
        # we add the original unchanged so that it is always in the index
        # the bb of the original also provides the size which is required for later processing
        self.bboxes = [(0,0,self.size[0],self.size[1])] + self.bboxes_simple
        self.patches = patchify_image(self.image_resized, self.bboxes)

        self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]



class PatchifyModel:
    """class to do the patching
    """
    def __init__(self, device='cpu', size=(224, 224), min_area = 60*60, 
                nms=True, replace_small=True, top_k=10, 
                filter_bb=True, min_area_replace = 60*60, **kwargs):

        self.scores = []

        # this is the resized size 
        self.size = size
        self.device = device

        self.min_area = min_area
        self.min_area_replace = min_area_replace
        self.nms = nms
        self.replace_small = replace_small
        self.top_k = top_k
        self.filter_bb = filter_bb
        self.new_size = (100,100)
        # consider changins
        self.iou_thresh = 0.6
        self.kwargs = kwargs
        self.n_postfilter = None

        # this one happens at the first stage before processing the bboxes
        self.top_k_scores = 100

        self._get_model_specific_parameters()
        self._load_and_cache_model()

    def _get_model_specific_parameters(self):
        # fill in with specifics
        self.model_name = None
        self.model_load_function = lambda x:x
        self.allowed_model_types = ()

    def _load_and_cache_model(self):
        model_type = (self.model_name, self.device)

        if model_type not in available_models:
            logger.info(f"loading model {model_type}")
            if model_type[0] in self.allowed_model_types:
                func = self.model_load_function
            else:
                raise TypeError(f"wrong model for {model_type}")

            self.model, self.preprocess = func(self.model_name, self.device)

            available_models[model_type] = (self.model, self.preprocess)
        else:
            self.model, self.preprocess = available_models[model_type]

    def _load_image(self, image):
        self.image, self.image_pt, self.original_size = load_rcnn_image(image, size=self.size)

    def infer(self, image):
        self._load_image(image)
        # input is image
        pass
        # output are unprocessed bounding boxes        

    def _filter_bb(self):
        if self.filter_bb:
            self.n_prefilter = len(self.boxes_xyxy)
            self.inds = filter_boxes(self.boxes_xyxy, min_area = self.min_area)
            self.boxes_xyxy = [bb for ind,bb in enumerate(self.boxes_xyxy) if ind in self.inds]
            if len(self.scores) == self.n_prefilter:
                self.scores = [bb for ind,bb in enumerate(self.scores) if ind in self.inds]
            self.n_postfilter = len(self.boxes_xyxy)
            logger.info(f"filtered {self.n_prefilter} boxes to {self.n_postfilter}")

    def _replace_small_bb(self):
        if self.replace_small:
            if len(self.boxes_xyxy):
                self.boxes_xyxy = replace_small_boxes(self.boxes_xyxy, min_area=self.min_area_replace, 
                                new_size=self.new_size)
                self.boxes_xyxy = clip_boxes(self.boxes_xyxy, 0, 0, self.size[0], self.size[1])

   
    def _nms_bb(self):
        if self.nms:
            if len(self.boxes_xyxy) > 1:
                logger.info(f"doing nms for {len(self.boxes_xyxy)} {self.n_postfilter} boxes...")
                #print(self.scores.shape, len(self.boxes_xyxy))
                self.scores_pt = torch.tensor(self.scores, dtype=torch.float32)
                
                self.inds = torchvision.ops.nms(torch.tensor(self.boxes_xyxy, dtype=torch.float32), 
                                                                    self.scores_pt.squeeze(), self.iou_thresh)
                
                self.boxes_xyxy =  [self.boxes_xyxy[ind] for ind in self.inds]
                self.scores =  [self.scores[ind] for ind in self.inds]

    def _keep_top_k_sorted(self):
        if len(self.scores) > self.top_k_scores:
            self.inds = np.argsort(np.array(self.scores).squeeze())[::-1][:self.top_k_scores]
            self.boxes_xyxy = [self.boxes_xyxy[ind] for ind in self.inds]
            self.scores = [self.scores[ind] for ind in self.inds]

    def _keep_top_k(self):
        if self.top_k is not None and self.top_k > len(self.boxes_xyxy):
            self.boxes_xyxy = _keep_topk(self.boxes_xyxy, k=self.top_k)

    def process(self):

        # v1
        # self._replace_small_bb()
        # self._filter_bb()
        # self._nms_bb()
        # self._keep_top_k()

        self._filter_bb()
        self._replace_small_bb()
        self._nms_bb()
        self._keep_top_k()


        # we add the original unchanged so that it is always in the index
        # the bb of the original also provides the size which is required for later processing
        self.bboxes = [(0,0,self.size[0],self.size[1])] + self.boxes_xyxy
        #print(self.bboxes)
        self.patches = patchify_image(self.image, self.bboxes)

        self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]


class PatchifyViT(PatchifyModel):
    """class to do the patching
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def _get_model_specific_parameters(self):
     
        # fill in with specifics
        self.model_name = 'vit_small'
        self.patch_size = 16
        self.attention_method = self.kwargs.get('attention_method', 'pos')

        self.model_load_function = partial(_load_DINO_model, patch_size=self.patch_size)
        self.allowed_model_types = ('vit_small', 'vit_base')


    def infer(self, image):
        self._load_image(image)

        self.attentions = DINO_inference(self.model, self.preprocess, self.image, 
                            self.patch_size, device=self.device)

        self.attentions_processed = self._process_attention(self.attentions, method=self.attention_method)
        # TODO try each and maybe a then some nms? or box merging?
        
        self.boxes_xyxy = []
        for attention in self.attentions_processed:
            self.boxes_xyxy += attention_to_bboxs(attention)

        self._calc_scores_bb()
        self._keep_top_k_sorted()

    def _calc_scores_bb(self):
        if len(self.boxes_xyxy) > 0:        
            self.scores = calc_area(self.boxes_xyxy, self.size)
        

    @staticmethod
    def _process_attention(attentions: ndarray, method: str = "abs") -> List[ndarray]:

        if method.startswith('abs'):
            return np.abs(attentions).mean(0)[np.newaxis, ...]

        elif method.startswith('pos'):
            attentions_copy = attentions[:]
            attentions_copy[attentions<0] = 0
            return attentions_copy

        else:
            raise TypeError(f"unknown method of {method}")


# class PatchifyViT(Patchify):
#     """class to do the patching
#     """
#     def __init__(self, device='cpu', size=(224, 224), filter_bb=True, 
#                     attention_method='abs', min_area = 40*40, nms=True, 
#                     replace_small=True):

#         self.size = size
#         self.device = device
#         self.input_shape = (224, 224)
#         self.model_name = "vit_small"
#         self.patch_size = 16
#         self.attention_method = attention_method
#         self.min_area = min_area
#         self.nms = nms
#         self.replace_small = replace_small
#         self.top_k = 10

#         model_type = (self.model_name, device)

#         if model_type not in available_models:
#             logger.info(f"loading model {model_type}")
#             if model_type[0] in ['vit_small', 'vit_base']:
#                 func = _load_DINO_model
#             else:
#                 raise TypeError(f"wrong model for {model_type}")

#             self.model, self.preprocess = func(self.model_name, self.patch_size, self.device)

#             available_models[model_type] = (self.model, self.preprocess)
#         else:
#             self.model, self.preprocess = available_models[model_type]

#         self.filter_bb = filter_bb

#     def infer(self, image):

#         self.image, self.image_pt, self.original_size = load_rcnn_image(image, size=self.size)
        
#         self.attentions = DINO_inference(self.model, self.preprocess, self.image, 
#                             self.patch_size, device=self.device)

#     @staticmethod
#     def _process_attention(attentions: ndarray, method: str = "abs") -> List[ndarray]:

#         if method.startswith('abs'):
#             return np.abs(attentions).mean(0)[np.newaxis, ...]

#         elif method.startswith('pos'):
#             attentions_copy = attentions[:]
#             attentions_copy[attentions<0] = 0
#             return attentions_copy

#         else:
#             raise TypeError(f"unknown method of {method}")

        
#     def process(self):

#         self.attentions_processed = self._process_attention(self.attentions, method=self.attention_method)
#         # TODO try each and maybe a then some nms? or box merging?
        
#         self.boxes_xyxy = []
#         for attention in self.attentions_processed:
#             self.boxes_xyxy += attention_to_bboxs(attention)

#         if self.filter_bb:
#             self.inds = filter_boxes(self.boxes_xyxy, min_area = self.min_area)
#             self.boxes_xyxy = [bb for ind,bb in enumerate(self.boxes_xyxy) if ind in self.inds]
#             # TODO update this
#             # self.boxes_xyxy = replace_small_boxes(self.boxes_xyxy, min_area=40*40, new_size=(80,80))
        
#         if self.replace_small:
#             if len(self.boxes_xyxy):
#                 self.boxes_xyxy = replace_small_boxes(self.boxes_xyxy, min_area=40*40, 
#                                 new_size=(100,100))
#                 self.boxes_xyxy = clip_boxes(self.boxes_xyxy, 0, 0, self.size[0], self.size[1])

#         if self.nms:
#             if len(self.boxes_xyxy) > 0:
#                 self.areas = torch.tensor(calc_area(self.boxes_xyxy, self.input_shape), dtype=torch.float32)
#                 self.inds = torchvision.ops.nms(torch.tensor(self.boxes_xyxy, dtype=torch.float32), 1 - self.areas, 0.6)
#                 self.boxes_xyxy =  [self.boxes_xyxy[ind] for ind in self.inds]

#         if self.top_k is not None:
#             self.boxes_xyxy = _keep_topk(self.boxes_xyxy, k=self.top_k)

#         # we add the original unchanged so that it is always in the index
#         # the bb of the original also provides the size which is required for later processing
#         self.bboxes = [(0,0,self.size[0],self.size[1])] + self.boxes_xyxy
#         #print(self.bboxes)
#         self.patches = patchify_image(self.image, self.bboxes)

#         self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]


class PatchifyPytorch(PatchifyModel):
    """class to do the patching
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.top_k_scores = 100

    def _get_model_specific_parameters(self):
     
        # fill in with specifics
        self.model_name = 'faster_rcnn'

        self.model_load_function = load_pytorch
        
        self.allowed_model_types = (self.model_name)
        self.input_shape = (384, 384)
        self.inds = []
        self.iou_thresh = 0.6

    def _keep_topk_sorted(self):
        if len(self.scores) > self.top_k_scores:
            self.inds = np.argsort(np.array(self.scores).squeeze())[::-1][:self.top_k_scores]
            self.boxes_xyxy = [self.boxes_xyxy[ind] for ind in self.inds]
            self.scores = [self.scores[ind] for ind in self.inds]

    def infer(self, image):
        self._load_image(image)
        self.batch = [self.preprocess(self.image_pt.to(self.device))]
        with torch.no_grad():
            self.results = self.model(self.batch)[0]

        self.boxes_xyxy = self.results['boxes'].detach().cpu().numpy()
        self.scores = self.results['scores'].detach().cpu().numpy()

        if isinstance(self.scores, (np.ndarray, np.generic)):
            self.scores = self.scores.tolist()

        self._keep_topk_sorted()
    

class PatchifyYolox(PatchifyModel):
    """class to do the patching
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.top_k_scores = 100

    def _get_model_specific_parameters(self):
     
        # fill in with specifics
        self.model_name = "/mnt/internal1/datasets/LVIS/YOLOX/yolox_s.onnx"
       
        self.model_load_function = load_yolox_onnx
        self.allowed_model_types = (self.model_name)
        self.input_shape = (384, 384)
        self.inds = []
        self.iou_thresh = 0.6

    def _keep_topk_sorted(self):
        if len(self.scores) > self.top_k_scores:
            self.inds = np.argsort(np.array(self.scores).squeeze())[::-1][:self.top_k_scores]
            self.boxes_xyxy = [self.boxes_xyxy[ind] for ind in self.inds]
            self.scores = [self.scores[ind] for ind in self.inds]

    def infer(self, image):
        self._load_image(image)

        # make cv2 format
        self.image_cv = _PIL_to_opencv(self.image)
        
        self.results, self.ratio = _infer_yolox(session=self.model, 
                            preprocess=self.preprocess, opencv_image=self.image_cv, 
                            input_shape=self.input_shape)

        self.boxes_xyxy, self.scores =  _process_yolox(output=self.results, ratio=self.ratio, size=self.input_shape)
        if isinstance(self.scores, (np.ndarray, np.generic)):
            self.scores = self.scores.tolist()
        
        self._keep_topk_sorted()