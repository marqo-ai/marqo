from functools import partial

import PIL
import numpy as np
import torch
import torchvision

from marqo.s2_inference.s2_inference import available_models,_create_model_cache_key
from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, ndarray, Literal
from marqo.s2_inference.clip_utils import format_and_load_CLIP_image
from marqo.s2_inference.errors import ChunkerError

from marqo.s2_inference.processing.DINO_utils import _load_DINO_model,attention_to_bboxs,DINO_inference
from marqo.s2_inference.processing.pytorch_utils import load_pytorch
from marqo.s2_inference.processing.yolox_utils import (
   _process_yolox,
    _infer_yolox, 
    load_yolox_onnx,
    get_default_yolox_model,
    _download_yolox
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

logger = get_logger(__name__)


def chunk_image(image: Union[str, ImageType], device: str, 
                        method: Literal[ 'simple', 'overlap',  'frcnn', 'marqo-yolo', 'yolox', 'dino-v1', 'dino-v2'],
                        size=get_default_size()) -> Tuple[List[ImageType], ndarray]:
    """_summary_
    wrapper function to do the chunking and return the patches and their bounding boxes
    in the original coordinates system

    Args:
        image (Union[str, ImageType]): image to process
        device (str): device to load models onto
        method (str, optional): the method to use. Defaults to 'simple'.
        size (_type_, optional): size the image should be loaded in as. Defaults to get_default_size().

    Raises:
        TypeError: _description_
        ValueError: _description_
        ChunkerError: Raises ChunkerError, if the chunker can't work for some reason

    Returns:
        Tuple[List[ImageType], ndarray]: list of PIL images and the corresponding bounding boxes
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
    
    # get the parameters from the method 'url'
    method, params = _process_patch_method(method)
    logger.debug(f"found method={method} and params={params}")

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

    elif method in ['marqo-yolo', 'yolox']:
        patch = PatchifyYolox(device=device, size=size)
    
    elif method in ['dino-v1', 'dino-v2', 'dino/v1', 'dino/v2']:
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
    """class to do the patching. this one creates non-overlapping boixes and chunks the image
    """
    def __init__(self, size: Tuple = (512, 512), hn: int = 3, wn: int = 3, overlap: bool = False, **kwargs):
        """_summary_

        Args:
            size (Tuple, optional): size the image is resied to. Defaults to (512, 512).
            hn (int, optional): number of boxes in the horizontal. Defaults to 3.
            wn (int, optional): number of boxes in the vertical. Defaults to 3.
            overlap (bool, optional): should they also have overlapping boxes? Defaults to False.
        """
        self.size = size
        self.hn = hn
        self.wn = wn
        self.overlap = overlap

    def infer(self, image: Union[str, ImageType]):

        self.image = format_and_load_CLIP_image(image, {})
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
    """class to do the patching. this is the base class for model based chunking
    """
    def __init__(self, device: str = 'cpu', size: Tuple = (224, 224), min_area: float = 60*60, 
                nms: bool = True, replace_small: bool = True, top_k: int = 10, 
                filter_bb: bool = True, min_area_replace: float = 60*60, **kwargs):
        """_summary_

        Args:
            device (str, optional): the device to run the model on. Defaults to 'cpu'.
            size (Tuple, optional): the final image size to go to the model. Defaults to (224, 224).
            min_area (float, optional): the min area (pixels) that a box must meet to be kept. 
                areas lower than this are removed. Defaults to 60*60.
            nms (bool, optional): perform nms or not. Defaults to True.
            replace_small (bool, optional): boxes smaller than min_area_replace are replaced with
                        boxes centered on the same position but larger size. Defaults to True.
            top_k (int, optional): keep this many boxes after all processin (max). Defaults to 10.
            filter_bb (bool, optional): perform filtering on the proposed boxes. Defaults to True.
            min_area_replace (float, optional): boxes with areas smaller than this are replaced with larger ones. Defaults to 60*60.
        """
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
        model_cache_key = _create_model_cache_key(self.model_name, self.device)

        if model_cache_key not in available_models:
            logger.info(f"loading model {model_type}")
            if model_type[0] in self.allowed_model_types:
                func = self.model_load_function
            else:
                raise TypeError(f"wrong model for {model_type}")

            self.model, self.preprocess = func(self.model_name, self.device)

            available_models[model_cache_key] = self.model, self.preprocess
        else:
            self.model, self.preprocess = available_models[model_cache_key]

    def _load_image(self, image):
        self.image, self.image_pt, self.original_size = load_rcnn_image(image, size=self.size)

    def infer(self, image):
        self._load_image(image)
        # input is image
        pass
        # output are unprocessed bounding boxes        

    def _filter_bb(self):
        """filters bounding boxes based on size and aspect ratio
        """
        if self.filter_bb:
            self.n_prefilter = len(self.boxes_xyxy)
            self.inds = filter_boxes(self.boxes_xyxy, min_area = self.min_area)
            self.boxes_xyxy = [bb for ind,bb in enumerate(self.boxes_xyxy) if ind in self.inds]
            if len(self.scores) == self.n_prefilter:
                self.scores = [bb for ind,bb in enumerate(self.scores) if ind in self.inds]
            self.n_postfilter = len(self.boxes_xyxy)
            logger.debug(f"filtered {self.n_prefilter} boxes to {self.n_postfilter}")

    def _replace_small_bb(self):
        """replaces boxes that are smaller than some area with a minimum sized box centered
        on the old one. if the box is out of bounds it is clipped to the image boundries
        """
        if self.replace_small:
            if len(self.boxes_xyxy):
                self.boxes_xyxy = replace_small_boxes(self.boxes_xyxy, min_area=self.min_area_replace, 
                                new_size=self.new_size)
                self.boxes_xyxy = clip_boxes(self.boxes_xyxy, 0, 0, self.size[0], self.size[1])

    def _nms_bb(self):
        """performs class agnostic nms over the bounding boxes
        """
        if self.nms:
            if len(self.boxes_xyxy) > 1:
                logger.debug(f"doing nms for {len(self.boxes_xyxy)} {self.n_postfilter} boxes...")
                self.scores_pt = torch.tensor(self.scores, dtype=torch.float32)
                
                self.inds = torchvision.ops.nms(torch.tensor(self.boxes_xyxy, dtype=torch.float32), 
                                                                    self.scores_pt.squeeze(), self.iou_thresh)
                
                self.boxes_xyxy =  [self.boxes_xyxy[ind] for ind in self.inds]
                self.scores =  [self.scores[ind] for ind in self.inds]

    def _keep_top_k_sorted(self):
        """sort the boxes based on score and keep top k
        """
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
        
        self.patches = patchify_image(self.image, self.bboxes)

        self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]


class PatchifyViT(PatchifyModel):
    """class to do the patching for an attention based model
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
        
        self.boxes_xyxy = []
        for attention in self.attentions_processed:
            self.boxes_xyxy += attention_to_bboxs(attention)

        self._calc_scores_bb()
        self._keep_top_k_sorted()

    def _calc_scores_bb(self):
        """we have no scores for the boxes so we go off area
        """
        if len(self.boxes_xyxy) > 0:        
            self.scores = calc_area(self.boxes_xyxy, self.size)
        
    @staticmethod
    def _process_attention(attentions: ndarray, method: Literal['abs', 'pos']) -> List[ndarray]:
        """processes a N x grey-scale attention maps 

        Args:
            attentions (ndarray):
            method (str, optional):  Defaults to "abs".

        Raises:
            TypeError: _description_

        Returns:
            List[ndarray]: _description_
        """
        if method.startswith('abs'):
            return np.abs(attentions).mean(0)[np.newaxis, ...]

        elif method.startswith('pos'):
            attentions_copy = attentions[:]
            attentions_copy[attentions<0] = 0
            return attentions_copy

        else:
            raise TypeError(f"unknown method of {method}")


class PatchifyPytorch(PatchifyModel):
    """class to do the patching for a pytorch based object detector
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # keep this many based on initial scoring before doing nms and other processing
        self.top_k_scores = 100

    def _get_model_specific_parameters(self):
     
        # fill in with specifics
        self.model_name = 'faster_rcnn'

        self.model_load_function = load_pytorch
        
        self.allowed_model_types = (self.model_name)
        self.input_shape = (384, 384)
        self.inds = []
        self.iou_thresh = 0.6

    def infer(self, image):
        self._load_image(image)
        self.batch = [self.preprocess(self.image_pt.to(self.device))]
        with torch.no_grad():
            self.results = self.model(self.batch)[0]

        self.boxes_xyxy = self.results['boxes'].detach().cpu().numpy()
        self.scores = self.results['scores'].detach().cpu().numpy()

        if isinstance(self.scores, (np.ndarray, np.generic)):
            self.scores = self.scores.tolist()

        self._keep_top_k_sorted()
    

class PatchifyYolox(PatchifyModel):
    """class to do the patching for a onnx yolox model
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.top_k_scores = 100

    def _get_model_specific_parameters(self):
     
        # fill in with specifics
        self.yolox_default = get_default_yolox_model()
        self.model_name = _download_yolox(**self.yolox_default)
       
        self.model_load_function = load_yolox_onnx
        self.allowed_model_types = (self.model_name)
        self.input_shape = (384, 384)
        self.inds = []
        self.iou_thresh = 0.6

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
        
        self._keep_top_k_sorted()