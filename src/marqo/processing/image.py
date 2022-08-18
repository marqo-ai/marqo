import copy

from PIL import Image
import numpy as np

import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
import torchvision

from marqo.s2_inference.s2_inference import available_models
from marqo.s2_inference.s2_inference import get_logger
from marqo.s2_inference.types import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray
from marqo.s2_inference.clip_utils import format_and_load_CLIP_image

logger = get_logger('image_chunks')

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

def get_default_size() -> Tuple:
    """this sets the default image size used for inference for the chunker

    Returns:
        Tuple: _description_
    """
    return (240,240)

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


def load_rcnn_image(image_name: str, size: Tuple = (320,320)) -> Tuple[ImageType, FloatTensor, Tuple[int, int]]:
    """this is the loading and processing for the input

    Args:
        image_name (str): _description_
        size (Tuple, optional): _description_. Defaults to (320,320).

    Returns:
        Tuple[ImageType, FloatTensor, Tuple[int, int]]: _description_
    """
    
    if isinstance(image_name, ImageType):
        image = image_name 
    elif isinstance(image_name, str):
        image = Image.open(image_name)
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




def distance_matrix(v: Union[ndarray, FloatTensor], vectors: Union[ndarray, FloatTensor]) -> List[float]:
    """calculates the distances between a vector v and a array of vectors vectors

    Args:
        v (Union[ndarray, FloatTensor]): _description_
        vectors (Union[ndarray, FloatTensor]): _description_

    Returns:
        List[Float]: _description_
    """
    # TODO vectorize
    return [_mse(v, u) for u in vectors]

def get_centers(vectors: Union[ndarray, FloatTensor]) -> ndarray:
    """calculates the centers of rectangles given the 4-tuple (x1,y1,x2,y2)

    Args:
        vectors (Union[ndarray, FloatTensor]): _description_

    Returns:
        ndarray: _description_
    """
    x_c = (vectors[:,2] - vectors[:,0])/2 + vectors[:,0]
    y_c = (vectors[:,3] - vectors[:,1])/2 + vectors[:,1]
    
    return np.stack([x_c, y_c], axis=-1)

def _mse(a: Union[FloatTensor, ndarray], b: Union[FloatTensor, ndarray]) -> float:
    """mean squared error of two vectors

    Args:
        a (Union[FloatTensor, ndarray]): _description_
        b (Union[FloatTensor, ndarray]): _description_

    Returns:
        Float: _description_
    """
    return np.sum(np.abs(a - b)**2)

def greedy_select(vectors: Union[FloatTensor, ndarray]) -> Tuple[List[int], List[float]]:
    """greedily selects the farthest point from the centroid of all selected points

    Args:
        vectors (Union[FloatTensor, ndarray]): _description_

    Returns:
        Tuple[List[int], List[Float]]: _description_
    """
    vectors_orig = copy.deepcopy(vectors)
    vectors = copy.deepcopy(vectors)
    # selecte a vector
    out_inds = []
    largest_distances = []
    # we need the indices of our original vectors
    inds = list(range(len(vectors)))
    # we need a starting vector, could also be selected other ways
    start_ind = inds[0]
    # this is our ordered indices that we use to get the keep order
    out_inds.append(start_ind)
    # we need the start vector, this will be updated with an average
    v = vectors[start_ind]
    while len(out_inds) != len(inds): # could also terminate early

        # get the distance to all
        distances = distance_matrix(v, vectors)
        # the next is found by taking the furthest away from v
        next_ind = inds[np.argmax(distances)]
        # we add it to the list
        out_inds.append(next_ind)
        # now get the next reference ind
        v = np.mean(vectors_orig[out_inds][:3], axis=0)
        # now set all the seen ones to the mean so the are not selected
        vectors[out_inds] = v

        largest_distances.append(distances[next_ind])
    return out_inds, largest_distances

def filter_boxes(bboxes: Union[FloatTensor, ndarray], max_aspect_ratio: int = 4, min_area: int = 40*40) -> List[int]:
    """filters a list of bounding boxes given as the 4-tuple (x1, y1, x2, y2)
    by area and aspect ratio

    Args:
        bboxes (Union[FloatTensor, ndarray]): _description_
        max_aspect_ratio (int, optional): _description_. Defaults to 4.
        min_area (int, optional): _description_. Defaults to 40*40.

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


# TODO generate boxes with overlap - take the bottom corners as centers
def generate_boxes(image_size: Tuple[int, int], hn: int, wn: int) -> List[Tuple]:
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

    return bboxes



class PatchifySimple:
    """class to do the patching
    """
    def __init__(self, size=(512, 512), hn=3, wn=3, **kwargs):

        self.size = size
        self.hn = hn
        self.wn = wn


    def infer(self, image):

        self.image = format_and_load_CLIP_image(image)
        self.original_size = self.image.size
        self.image_resized = self.image.resize(self.size)
        self.bboxes_simple = generate_boxes(self.size, self.hn, self.wn)

    def process(self):
        
        # we add the original unchanged so that it is always in the index
        # the bb of the original also provides the size which is required for later processing
        self.bboxes = [(0,0,self.size[0],self.size[1])] + self.bboxes_simple
        self.patches = patchify_image(self.image_resized, self.bboxes)

        self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]


class PatchifyPytorch:
    """class to do the patching
    """
    def __init__(self, device='cpu', size=(240, 240), nms=True, filter_bb=True):

        self.size = size

        model_type = ('faster_rcnn', device)

        if model_type not in available_models:
            logger.info(f"loading model {model_type}")
            if model_type[0] == 'faster_rcnn':
                func = load_pytorch_rcnn
            elif model_type[0] == 'mobilenet':
                func = load_pretrained_mobilenet320
            else:
                raise TypeError(f"wrong model for {model_type}")
            self.model, self.preprocess = func()

            available_models[model_type] = (self.model, self.preprocess)
        else:
            self.model, self.preprocess = available_models[model_type]

        self.device = device
        self.model.to(self.device)
        self.nms = nms
        self.filter_bb = filter_bb

    def infer(self, image):

        self.image, self.image_pt, self.original_size = load_rcnn_image(image, size=self.size)
        self.batch = [self.preprocess(self.image_pt.to(self.device))]
        with torch.no_grad():
            self.results = self.model(self.batch)[0]
        
    def process(self):
        self.areas = torch.tensor(calc_area(self.results['boxes'], self.size))
        self.bboxes_pt = self.results['boxes'].detach().cpu()
        
        if self.filter_bb:
            inds = filter_boxes(self.bboxes_pt)
            self.bboxes_pt = self.bboxes_pt.clone().detach()[inds]
            self.areas = self.areas[inds]
        if self.nms:
            self.inds = torchvision.ops.nms(self.bboxes_pt, 1 - self.areas, 0.6)
            self.bboxes_pt = self.bboxes_pt.clone().detach()[self.inds]
        # we add the original unchanged so that it is always in the index
        # the bb of the original also provides the size which is required for later processing
        self.bboxes = [(0,0,self.size[0],self.size[1])] + self.bboxes_pt.numpy().astype(int).tolist()
        self.patches = patchify_image(self.image, self.bboxes)

        self.bboxes_orig = [rescale_box(bb, self.size, self.original_size) for bb in self.bboxes]

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

def chunk_image(image: Union[str, ImageType], device: str, method: str = 'simple') -> Tuple[List[ImageType], ndarray]:
    """wrapper function to do the chunking and return the patches and their bounding boxes
    in the original coordinates system

    Args:
        image (Union[str, ImageType]): _description_
        device (str): _description_

    Returns:
        Tuple[List[ImageType], ndarray]: _description_
    """

    if method in [None, 'none', '', "None", ' ']:
        if isinstance(image, str):
            return [image],[image]      
        elif isinstance(image, ImageType):
            return [image], [(0, 0, image.size[0], image.size[1])]
        else:
            raise TypeError(f'only pointers to an image or a PIL image are allowed. received {type(image)}')
    if method == 'simple':
        patch = PatchifySimple(size=(512, 512), hn=3, wn=3)
    elif method in ['fastercnn', 'frcnn']:
        patch = PatchifyPytorch(device=device, size=get_default_size())
    else:
        raise ValueError(f"unexpected image chunking type. found {method}")

    patch.infer(image)
    patch.process()

    return patch.patches,patch.bboxes_orig


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

# TODO add YOLOX https://github.com/Megvii-BaseDetection/YOLOX
 
