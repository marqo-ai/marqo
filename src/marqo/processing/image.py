from PIL import Image
import numpy as np
from marqo.s2_inference.types import *

from random import randint

def random_bbox(bbox):
    v = [randint(0, v) for v in bbox]
    left = min(v[0], v[2])
    upper = min(v[1], v[3])
    right = max(v[0], v[2])
    lower = max(v[1], v[3])
    return [left, upper, right, lower]

def random_bbox_with_offset(bbox):
    # given as (left_upper_corner, right_lower_corner) = ((x1,y1), (x2,y2))
    # The right can also be represented as (left+width)
    # and lower can be represented as (upper+height).
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # ()
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop

    max_height = bbox[2] - bbox[0]
    max_width = bbox[3] - bbox[1]

    min_height = max_height//4
    min_width = max_width//4

    x_left_offset = np.random.randint(0, max_height - min_height -1)
    y_upper_offset = np.random.randint(0, max_width - min_width -1)

    x_right_offset = np.random.randint(x_left_offset, x_left_offset + max_height)
    y_lower_offset = np.random.randint(y_upper_offset, y_upper_offset + max_height)

    assert x_left_offset < x_right_offset
    assert y_upper_offset < y_lower_offset

    return (x_left_offset, y_upper_offset, x_right_offset, y_lower_offset)
    


def _extract_patch(image, bbox):
    # assumes 
    return image.crop(bbox)

# def simple_chunker(image: Union[ndarray, ImageType]) -> List[ImageType]:

#     if isinstance(image, ImageType):
#         image = image.convert('RGB')
#         image = np.array(image)
#         assert image.shape[-1] == 3
    
#     nr,nc,ch = image.shape

#     bboxes = [(0,0),(nr//2,0),(nr//2,nc//2),(0,nc//2)]


def random_chunker(image: Union[ndarray, ImageType], n_patches: int = 5) -> Tuple[List[ImageType] ,List[ImageType]]:

    if not isinstance(image, ImageType):
        raise TypeError('wrong')
    
    # get the full size of the image in pixels
    bbox = image.getbbox()

    bboxes = [random_bbox(bbox) for i in range(n_patches)]

    patches = [_extract_patch(image, bbox) for bbox in bboxes]

    return bboxes,patches