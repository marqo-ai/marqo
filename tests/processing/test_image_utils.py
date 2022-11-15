import unittest
import tempfile
import os

import numpy as np
from PIL import Image
 

from marqo.s2_inference.types import List, Tuple, ImageType

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

from marqo.s2_inference.processing.image import (
    chunk_image,
)


class TestImageUtils(unittest.TestCase):

    def setUp(self) -> None:
        pass 
    
    def test_get_default_size(self):

        size = get_default_size()
        assert isinstance(size, (Tuple, List))

        assert len(size) == 2

    def test_PIL_to_opencv(self):

        image_size = (250, 200)
        img = Image.fromarray(np.random.randint(0,255, size=image_size).astype(np.uint8))

        assert isinstance(img, ImageType)

        img_cv2 = _PIL_to_opencv(img)

        assert isinstance(img_cv2, np.ndarray)

        assert np.allclose( np.array(img.convert('RGB'))[:, :, ::-1],  img_cv2)

    def test_replace_small_boxes(self):
        
        def _calc_area(box): return (box[2]-box[0])*(box[3] - box[1])

        min_areas = [10*10, 40*40, 100*100]
        new_sizes = [(100,100), (11,20), (45,67)]
        for min_area,new_size in zip(min_areas, new_sizes):

            boxes = [(0,0,10,20), (1,4,23,89), (0,0,211,32)]
            new_boxes = replace_small_boxes(boxes, min_area=min_area, new_size=new_size)

            for box,new_box in zip(boxes, new_boxes):
                area = _calc_area(box)
                if area < min_area:
                    assert box != new_box
                    assert _calc_area(new_box) == new_size[0]*new_size[1]
                else:
                    assert box == new_box

    def test_keep_topk(self):
        boxes = [(0,0,10,20), (1,4,23,89), (0,0,211,32)]
        # xmin, ymin, xmax, ymax

        top_k = _keep_topk(boxes, k=1)
        assert len(top_k) == 1
        assert top_k[0] == boxes[0]

        top_k = _keep_topk(boxes, k=2)
        assert len(top_k) == 2
        assert top_k[0] == boxes[0]
        assert top_k[1] == boxes[1]

        top_k = _keep_topk(boxes, k=0)
        assert len(top_k) == 0

        top_k = _keep_topk(boxes, k=len(boxes)+1)
        assert len(top_k) == len(boxes)


    def test_clip_boxes(self):
        boxes = [(0,0,10,20), (1,4,23,89), (0,0,211,32)]
        # xmin, ymin, xmax, ymax
        limits = [(1,2,20,20), (0,0,40,90), (-1,5,11,31)]

        for box,limit in zip(boxes, limits):
            new_boxs = clip_boxes([box], *limit)
            new_box = new_boxs[0]
            
            if box[0] < limit[0]:
                assert new_box[0] == limit[0]
            else:
                assert new_box[0] == box[0]

            if box[1] < limit[1]:
                assert new_box[1] == limit[1]
            else:
                assert new_box[1] == box[1]

            if box[2] > limit[2]:
                assert new_box[2] == limit[2]
            else:
                assert new_box[2] == box[2]

            if box[3] > limit[3]:
                assert new_box[3] == limit[3]
            else:
                assert new_box[3] == box[3]

            


    def test_load_rcnn_image(self):
        image_size = (100,100,3)
        scaled_size = (320,320)

        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)
            gt_size = img.size
            image, image_pt,original_size = load_rcnn_image(temp_file_name, size=scaled_size)

            assert image.size == scaled_size[:2]
            assert original_size == image_size[:2]
            assert gt_size == image_size[:2]
            assert image_pt.shape[0] == 3
            assert image_pt.shape[1:] == scaled_size

    def test_area(self):

        bboxes = [(0,0,10,10), (10,10,11,11)]
        areas_gt = [ (bb[3] - bb[1])*(bb[2] - bb[0]) for bb in bboxes]

        areas = calc_area(bboxes)

        assert abs(np.array(areas_gt) - np.array(areas)).sum() < 1e-6

        areas = calc_area(bboxes, size = (20,20))
        areas_gt = [ (bb[3] - bb[1])*(bb[2] - bb[0])/(20*20) for bb in bboxes]

        assert abs(np.array(areas_gt) - np.array(areas)).sum() < 1e-6

    def test_filter_boxes(self):
        boxes = [[0,0,100,100], [0,0,200,100], [5,3,50,700], [1,1,3,3]]

        # should filter out the 3rd and 4th
        inds = filter_boxes(boxes, max_aspect_ratio=2.1, min_area=10)
        assert inds == [0,1]

        # should filter out all boxes because of aspect ratio
        inds = filter_boxes(boxes, max_aspect_ratio=1, min_area=10)
        assert inds == []

        # should filter out all except 1st due to aspect ratio
        inds = filter_boxes(boxes, max_aspect_ratio=1.01, min_area=10)
        assert inds == [0]

        # should filter last only due to small area
        inds = filter_boxes(boxes, max_aspect_ratio=100, min_area=10)
        assert inds == [0, 1, 2]

        # not filter any
        inds = filter_boxes(boxes, max_aspect_ratio=100, min_area=1)
        assert inds == [0, 1, 2, 3]

        # filter all because of area
        inds = filter_boxes(boxes, max_aspect_ratio=100, min_area=1e6)
        assert inds == []

    def test_rescale_box(self):

        boxes = [[0,0,100,100], [0,0,200,100], [5,3,50,70]]

        boxes_gt = [(0.0, 0.0, 200.0, 200.0), (0.0, 0.0, 300.0, 200.0), (10.0, 6.0, 100.0, 140.0)]

        original_sizes = [(100,100), (200,100), (50,70)]

        target_sizes = [(200,200), (300,200), (100,140)]

        for gt,bb,orig_size,target_size in zip(boxes_gt,boxes, original_sizes, target_sizes):
            res_bb = rescale_box(bb, orig_size, target_size)
            inv_bb = rescale_box(res_bb, target_size, orig_size)
            
            assert abs(np.array(bb) - np.array(inv_bb)).sum() < 1e-6
            assert abs(np.array(gt) - np.array(res_bb)).sum() < 1e-6

    def test_generate_boxes(self):

        image_size = (100,100)
        hn = 4
        wn = 4
        width = image_size[0]//wn
        height = image_size[1]//hn

        bboxes = generate_boxes(image_size, hn, wn)

        assert len(bboxes) == (hn*wn)
        assert bboxes[0] == (0, 0, image_size[0]//wn, image_size[1]//hn)
        assert bboxes[-1] == ( width*(wn-1), height*(hn-1), image_size[0], image_size[1])

        image_size = (100,100)
        hn = 3
        wn = 2
        width = image_size[0]//wn
        height = image_size[1]//hn

        bboxes = generate_boxes(image_size, hn, wn)

        assert len(bboxes) == (hn*wn)
        assert bboxes[0] == (0, 0, image_size[0]//wn, image_size[1]//hn)
        assert bboxes[-1] == ( width*(wn-1), height*(hn-1), image_size[0], image_size[1] -1 ) # rounding

        image_size = (150,120)
        hn = 3
        wn = 6
        width = image_size[0]//wn
        height = image_size[1]//hn

        bboxes = generate_boxes(image_size, hn, wn)

        assert len(bboxes) == (hn*wn)
        assert bboxes[0] == (0, 0, image_size[0]//wn, image_size[1]//hn)
        assert bboxes[-1] == ( width*(wn-1), height*(hn-1), image_size[0], image_size[1]) 

    def test_generate_boxes_overlap(self):
        
        def within_a_pixel(arr1, arr2, tolerance=2):
            return np.abs(np.array(arr1) - np.array(arr2)).sum() <= tolerance

        test_sizes = [((100,100), 2 , 2), ((150,100), 2 , 2), ((150,100), 3 , 2),
                    ((240,240), 3 , 3), ((240,240), 4 , 3), ((240,240), 3 , 4)]

        for image_size, hn, wn, in test_sizes:

            width = image_size[0]//wn
            height = image_size[1]//hn

            bboxes = generate_boxes(image_size, hn, wn, overlap=True)

            assert len(set(bboxes)) == (hn*wn) + (hn-1)*(wn-1)
            assert bboxes[0] == (0, 0, image_size[0]//wn, image_size[1]//hn)
            assert within_a_pixel(bboxes[-1], ( width*(wn-1), height*(hn-1), image_size[0], image_size[1]))

    def test_process_patch_method(self):

        urls = (
            ('simple', 'simple', dict()),
            ('overlap', 'overlap', dict()),
            ('simple?hn=3', 'simple', {'hn':'3'}),
            ('overlap?hn=3', 'overlap', {'hn':'3'}),
            ('simple?wn=3', 'simple', {'wn':'3'}),
            ('overlap?wn=3', 'overlap', {'wn':'3'}),
            ('simple?hn=3&wn=4', 'simple', {'hn':'3', 'wn':'4'}),
            ('overlap?hn=3&wn=4', 'overlap', {'hn':'3', 'wn':'4'}),
            ('simple?wn=3&wn=4', 'simple', {'wn':'3', 'wn':'4'}),
            ('overlap?wn=3&wn=4','overlap', {'wn':'3', 'wn':'4'}),
        )

        for url,path,params in urls:
            path_out, params_out = _process_patch_method(url)
            assert path_out == path
            assert params_out == params


    def test_patchify(self):

        image_size = (250, 200)
        bboxes = [(0,0,250,200), (10, 20, 50, 70), (200, 150, 250, 200)]
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, size=image_size).astype(np.uint8))
            patches = patchify_image(img, bboxes)

            assert len(patches) == len(bboxes)
            assert isinstance(patches, list)
            
            for patch,bb in zip(patches, bboxes):
                a = bb[2] - bb[0]
                b = bb[3] - bb[1]
                assert patch.size == (a, b)

    def test_chunk_image_simple(self):

        SIZE = (256, 384)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, 
                            size=SIZE).astype(np.uint8))
            img.save(temp_file_name)

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'simple', size=SIZE)
            
            assert len(patches) == (3*3) + 1
            assert patches[0].size == SIZE

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'simple?hn=2&wn=3', size=SIZE)
            
            assert len(patches) == (2*3) + 1, len(patches)
            assert patches[0].size == SIZE


    def test_chunk_image_overlap(self):

        SIZE = (256, 384)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, 
                            size=SIZE).astype(np.uint8))
            img.save(temp_file_name)

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'overlap', size=SIZE)
            
            assert len(patches) == (3*3) + (3-1)*(3-1) +  1
            assert patches[0].size == SIZE

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'overlap?wn=4&hn=2', size=SIZE)
            
            assert len(patches) == (4*2) + (4-1)*(2-1) +  1
            assert patches[0].size == SIZE

    def test_chunk_image_pytorch(self):

        SIZE = (256, 384)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, 
                            size=SIZE).astype(np.uint8))
            img.save(temp_file_name)

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'frcnn', size=SIZE)
            
            assert len(patches) >= 1
            assert patches[0].size == SIZE

    def test_str2bool(self):

        assert str2bool('true')
        assert str2bool('True')
        assert str2bool('T')
        assert str2bool('t')        
        assert str2bool('1')

        assert not str2bool('false')
        assert not str2bool('False')
        assert not str2bool('F')
        assert not str2bool('f')        
        assert not str2bool('0')

        assert not str2bool('foo')
        assert not str2bool('hello')
        assert not str2bool('something')
        assert not str2bool(' ')        
        assert not str2bool('sd;lkmsnd')
