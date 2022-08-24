import unittest
import tempfile
import os

import numpy as np
from PIL import Image

from marqo.s2_inference.processing.image import (
    load_rcnn_image, 
    calc_area,
    filter_boxes,
    rescale_box,
    generate_boxes,
    PatchifySimple,
    patchify_image,
)


class TestImageChunking(unittest.TestCase):

    def setUp(self) -> None:
        pass 

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
        width = image_size[1]//wn
        height = image_size[0]//hn

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

    def test_PatchifySimple(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifySimple(size=image_size, hn=3, wn=3)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image_resized)).sum() < 1e-6

            patcher = PatchifySimple(size=image_size, hn=3, wn=3)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image_resized)).sum() < 1e-6

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