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
    PatchifyPytorch,
    patchify_image,
    _process_patch_method,
    chunk_image,
    str2bool
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

    def test_PatchifyPytorch(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifyPytorch(size=image_size)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyPytorch(size=image_size)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

    def test_PatchifyPytorchPrior(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifyPytorch(size=image_size, hn=3, wn=3, prior=True)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6
            assert len(patcher.bboxes_simple) == (3*3) + (3-1)*(3-1)
 
            patcher = PatchifyPytorch(size=image_size, hn=3, wn=3, prior=True)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert len(patcher.bboxes_simple) == (3*3) + (3-1)*(3-1)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6


    def test_PatchifyOverlap(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifySimple(size=image_size, hn=3, wn=3, overlap=True)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            # the first term is non-overlapping, second term is overlapping, third term is original box
            assert len(set(patcher.bboxes)) == (3*3) + (3-1)*(3-1) + 1 
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image_resized)).sum() < 1e-6

            patcher = PatchifySimple(size=image_size, hn=3, wn=3, overlap=True)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert len(set(patcher.bboxes)) == (3*3) + (3-1)*(3-1) + 1
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

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'overlap/frcnn?wn=4&hn=2', size=SIZE)
            
            assert len(patches) >= (4*2) + (4-1)*(2-1) +  1
            assert patches[0].size == SIZE

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'overlap/frcnn?nms=false', size=SIZE)
            
            assert len(patches) >= (4*2) + (4-1)*(2-1) +  1
            assert patches[0].size == SIZE

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'overlap/frcnn?filter_bb=false', size=SIZE)
            
            assert len(patches) >= (4*2) + (4-1)*(2-1) +  1
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
