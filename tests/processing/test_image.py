import unittest
import tempfile
import os

import numpy as np
from PIL import Image

from marqo.s2_inference.processing.image import (
    PatchifySimple,
    PatchifyPytorch,
    PatchifyViT,
    PatchifyYolox,
    patchify_image,
    chunk_image,
)

from marqo.s2_inference.s2_inference import clear_loaded_models

class TestImageChunking(unittest.TestCase):

    def setUp(self) -> None:
        pass 

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

    def test_PatchifyVit(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifyViT(size=image_size, attention_method='abs')
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyViT(size=image_size)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyViT(size=image_size, attention_method='pos')
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyViT(size=image_size)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

    def test_PatchifyYolox(self):

        image_size = (400,500)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255,size=image_size).astype(np.uint8))
            img.save(temp_file_name)

            patcher = PatchifyYolox(size=image_size)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyYolox(size=image_size)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyYolox(size=image_size)
            patcher.infer(img)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

            patcher = PatchifyYolox(size=image_size)
            patcher.infer(temp_file_name)
            patcher.process()

            assert len(patcher.patches) == len(patcher.bboxes)
            assert len(patcher.patches) == len(patcher.bboxes_orig)
            assert abs(np.array(patcher.patches[0]) - np.array(patcher.image)).sum() < 1e-6

    
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

    def test_chunk_image_yolox(self):

        SIZE = (256, 384)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, 
                            size=SIZE).astype(np.uint8))
            img.save(temp_file_name)

            patches, bboxes = chunk_image(img, device='cpu', 
                                method = 'yolox', size=SIZE)
            
            assert len(patches) >= 1
            assert patches[0].size == SIZE

    def test_chunk_image_dino(self):

        SIZE = (256, 384)
        with tempfile.TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'test_image.png')
            img = Image.fromarray(np.random.randint(0,255, 
                            size=SIZE).astype(np.uint8))
            img.save(temp_file_name)

            patches, bboxes = chunk_image(img, device='cpu', 
                    method = 'dino/v1', size=SIZE)
            
            assert len(patches) >= 1
            assert patches[0].size == SIZE

            patches, bboxes = chunk_image(img, device='cpu', 
                    method = 'dino/v2', size=SIZE)
            
            assert len(patches) >= 1
            assert patches[0].size == SIZE

