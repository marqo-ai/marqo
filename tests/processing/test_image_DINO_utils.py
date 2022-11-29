import unittest
import tempfile
import os
import requests

import numpy as np
from PIL import Image
import torch

from marqo.s2_inference.types import List, Dict, ImageType
from marqo.s2_inference.s2_inference import clear_loaded_models

from marqo.s2_inference.processing.DINO_utils import (
    _load_DINO_model,
    _get_DINO_transform,
    DINO_inference,
    _rescale_image,
    attention_to_bboxs
)


class TestImageUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.device = 'cpu'
        self.test_image_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'
        self.test_image = Image.open(requests.get(self.test_image_name, stream=True).raw)
        self.size = (224, 224)
    
    def tearDown(self) -> None:
        clear_loaded_models()

    def test_load_transform(self):
        tform = _get_DINO_transform(image_size=self.size)

        out = tform(self.test_image)

        assert out.shape[1:] == self.size

    def test_load_model(self):

        model, tform = _load_DINO_model(arch='vit_small', device=self.device, 
                                        patch_size=16, image_size=self.size)
        
        img = tform(self.test_image)
        with torch.no_grad():
            attentions = model.get_last_selfattention(img.unsqueeze(0).to(self.device))

        assert isinstance(attentions, torch.FloatTensor)

    def test_dino_inference(self):
        model, tform = _load_DINO_model(arch='vit_small', device=self.device, 
                                        patch_size=16, image_size=self.size)
        attentions = DINO_inference(model=model, transform=tform, img=self.test_image,
                        patch_size=16, device=self.device)

        assert len(attentions[0]) > 1
        assert attentions.shape[1:] == self.size

    def test_rescale_image(self):
        _img = np.array(self.test_image)*.9
        img = _rescale_image(_img)

        assert np.sum(np.abs(_img - img)) > 1e-6

        assert _img.max() <= 255
        assert _img.max() > 0, 'the image cannot be empty'
        assert _img.min() >= 0
        assert img.max() == 255

        assert np.sum(np.abs(_rescale_image(img) - img)) < 1e-6

    def test_attention_to_boxes(self):
        
        img = np.zeros((224,224))
        img[100:120, 100:120] = 1
        img[20:40, 20:40] = 1

        bboxes = attention_to_bboxs(img)
        assert len(bboxes) == 2
        assert bboxes[0] == (100, 100, 120, 120)
        assert bboxes[1] == (20, 20, 40, 40)