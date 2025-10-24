import unittest
import os
import requests

import numpy as np
from PIL import Image
import onnxruntime

from marqo.s2_inference.types import ndarray
from marqo.s2_inference.s2_inference import clear_loaded_models

from marqo.s2_inference.processing.yolox_utils import (
    get_default_yolox_model,
    _download_yolox,
    preprocess_yolox,
    load_yolox_onnx,
    _infer_yolox,
    _process_yolox
)


class TestImageUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.device = 'cpu'
        self.test_image_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'
        self.test_image = np.array(Image.open(requests.get(self.test_image_name, stream=True).raw))
        self.size = (384, 384)

    def tearDown(self) -> None:
        clear_loaded_models()


    def test_yolox_config(self):
        params = get_default_yolox_model()
        assert isinstance(params, dict)
        assert 'repo_id' in params
        assert 'filename' in params

    def test_download_yolox(self):
        params = get_default_yolox_model()
        yolox_path = _download_yolox(**params)

        assert isinstance(yolox_path, str)
        assert params['filename'] in yolox_path
        assert os.path.isfile(yolox_path)

    def test_preprocess(self):
        padded_img, r = preprocess_yolox(img=self.test_image, input_size=self.size)

        assert isinstance(padded_img, ndarray)
        assert padded_img.shape[0] == 3
        assert padded_img.shape[1:] == self.size
        assert r > 0 and r < 10

    def test_load_yolox_onnx(self):
        params = get_default_yolox_model()
        yolox_path = _download_yolox(**params)
        session, preprocess = load_yolox_onnx(model_name=yolox_path, device='cpu')

        assert isinstance(session, onnxruntime.InferenceSession)
        assert preprocess is preprocess_yolox


    def test_infer_and_process_yolox(self):
        params = get_default_yolox_model()
        yolox_path = _download_yolox(**params)
        session, preprocess = load_yolox_onnx(model_name=yolox_path, device='cpu')
        opencv_image = np.array(self.test_image)[:, :, ::-1]
        results, ratio = _infer_yolox(session=session, preprocess=preprocess, 
                    opencv_image=opencv_image, input_shape=self.size)
        
        boxes, scores = _process_yolox(output=results, ratio=ratio, size=self.size) 

        assert len(boxes) == len(scores)
        assert boxes.shape[1] == 4
