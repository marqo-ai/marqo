import unittest

from marqo.s2_inference.processing.pytorch_utils import (
    get_default_rcnn_params,
    load_pytorch,
)
from marqo.s2_inference.s2_inference import clear_loaded_models
from marqo.s2_inference.types import Dict


class TestImageUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def tearDown(self) -> None:
        clear_loaded_models()

    def test_get_rcnn_params(self):
        params = get_default_rcnn_params()
        assert isinstance(params, Dict)

        assert len(params) >= 1

    def test_load_pytorch(self):
        model, preprocess = load_pytorch(model_name="frcnn", device=self.device)

        try:
            model, preprocess = load_pytorch(model_name="frc-nn", device=self.device)
        except Exception as e:
            assert "incorrect model specified" in str(e)
