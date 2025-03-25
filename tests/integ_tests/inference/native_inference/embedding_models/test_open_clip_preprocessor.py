import json
import os
from pathlib import Path

import pytest
from parameterized import parameterized_class

from integ_tests.inference.inference_test_case import *
from integ_tests.marqo_test import TestImageUrls
from marqo.inference.media_download_and_preprocess.image_download import load_image_from_path
from marqo.inference.native_inference.load_model import load_model, clear_loaded_models
from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPPreprocessor
from torch import Tensor

OPEN_CLIP_PREPROCESSOR_TEST_MODELS = [
    'open_clip/ViT-B-32/laion2b_s34b_b79k',
]


@parameterized_class([{"model_name": model_name} for model_name in OPEN_CLIP_PREPROCESSOR_TEST_MODELS])
class TestOpenClipModelPreprocessor(InferenceTestCase):
    """
    Tests for OpenCLIPPreprocessor. Currently, we only test one model, but it can be extended to test multiple models
    in the future.
    """

    model_name: str # A class variable to store the model name that will be populated by the parameterized decorator
    device = "cpu"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        clear_loaded_models()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        clear_loaded_models()

    def setUp(self):
        super().setUp()
        self.model = load_model(
            self.model_name,
            model_properties=self.get_model_properties_from_registry(self.model_name),
            model_auth=None,
            device=self.device
        )
        self.preprocessor: OpenCLIPPreprocessor  = self.model.get_preprocessor()

    def test_preprocessor_devices(self):
        self.assertEqual(self.model.device, self.preprocessor.device)

    def test_to_device_called_on_preprocessor_for_text(self):
        """A test to ensure the Tensor is moved to the correct device when self.preprocessor.preprocess is called."""
        test_texts = ["Marqo is awesome!", "Testing OpenCLIP Preprocessor."]
        text_outputs = self.preprocessor.preprocess(test_texts, modality=Modality.TEXT)

        self.assertIsInstance(text_outputs, list)
        self.assertEqual(2, len(text_outputs))

        for text_output in text_outputs:
            self.assertIsInstance(text_output, Tensor)
            self.assertEqual(text_output.device.type, self.device)

    def test_to_device_called_on_preprocessor_for_image(self):
        """A test to ensure the Tensor is moved to the correct device when self.preprocessor.preprocess is called."""
        test_image_path = [TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value]

        test_images = [
            load_image_from_path(image_path, media_download_headers=dict()) for image_path in test_image_path
        ]

        image_outputs = self.preprocessor.preprocess(test_images, modality=Modality.IMAGE)

        # Assertions for image tensors
        self.assertIsInstance(image_outputs, list)
        self.assertEqual(2, len(image_outputs))

        for tensor in image_outputs:
            self.assertIsInstance(tensor, Tensor)
            # Check tensor is on the correct device
            self.assertEqual(tensor.device.type, self.device)


@pytest.mark.largemodel
@parameterized_class([{"model_name": model_name} for model_name in OPEN_CLIP_PREPROCESSOR_TEST_MODELS])
class TestOpenClipModelPreprocessorCuda(InferenceTestCase):
    """
    Tests for OpenCLIPPreprocessor. Currently, we only test one model, but it can be extended to test multiple models
    in the future.
    """

    model_name: str # A class variable to store the model name that will be populated by the parameterized decorator
    device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        clear_loaded_models()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        clear_loaded_models()

    def setUp(self):
        super().setUp()
        self.model = load_model(
            self.model_name,
            model_properties=self.get_model_properties_from_registry(self.model_name),
            model_auth=None,
            device=self.device
        )
        self.preprocessor: OpenCLIPPreprocessor  = self.model.get_preprocessor()

    def test_preprocessor_devices(self):
        self.assertEqual(self.model.device, self.preprocessor.device)

    def test_to_device_called_on_preprocessor_for_text(self):
        """A test to ensure the Tensor is moved to the correct device when self.preprocessor.preprocess is called."""
        test_texts = ["Marqo is awesome!", "Testing OpenCLIP Preprocessor."]
        text_outputs = self.preprocessor.preprocess(test_texts, modality=Modality.TEXT)

        self.assertIsInstance(text_outputs, list)
        self.assertEqual(2, len(text_outputs))

        for text_output in text_outputs:
            self.assertIsInstance(text_output, Tensor)
            self.assertEqual(text_output.device.type, self.device)

    def test_to_device_called_on_preprocessor_for_image(self):
        """A test to ensure the Tensor is moved to the correct device when self.preprocessor.preprocess is called."""
        test_image_path = [TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value]
        test_images = [load_image_from_path(image_path, media_download_headers=dict()) for image_path in test_image_path]

        image_outputs = self.preprocessor.preprocess(test_images, modality=Modality.IMAGE)

        # Assertions for image tensors
        self.assertIsInstance(image_outputs, list)
        self.assertEqual(2, len(image_outputs))

        for tensor in image_outputs:
            self.assertIsInstance(tensor, Tensor)
            # Check tensor is on the correct device
            self.assertEqual(tensor.device.type, self.device)