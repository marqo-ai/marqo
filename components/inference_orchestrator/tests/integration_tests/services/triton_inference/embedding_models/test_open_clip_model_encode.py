import json
import os
from pathlib import Path

import numpy as np
from parameterized import parameterized_class

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.media_download_and_preprocess.image_download import (
    load_image_from_path,
)
from inference_orchestrator.services.triton_inference.model_manager.model_manager import (
    load_model,
)
from tests.integration_tests.test_case import InferenceTestCase, TestImageUrls

OPEN_CLIP_TEST_MODELS = [
    # --- Marqo OpenCLIP models for testing
    "Marqo/marqo-fashionCLIP",
    "Marqo/marqo-fashionSigLIP",
    "Marqo/marqo-ecommerce-embeddings-L",
    "Marqo/marqo-ecommerce-embeddings-B",
    # --- Other OpenCLIP models for testing
    "open_clip/ViT-B-32/laion2b_s34b_b79k",
    "laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    "timm/ViT-L-16-SigLIP2-256",
    "open_clip/ViT-L-16-SigLIP-256/webli",
    "open_clip/ViT-B-16-SigLIP/webli",
    "open_clip/ViT-L-14/laion2b_s32b_b82k",
    "timm/ViT-B-16-SigLIP2-256",
    "open_clip/ViT-L-14/laion400m_e32",
]


@parameterized_class(
    [{"model_name": model_name} for model_name in OPEN_CLIP_TEST_MODELS]
)
class TestOpenClipModelEncode(InferenceTestCase):
    """
    Tests for OpenCLIP models, which are heavily used in production.

    This test class is dynamically generated for each model in the OPEN_CLIP_TEST_MODELS list using
    the @parameterized_class decorator. Each model gets its own dedicated test class at runtime. This ensures
    that all tests are run sequentially for a single model before moving on to the next one, improving efficiency
    and making it easier to identify model-specific failures.

    ⚠️ Note:
    - You won't be able to run this test class or its methods directly via the IDE, as the test classes are
      dynamically created at runtime.
    - To run the tests, execute the entire test file (test_open_clip_model_encode.py) using pytest/unittest.
      Example:
          pytest -v tests/integration_tests/services/triton_inference/test_open_clip_model_encode.py
    - Be aware that running the full test file will download and load multiple models. This can consume
      significant time and disk space.

    These tests validate:
    - That OpenCLIP models produce consistent and normalized embeddings for both text and image inputs.
    - That the model outputs match the results from the pipeline encode methods.
    """

    model_name: str  # A class variable to store the model name that will be populated by the parameterized decorator

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.eject_all_models()
        current_file = Path(__file__).resolve()
        target_dir = current_file.parent.parent
        text_json_file = (
            target_dir
            / "embeddings_reference"
            / "open_clip_text_marqo_2_24_2_embeddings.json"
        )
        image_json_file = (
            target_dir
            / "embeddings_reference"
            / "open_clip_image_marqo_2_24_2_embeddings.json"
        )
        if not os.path.exists(text_json_file):
            raise FileNotFoundError(
                f"File {text_json_file} not found, which is needed to compare embeddings."
            )
        if not os.path.exists(image_json_file):
            raise FileNotFoundError(
                f"File {image_json_file} not found, which is needed to compare embeddings."
            )
        with open(text_json_file, "r") as f:
            cls.open_clip_text_embeddings_reference = json.load(f)
        with open(image_json_file, "r") as f:
            cls.open_clip_image_embeddings_reference = json.load(f)

    def setUp(self):
        super().setUp()
        self.model = load_model(
            self.model_name,
            model_properties=self.get_model_properties_from_registry(self.model_name),
            triton_client=self.config.triton_client,
            model_management_client=self.config.model_management_client,
        )
        self.eps = 1e-6

    def test_embeddings_regression_text(self):
        self.model_embeddings_reference = self.open_clip_text_embeddings_reference[
            self.model_name
        ]

        for text, embeddings in self.model_embeddings_reference.items():
            with self.subTest(f"Test text: {text}"):
                embeddings_reference = np.array(embeddings).reshape(-1)
                pipeline_embeddings = self.encode_content_helper(
                    content=[text],
                    model_name=self.model_name,
                    modality=Modality.TEXT,
                    normalize_embeddings=False,
                )

                embeddings_difference = self.calculate_embeddings_difference(
                    embeddings_reference, pipeline_embeddings[0]
                )
                self.assertTrue(
                    embeddings_difference < 3e-3,
                    f"The embedding difference is {embeddings_difference}.",
                )

    def test_embeddings_regression_image(self):
        self.model_embeddings_reference = self.open_clip_image_embeddings_reference[
            self.model_name
        ]

        for image_url, embeddings in self.model_embeddings_reference.items():
            with self.subTest(f"Test image url: {image_url}"):
                embeddings_reference = np.array(embeddings).reshape(-1)
                pipeline_embeddings = self.encode_content_helper(
                    content=[image_url],
                    model_name=self.model_name,
                    modality=Modality.IMAGE,
                    normalize_embeddings=False,
                )

                embeddings_difference = self.calculate_embeddings_difference(
                    embeddings_reference, pipeline_embeddings[0]
                )
                self.assertTrue(
                    embeddings_difference < 3e-3,
                    f"The embedding difference is {embeddings_difference}.",
                )

    def test_open_clip_encode_text_normalized(self):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for text inputs when
        normalize is set to True.
        """
        texts = [
            "hello",
            "big",
            "asasasasaaaaaaaaaaaa",
            "",
            "a word. another one!?. #$#.",
        ]

        tokenized_text = self.model.get_preprocessor().preprocess(
            texts, modality=Modality.TEXT
        )
        raw_embeddings = self.model.encode(
            tokenized_text, modality=Modality.TEXT, normalize=True
        )
        pipeline_embeddings = self.encode_content_helper(
            content=texts,
            model_name=self.model_name,
            modality=Modality.TEXT,
            normalize_embeddings=True,
        )

        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(
                raw_embedding.shape[0], self.model.model_properties.dimensions
            )
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=True)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=True)

    def test_open_clip_encode_image_normalized(self):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for image inputs when
        normalize is set to True.
        """
        image_urls = [
            TestImageUrls.IMAGE0.value,
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
        ]

        images = [
            load_image_from_path(image, media_download_headers=dict())
            for image in image_urls
        ]

        preprocessed_images = self.model.get_preprocessor().preprocess(
            images, modality=Modality.IMAGE
        )
        raw_embeddings = self.model.encode(
            preprocessed_images, modality=Modality.IMAGE, normalize=True
        )

        pipeline_embeddings = self.encode_content_helper(
            content=image_urls,
            model_name=self.model_name,
            modality=Modality.IMAGE,
            normalize_embeddings=True,
        )
        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(
                raw_embedding.shape[0], self.model.model_properties.dimensions
            )
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=True)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=True)

    def test_batch_results_are_different_text(self):
        """
        There could be a bug where the model returns the same embedding for all inputs in a batch. This is a bug
        in the onnx conversion of some models. This test ensures that the embeddings for different inputs in a batch
        are different.
        """

        inputs = ["hello world", "big world", "small world", "another world"]
        embeddings = self.encode_content_helper(
            content=inputs,
            model_name=self.model_name,
            modality=Modality.TEXT,
            normalize_embeddings=True,
        )

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                with self.subTest(
                    f"Comparing embeddings for inputs {inputs[i]} and {inputs[j]}"
                ):
                    self.assertFalse(
                        np.allclose(embeddings[i], embeddings[j]),
                        f"Embeddings for inputs {inputs[i]} and {inputs[j]} are the same, which is a bug.",
                    )

    def test_batch_results_are_different_image(self):
        """
        There could be a bug where the model returns the same embedding for all inputs in a batch. This is a bug
        in the onnx conversion of some models. This test ensures that the embeddings for different inputs in a batch
        are different.
        """

        inputs = [
            TestImageUrls.IMAGE0.value,
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
            TestImageUrls.IMAGE3.value,
        ]
        embeddings = self.encode_content_helper(
            content=inputs,
            model_name=self.model_name,
            modality=Modality.IMAGE,
            normalize_embeddings=True,
        )

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                with self.subTest(
                    f"Comparing embeddings for inputs {inputs[i]} and {inputs[j]}"
                ):
                    self.assertFalse(
                        np.allclose(embeddings[i], embeddings[j]),
                        f"Embeddings for inputs {inputs[i]} and {inputs[j]} are the same, which is a bug.",
                    )
