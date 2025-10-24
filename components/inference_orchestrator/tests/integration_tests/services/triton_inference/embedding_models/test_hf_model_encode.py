import json
import os
from pathlib import Path

import numpy as np
from parameterized import parameterized_class

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.triton_inference.model_manager.model_manager import (
    load_model,
)
from tests.integration_tests.test_case import InferenceTestCase

HF_TEST_MODELS = ["hf/e5-base-v2", "hf/e5-small-v2", "hf/all-MiniLM-L6-v2"]


@parameterized_class([{"model_name": model_name} for model_name in HF_TEST_MODELS])
class TestHFModelEncode(InferenceTestCase):
    model_name: str  # A class variable to store the model name that will be populated by the parameterized decorator
    device = "cpu"

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
        json_file = (
            target_dir / "embeddings_reference" / "hf_marqo_2_24_2_embeddings.json"
        )
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"File {json_file} not found, which is needed to compare embeddings."
            )

        with open(json_file, "r") as f:
            cls.hf_embeddings_reference = json.load(f)

    def setUp(self):
        super().setUp()
        self.model = load_model(
            self.model_name,
            model_properties=self.get_model_properties_from_registry(self.model_name),
            model_management_client=self.config.model_management_client,
            triton_client=self.config.triton_client,
        )
        self.eps = 1e-6

    def test_embeddings_regression(self):
        self.model_embeddings_reference = self.hf_embeddings_reference[self.model_name]

        text_texts = list(self.model_embeddings_reference.keys())
        for text in text_texts:
            with self.subTest(f"Test text: {text}"):
                embeddings_reference = np.array(
                    self.model_embeddings_reference[text]
                ).reshape(-1)
                pipeline_embeddings = self.encode_content_helper(
                    content=[text],
                    model_name=self.model_name,
                    modality=Modality.TEXT,
                    normalize_embeddings=False,
                )

                embeddings_difference = self.calculate_embeddings_difference(
                    embeddings_reference, pipeline_embeddings[0]
                )
                self.assertTrue(embeddings_difference < 1e-4, embeddings_reference)

    def test_hf_text_normalized(self):
        """
        A test to ensure that the hf model generates the same embeddings as the pipeline for text inputs when
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

    def test_hf_encode_text_not_normalized(self):
        """
        A test to ensure that the hf model generates the same embeddings as the pipeline for text inputs when
        normalize is set to False.
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
            tokenized_text, modality=Modality.TEXT, normalize=False
        )
        pipeline_embeddings = self.encode_content_helper(
            content=texts,
            model_name=self.model_name,
            modality=Modality.TEXT,
            normalize_embeddings=False,
        )

        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(
                raw_embedding.shape[0], self.model.model_properties.dimensions
            )
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=False)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=False)

    def test_batch_results_are_different(self):
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
