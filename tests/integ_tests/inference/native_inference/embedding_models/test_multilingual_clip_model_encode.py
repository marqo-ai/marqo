import json
import os
from pathlib import Path

from parameterized import parameterized_class

from integ_tests.inference.inference_test_case import *
from integ_tests.marqo_test import TestImageUrls
from marqo.inference.media_download_and_preprocess.image_download import load_image_from_path
from marqo.inference.native_inference.load_model import load_model, clear_loaded_models
from unittest import mock

MULTILINGUAL_CLIP_TEST_MODELS = [
    "multilingual-clip/XLM-Roberta-Large-Vit-B-32"
]


@parameterized_class([{"model_name": model_name} for model_name in MULTILINGUAL_CLIP_TEST_MODELS])
class TestMultilingualCLIPModelEncode(InferenceTestCase):
    """
    Tests for multilingual CLIP models, on a CPU device.
    """

    model_name: str # A class variable to store the model name that will be populated by the parameterized decorator
    device = "cpu"
    embeddings_test_texts = [
        "hello",
        "This is a test sentence"
    ]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        clear_loaded_models()
        cls.device_patcher.stop()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        current_file = Path(__file__).resolve()
        target_dir = current_file.parent.parent.parent
        json_file = target_dir / "embeddings_reference" / "embeddings_multilingual_clip_python_3_9.json"
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"File {json_file} not found, which is needed to compare embeddings.")

        with open(json_file, 'r') as f:
            cls.multilingual_clip_embeddings_reference = json.load(f)

        # Temporarily set the MARQO_MAX_CPU_MODEL_MEMORY environment variable to 15 to load the
        # multilingual clip model on CPU.
        cls.device_patcher = mock.patch.dict(os.environ, {
            "MARQO_MAX_CPU_MODEL_MEMORY": "15"
        })

        cls.device_patcher.start()


    def setUp(self):
        super().setUp()
        self.model = load_model(
            self.model_name,
            model_properties=self.get_model_properties_from_registry(self.model_name),
            model_auth=None,
            device=self.device
        )
        self.eps = 1e-6

    def test_embeddings_regression(self):
        try:
            self.model_embeddings_reference = self.multilingual_clip_embeddings_reference[self.model_name]
        except KeyError:
            raise KeyError(
                f"Model {self.model_name} not found in the reference embeddings. "
                f"Please run the test to generate the reference embeddings."
            )

        for text in self.embeddings_test_texts:
            with self.subTest(f"Test text: {text}"):
                embeddings_reference = np.array(self.model_embeddings_reference[text]).reshape(-1)
                pipeline_embeddings = self.encode_content_helper(
                    content=[text],
                    model_name=self.model_name,
                    modality=Modality.TEXT,
                    device=self.device,
                    normalize_embeddings=False
                )

                embeddings_difference = self.calculate_embeddings_difference(
                    embeddings_reference, pipeline_embeddings[0]
                )
                self.assertTrue(embeddings_difference < 1e-4, embeddings_reference)

    def test_multilingual_clip_encode_text_normalized(self):
        """
        A test to ensure that the multilingual_clip model generates the same embeddings as the pipeline for text inputs when
        normalize is set to True.
        """
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        tokenized_text = self.model.get_preprocessor().preprocess(texts, modality=Modality.TEXT)
        raw_embeddings = self.model.encode(tokenized_text, modality=Modality.TEXT, normalize=True)
        pipeline_embeddings = self.encode_content_helper(
            content=texts,
            model_name=self.model_name,
            modality=Modality.TEXT,
            device=self.device,
            normalize_embeddings=True
        )

        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(raw_embedding.shape[0], self.model.model_properties.dimensions)
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=True)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=True)

    def test_multilingual_clip_encode_image_normalized(self):
        """
        A test to ensure that the multilingual_clip model generates the same embeddings as the pipeline for image inputs when
        normalize is set to True.
        """
        image_urls = [
            TestImageUrls.IMAGE0.value,
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
        ]

        images = [load_image_from_path(image, media_download_headers=dict()) for image in image_urls]

        preprocessed_images = self.model.get_preprocessor().preprocess(images, modality=Modality.IMAGE)
        raw_embeddings = self.model.encode(preprocessed_images, modality=Modality.IMAGE, normalize=True)

        pipeline_embeddings = self.encode_content_helper(
            content=image_urls,
            model_name=self.model_name,
            modality=Modality.IMAGE,
            device=self.device,
            normalize_embeddings=True
        )
        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(raw_embedding.shape[0], self.model.model_properties.dimensions)
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=True)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=True)


    @patch("marqo.inference.native_inference.embedding_models.multilingual_clip_model.torch.cuda.amp.autocast")
    def test_multilingual_clip_encode_text_not_normalized(self, mock_autocast):
        """
        A test to ensure that the multilingual_clip model generates the same embeddings as the pipeline for text inputs when
        normalize is set to False.
        """
        texts = ['hello', 'big', 'asasasasaaaaaaaaaaaa', '', 'a word. another one!?. #$#.']

        tokenized_text = self.model.get_preprocessor().preprocess(texts, modality=Modality.TEXT)
        raw_embeddings = self.model.encode(tokenized_text, modality=Modality.TEXT, normalize=False)
        pipeline_embeddings = self.encode_content_helper(
            content=texts,
            model_name=self.model_name,
            modality=Modality.TEXT,
            device=self.device,
            normalize_embeddings=False
        )

        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(raw_embedding.shape[0], self.model.model_properties.dimensions)
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=False)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=False)

        mock_autocast.assert_not_called()

    @patch("marqo.inference.native_inference.embedding_models.multilingual_clip_model.torch.cuda.amp.autocast")
    def test_multilingual_clip_encode_image_not_normalized(self, mock_autocast):
        """
        A test to ensure that the multilingual_clip models generates the same embeddings as the pipeline for image
        inputs when normalize is set to False.
        """
        image_urls = [
            TestImageUrls.IMAGE0.value,
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
        ]

        images = [load_image_from_path(image, media_download_headers=dict()) for image in image_urls]

        preprocessed_images = self.model.get_preprocessor().preprocess(images, modality=Modality.IMAGE)
        raw_embeddings = self.model.encode(preprocessed_images, modality=Modality.IMAGE, normalize=False)

        pipeline_embeddings = self.encode_content_helper(
            content=image_urls,
            model_name=self.model_name,
            modality=Modality.IMAGE,
            device=self.device,
            normalize_embeddings=False
        )
        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertEqual(raw_embedding.shape[0], self.model.model_properties.dimensions)
            self.validate_norm(raw_embedding, epsilon=self.eps, normalize=False)
            self.validate_norm(pipeline_embedding, epsilon=self.eps, normalize=False)

        mock_autocast.assert_not_called()