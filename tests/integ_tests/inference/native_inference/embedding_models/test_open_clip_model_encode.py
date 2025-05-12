import json
import os
from pathlib import Path

from parameterized import parameterized_class

from integ_tests.inference.inference_test_case import *
from integ_tests.marqo_test import TestImageUrls
from marqo.inference.media_download_and_preprocess.image_download import load_image_from_path
from marqo.inference.native_inference.load_model import load_model, clear_loaded_models

OPEN_CLIP_TEST_MODELS = [
    'open_clip/RN50/yfcc15m',
    'Marqo/ViT-B-32.laion2b_s34b_b79k',
    'open_clip/ViT-B-32/laion2b_s34b_b79k',
    'open_clip/ViT-B-32/laion400m_e31',
    'open_clip/ViT-B-16/laion2b_s34b_b88k',
    'Marqo/ViT-B-16.laion2b_s34b_b88k',
    'open_clip/convnext_base/laion400m_s13b_b51k',
    'open_clip/convnext_base_w/laion_aesthetic_s13b_b82k',
    'open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k',
    'open_clip/EVA02-B-16/merged2b_s8b_b131k',
    # "open_clip/MobileCLIP-B/datacompdr_lt",
    # "open_clip/MobileCLIP-S1/datacompdr"
]


@parameterized_class([{"model_name": model_name} for model_name in OPEN_CLIP_TEST_MODELS])
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
          pytest -v tests/integ_tests/inference/native_inference/embedding_models/test_open_clip_model_encode.py
    - Be aware that running the full test file will download and load multiple models. This can consume
      significant time and disk space.

    These tests validate:
    - That OpenCLIP models produce consistent and normalized embeddings for both text and image inputs.
    - That the model outputs match the results from the pipeline encode methods.
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
        current_file = Path(__file__).resolve()
        target_dir = current_file.parent.parent.parent
        json_file = target_dir / "embeddings_reference" / "embeddings_open_clip_python_3_8.json"
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"File {json_file} not found, which is needed to compare embeddings.")

        with open(json_file, 'r') as f:
            cls.open_clip_embeddings_reference = json.load(f)

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
            self.model_embeddings_reference = self.open_clip_embeddings_reference[self.model_name]
        except KeyError:
            self.skipTest(reason=f"Model {self.model_name} not found in the embeddings reference file.")

        text_texts = ['hello', 'this is a test sentence. so is this.']
        for text in text_texts:
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

    def test_open_clip_encode_text_normalized(self):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for text inputs when
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


    @patch("marqo.inference.native_inference.embedding_models.open_clip_model.torch.cuda.amp.autocast")
    def test_open_clip_encode_text_not_normalized(self, mock_autocast):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for text inputs when
        normalize is set to False.
        """
        if self.model_name in [
            # This model always normalizes embeddings
            "open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k"
        ]:
            self.skipTest(f"{self.model_name} always outputs normalized embeddings.")

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

    @patch("marqo.inference.native_inference.embedding_models.open_clip_model.torch.cuda.amp.autocast")
    def test_open_clip_encode_image_not_normalized(self, mock_autocast):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for image inputs when
        normalize is set to False.
        """
        if self.model_name in [
            # This model always normalizes embeddings
            "open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k"
        ]:
            self.skipTest(f"{self.model_name} always outputs normalized embeddings.")

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