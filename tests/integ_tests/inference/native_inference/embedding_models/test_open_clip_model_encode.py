import numpy as np
from parameterized import parameterized_class

from integ_tests.inference.inference_test_case import *
from marqo.inference.media_download_and_preprocess.image_download import load_image_from_path
from marqo.inference.native_inference.load_model import load_model
from tests.integ_tests.marqo_test import TestImageUrls

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
class TestOpenClipModelEncoding(InferenceTestCase):
    '''
    This test is for open clip models as they are heavily used in production.
    '''

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
        self.eps = 1e-6

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
            self.assertTrue(np.linalg.norm(raw_embedding) - 1 < self.eps)
            self.assertTrue(np.linalg.norm(pipeline_embedding) -1 < self.eps)

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
            self.assertTrue(np.linalg.norm(raw_embedding) -1 < self.eps, np.linalg.norm(raw_embedding))
            self.assertTrue(np.linalg.norm(pipeline_embedding) -1 < self.eps)

    @patch("marqo.inference.native_inference.embedding_models.open_clip_model.torch.cuda.amp.autocast")
    def test_open_clip_encode_text_not_normalized(self, mock_autocast):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for text inputs when
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

        unnormlised_epsilon = 1e-3

        for i, raw_embedding in enumerate(raw_embeddings):
            pipeline_embedding = pipeline_embeddings[i]
            self.assertEqual(raw_embedding.shape, pipeline_embedding.shape)
            self.assertTrue((raw_embedding - pipeline_embedding < self.eps).all())
            self.assertTrue(np.linalg.norm(raw_embedding) - 1 > unnormlised_epsilon)
            self.assertTrue(np.linalg.norm(pipeline_embedding) -1 > unnormlised_epsilon)

        mock_autocast.assert_not_called()

    @patch("marqo.inference.native_inference.embedding_models.open_clip_model.torch.cuda.amp.autocast")
    def test_open_clip_encode_image_not_normalized(self, mock_autocast):
        """
        A test to ensure that the open clip model generates the same embeddings as the pipeline for image inputs when
        normalize is set to False.
        """
        image_urls = [
            TestImageUrls.IMAGE0.value,
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
        ]
        unnormlised_epsilon = 1e-3

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
            self.assertTrue(np.linalg.norm(raw_embedding) -1 > unnormlised_epsilon)
            self.assertTrue(np.linalg.norm(pipeline_embedding) -1 > unnormlised_epsilon)

        mock_autocast.assert_not_called()