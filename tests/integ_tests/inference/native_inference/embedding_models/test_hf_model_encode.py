import json
import os
from pathlib import Path

from parameterized import parameterized_class

from integ_tests.inference.inference_test_case import *
from marqo.inference.native_inference.load_model import load_model, clear_loaded_models

HF_TEST_MODELS = [
    "hf/e5-base-v2",
    "hf/all-MiniLM-L6-v1",
    "hf/all_datasets_v4_MiniLM-L6",
    "hf/bge-small-en-v1.5",
    "hf/e5-small",
    "hf/e5-base",
    "hf/e5-small-unsupervised",
    "hf/e5-base-unsupervised",
    "hf/e5-base-v2",
    "hf/multilingual-e5-small",
    "hf/multilingual-e5-large",
    "hf/e5-large-v2",
    "hf/e5-small-v2",
    "hf/multilingual-e5-base",
    "hf/bge-small-en-v1.5",
    "hf/bge-base-en-v1.5",
    "hf/snowflake-arctic-embed-m",
    "hf/snowflake-arctic-embed-m-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]


@parameterized_class([{"model_name": model_name} for model_name in HF_TEST_MODELS])
class TestHFModelEncode(InferenceTestCase):

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
        json_file = target_dir / "embeddings_reference" / "embeddings_all_models_python_3_8.json"
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"File {json_file} not found, which is needed to compare embeddings.")

        with open(json_file, 'r') as f:
            cls.hf_embeddings_reference = json.load(f)

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
            self.model_embeddings_reference = self.hf_embeddings_reference[self.model_name]
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

    def test_hf_text_normalized(self):
        """
        A test to ensure that the hf model generates the same embeddings as the pipeline for text inputs when
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

    def test_hf_encode_text_not_normalized(self):
        """
        A test to ensure that the hf model generates the same embeddings as the pipeline for text inputs when
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