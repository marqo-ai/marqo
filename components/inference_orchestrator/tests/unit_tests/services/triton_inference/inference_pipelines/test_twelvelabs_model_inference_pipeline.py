import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.triton_inference.embedding_models.twelvelabs.twelvelabs_model import (
    TwelveLabsModel,
)
from inference_orchestrator.services.triton_inference.inference_pipelines.twelvelabs_model_inference_pipeline import (
    TwelveLabsModelInferencePipeline,
)

MODEL_PROPERTIES = {
    "type": "twelvelabs",
    "name": "Marqo/marengo-3.0",
    "dimensions": 512,
    "apiModelName": "marengo3.0",
}


def _make_model() -> TwelveLabsModel:
    """A TwelveLabsModel whose TwelveLabs client is replaced with a mock so no
    network calls and no API key are required."""
    model = TwelveLabsModel(MODEL_PROPERTIES)
    model._client = MagicMock()
    return model


def _mock_embedding_response(vector):
    segment = MagicMock()
    segment.float_ = list(vector)
    response = MagicMock()
    response.text_embedding.segments = [segment]
    response.image_embedding.segments = [segment]
    return response


class TestTwelveLabsModelInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.config = EmbeddingModelConfig(
            model_name="Marqo/marengo-3.0", normalize_embeddings=True
        )

    def test_init_uses_default_api_model_name(self):
        self.assertEqual("marengo3.0", self.model._model_properties.api_model_name)
        self.assertEqual(512, self.model._model_properties.dimensions)

    def test_encode_text_returns_normalised_vectors(self):
        vec = np.arange(512, dtype=np.float32)
        self.model._client.embed.create.return_value = _mock_embedding_response(vec)

        embeddings = self.model.encode(
            inputs=["a red car"], modality=Modality.TEXT, normalize=True
        )

        self.assertEqual(1, len(embeddings))
        self.assertEqual((512,), embeddings[0].shape)
        np.testing.assert_allclose(np.linalg.norm(embeddings[0]), 1.0, rtol=1e-5)
        self.model._client.embed.create.assert_called_once_with(
            model_name="marengo3.0", text="a red car"
        )

    def test_encode_image_uses_image_url(self):
        vec = np.ones(512, dtype=np.float32)
        self.model._client.embed.create.return_value = _mock_embedding_response(vec)

        self.model.encode(
            inputs=["http://example.com/cat.jpg"],
            modality=Modality.IMAGE,
            normalize=False,
        )

        self.model._client.embed.create.assert_called_once_with(
            model_name="marengo3.0", image_url="http://example.com/cat.jpg"
        )

    def test_content_preprocessing_image_passthrough(self):
        request = InferenceRequest(
            contents=["a.jpg", "b.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.config,
            preprocessing_config=ImagePreprocessingConfig(),
        )
        pipeline = TwelveLabsModelInferencePipeline(self.model, request)
        self.assertEqual(
            [[("a.jpg", "a.jpg")], [("b.jpg", "b.jpg")]],
            pipeline._content_preprocessing(),
        )

    def test_collect_valid_content_skips_errors(self):
        request = InferenceRequest(
            contents=["x"],
            modality=Modality.TEXT,
            embedding_model_config=self.config,
            preprocessing_config=TextPreprocessingConfig(),
        )
        pipeline = TwelveLabsModelInferencePipeline(self.model, request)
        preprocessed = [
            [("o1", "c1")],
            InferenceErrorModel(error_message="boom"),
            [("o2", "c2")],
        ]
        self.assertEqual(
            ["c1", "c2"], pipeline._collect_valid_content_to_encode(preprocessed)
        )

    def test_run_pipeline_text_end_to_end(self):
        request = InferenceRequest(
            contents=["one", "two"],
            modality=Modality.TEXT,
            embedding_model_config=self.config,
            preprocessing_config=TextPreprocessingConfig(),
        )
        pipeline = TwelveLabsModelInferencePipeline(self.model, request)

        def fake_create(model_name, text):
            return _mock_embedding_response(np.full(512, len(text), dtype=np.float32))

        self.model._client.embed.create.side_effect = fake_create

        with patch(
            "inference_orchestrator.services.triton_inference.inference_pipelines."
            "twelvelabs_model_inference_pipeline.split_prefix_preprocess_text"
        ) as mock_split:
            mock_split.return_value = [[("one", "one")], [("two", "two")]]
            result = pipeline.run_pipeline()

        self.assertEqual(2, len(result.result))
        self.assertEqual((512,), result.result[0][0][1].shape)

    def test_missing_api_key_raises(self):
        model = TwelveLabsModel(MODEL_PROPERTIES)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                model.load()
        self.assertIn("TWELVELABS_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
